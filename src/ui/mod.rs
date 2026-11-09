//! The terminal interface: the transcript, the input, the approval of tool
//! calls, the status line.

mod input;
mod text;

use std::collections::{HashMap, VecDeque};
use std::io::stdout;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender};
use std::time::{Duration, Instant};

use ratatui::crossterm::event::{
    self, DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
    Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers, KeyboardEnhancementFlags,
    MouseEventKind, PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags,
};
use ratatui::crossterm::execute;
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::{DefaultTerminal, Frame};

use crate::agent::{Cmd, Decision, Msg};
use crate::tools::{DiffLine, Kind, Tool, clip};
use input::Input;
use text::{markdown, width, wrap, wrap_plain};

const COMMANDS: [(&str, &str); 10] = [
    ("/help", "keys and commands"),
    ("/clear", "start a new conversation"),
    (
        "/compact",
        "replace the conversation by a summary, to free context",
    ),
    ("/think", "let the model reason before answering"),
    ("/nothink", "answer without reasoning (faster)"),
    ("/mode", "cycle the permission mode (also shift+tab)"),
    (
        "/pilot",
        "pilot mode on or off: the agent plans and directs subagents",
    ),
    ("/context", "show the context use"),
    ("/quit", "exit"),
    ("/exit", "exit"),
];

const HELP_KEYS: &str = "\
enter send · shift+enter, alt+enter, ctrl+j or \\ then enter: new line
esc interrupt · ctrl+c interrupt, clear, or quit · ctrl+d quit
shift+tab permission mode · ctrl+t show reasoning · ctrl+o expand tool output
pgup/pgdn or the mouse wheel scroll · up/down history
!<command> runs a shell command; its output goes with your next message";

const DIM: Style = Style::new().fg(Color::DarkGray);
const OUTPUT_LINES: usize = 5;
const DIFF_LINES: usize = 40;
const MAX_TOOL_OUTPUT: usize = 256 * 1024;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    Ask,
    AcceptEdits,
    AcceptAll,
}

impl Mode {
    fn next(self) -> Self {
        match self {
            Mode::Ask => Mode::AcceptEdits,
            Mode::AcceptEdits => Mode::AcceptAll,
            Mode::AcceptAll => Mode::Ask,
        }
    }

    fn label(self) -> &'static str {
        match self {
            Mode::Ask => "ask before edits and commands",
            Mode::AcceptEdits => "accept edits",
            Mode::AcceptAll => "accept everything",
        }
    }

    fn allows(self, kind: Kind) -> bool {
        matches!(
            (self, kind),
            (_, Kind::Read) | (Mode::AcceptAll, _) | (Mode::AcceptEdits, Kind::Edit)
        )
    }
}

#[derive(Clone, PartialEq)]
enum Phase {
    Loading,
    Idle,
    Prompt,
    Thinking,
    Writing,
    Tool(Option<Tool>),
    Asking,
    Dead,
}

enum ToolState {
    Waiting,
    Asking,
    Running,
    Done { ok: bool, summary: String },
}

/// A call as the transcript shows it
struct ToolRun {
    tool: Option<Tool>,
    title: String,
    diff: Option<Vec<DiffLine>>,
    output: String,
    state: ToolState,
    /// the entry of the task, for a call of a subagent
    parent: Option<usize>,
    /// of a task: the text of the subagent's current turn
    activity: String,
}

impl ToolRun {
    fn kind(&self) -> Kind {
        // a call of a function that doesn't exist does nothing
        self.tool.map_or(Kind::Read, Tool::kind)
    }
}

enum Entry {
    Banner,
    User(String),
    Thinking {
        text: String,
        done: Option<f64>,
        start: Instant,
    },
    Answer(String),
    Tool(ToolRun),
    Shell {
        cmd: String,
        output: Option<(bool, String)>,
    },
    Info(String),
    Error(String),
    Stats(String),
    /// the pilot's view of the project and its tasks
    Plan {
        overview: String,
        tasks: String,
    },
}

pub struct Options {
    pub cwd: PathBuf,
    pub thinking: bool,
    pub accept_all: bool,
    pub mouse: bool,
}

struct App {
    entries: Vec<Entry>,
    /// the entries in the order shown: a subagent's calls after its task's
    /// earlier ones, even while other tasks' calls come in between
    order: Vec<usize>,
    /// the lines of each entry, for a width
    cache: Vec<Option<(usize, Vec<Line<'static>>)>>,
    input: Input,
    cmd: Sender<Cmd>,
    /// to send ourselves the result of `!` commands
    own: Sender<Msg>,
    stop: Arc<AtomicBool>,
    cwd: PathBuf,

    phase: Phase,
    busy_since: Option<Instant>,
    status: String,
    model: String,
    backend: String,
    context: usize,
    used: usize,
    mode: Mode,
    thinking: bool,
    /// the agent only directs subagents
    pilot: bool,
    /// tasks of the pilot's plan done, and in all
    progress: Option<(usize, usize)>,
    show_thinking: bool,
    verbose: bool,
    /// the call being approved
    permission: Option<(usize, Sender<Decision>)>,
    /// calls waiting for approval after it, of subagents working at the
    /// same time
    asks: VecDeque<(usize, Sender<Decision>)>,
    tools: HashMap<usize, usize>,
    queue: VecDeque<String>,

    /// lines from the bottom of the transcript, 0 to follow it
    scroll: usize,
    last_total: usize,
    page: usize,
    quit_armed: Option<Instant>,
    notice: Option<(String, Instant)>,
    quit: bool,
}

pub fn run(
    opts: Options,
    cmd: Sender<Cmd>,
    own: Sender<Msg>,
    msgs: Receiver<Msg>,
    stop: Arc<AtomicBool>,
) -> std::io::Result<()> {
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = execute!(
            stdout(),
            DisableMouseCapture,
            DisableBracketedPaste,
            PopKeyboardEnhancementFlags
        );
        hook(info);
    }));
    let mut terminal = ratatui::init();
    let keyboard = matches!(
        ratatui::crossterm::terminal::supports_keyboard_enhancement(),
        Ok(true)
    );
    let _ = execute!(stdout(), EnableBracketedPaste);
    if opts.mouse {
        let _ = execute!(stdout(), EnableMouseCapture);
    }
    if keyboard {
        let _ = execute!(
            stdout(),
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::DISAMBIGUATE_ESCAPE_CODES)
        );
    }
    let mut app = App::new(opts, cmd, own, stop);
    let result = app.event_loop(&mut terminal, &msgs);
    if keyboard {
        let _ = execute!(stdout(), PopKeyboardEnhancementFlags);
    }
    let _ = execute!(stdout(), DisableMouseCapture, DisableBracketedPaste);
    ratatui::restore();
    result
}

impl App {
    fn new(opts: Options, cmd: Sender<Cmd>, own: Sender<Msg>, stop: Arc<AtomicBool>) -> Self {
        App {
            entries: vec![Entry::Banner],
            order: vec![0],
            cache: vec![None],
            input: Input::default(),
            cmd,
            own,
            stop,
            cwd: opts.cwd,
            phase: Phase::Loading,
            busy_since: None,
            status: "starting".into(),
            model: String::new(),
            backend: String::new(),
            context: 0,
            used: 0,
            mode: if opts.accept_all {
                Mode::AcceptAll
            } else {
                Mode::Ask
            },
            thinking: opts.thinking,
            pilot: false,
            progress: None,
            show_thinking: false,
            verbose: false,
            permission: None,
            asks: VecDeque::new(),
            tools: HashMap::new(),
            queue: VecDeque::new(),
            scroll: 0,
            last_total: 0,
            page: 20,
            quit_armed: None,
            notice: None,
            quit: false,
        }
    }

    fn event_loop(
        &mut self,
        terminal: &mut DefaultTerminal,
        msgs: &Receiver<Msg>,
    ) -> std::io::Result<()> {
        let mut dirty = true;
        let mut last_draw = Instant::now();
        while !self.quit {
            let animating = self.busy();
            if dirty || (animating && last_draw.elapsed() >= Duration::from_millis(100)) {
                terminal.draw(|f| self.draw(f))?;
                last_draw = Instant::now();
                dirty = false;
            }
            if event::poll(Duration::from_millis(30))? {
                // everything pending, then one draw
                loop {
                    match event::read()? {
                        TermEvent::Key(k) => self.key(k),
                        TermEvent::Paste(s) => {
                            if self.permission.is_none() {
                                self.input.insert(&s);
                            }
                        }
                        TermEvent::Mouse(m) => match m.kind {
                            MouseEventKind::ScrollUp => self.scroll += 3,
                            MouseEventKind::ScrollDown => {
                                self.scroll = self.scroll.saturating_sub(3)
                            }
                            _ => continue,
                        },
                        TermEvent::Resize(..) => {}
                        _ => continue,
                    }
                    dirty = true;
                    if !event::poll(Duration::ZERO)? {
                        break;
                    }
                }
            }
            while let Ok(m) = msgs.try_recv() {
                self.message(m);
                dirty = true;
            }
            if let Some((_, t)) = &self.notice
                && t.elapsed() > Duration::from_secs(3)
            {
                self.notice = None;
                dirty = true;
            }
        }
        Ok(())
    }

    fn busy(&self) -> bool {
        !matches!(self.phase, Phase::Idle | Phase::Dead)
    }

    fn push(&mut self, e: Entry) -> usize {
        let i = self.entries.len();
        // a subagent's call goes after its task and the task's other calls
        let at = match &e {
            Entry::Tool(ToolRun {
                parent: Some(p), ..
            }) => self
                .order
                .iter()
                .rposition(|&j| {
                    j == *p || matches!(&self.entries[j], Entry::Tool(t) if t.parent == Some(*p))
                })
                .map_or(self.order.len(), |k| k + 1),
            _ => self.order.len(),
        };
        self.entries.push(e);
        self.cache.push(None);
        self.order.insert(at, i);
        i
    }

    fn touch(&mut self, i: usize) {
        if let Some(c) = self.cache.get_mut(i) {
            *c = None;
        }
    }

    fn touch_all(&mut self) {
        self.cache.iter_mut().for_each(|c| *c = None);
    }

    fn notify(&mut self, s: impl Into<String>) {
        self.notice = Some((s.into(), Instant::now()));
    }

    fn tool(&mut self, id: usize) -> Option<(usize, &mut ToolRun)> {
        let i = *self.tools.get(&id)?;
        match self.entries.get_mut(i) {
            Some(Entry::Tool(t)) => Some((i, t)),
            _ => None,
        }
    }

    fn message(&mut self, m: Msg) {
        match m {
            Msg::Status(s) => {
                self.status = s;
                self.touch(0);
            }
            Msg::Ready {
                model,
                backend,
                context,
            } => {
                self.model = model;
                self.backend = backend;
                self.context = context;
                self.touch(0);
            }
            Msg::Generating => self.phase = Phase::Prompt,
            Msg::PromptDone => {
                self.phase = if self.thinking {
                    Phase::Thinking
                } else {
                    Phase::Writing
                };
            }
            Msg::Thinking(t) => {
                self.phase = Phase::Thinking;
                let last = self.entries.len() - 1;
                if let Some(Entry::Thinking {
                    text, done: None, ..
                }) = self.entries.last_mut()
                {
                    text.push_str(&t);
                    self.touch(last);
                } else {
                    self.push(Entry::Thinking {
                        text: t,
                        done: None,
                        start: Instant::now(),
                    });
                }
            }
            Msg::ThinkingDone => {
                self.phase = Phase::Writing;
                let last = self.entries.len() - 1;
                if let Some(Entry::Thinking { done, start, .. }) = self.entries.last_mut() {
                    *done = Some(start.elapsed().as_secs_f64());
                    self.touch(last);
                }
            }
            Msg::Answer(t) => {
                self.phase = Phase::Writing;
                let last = self.entries.len() - 1;
                if let Some(Entry::Answer(text)) = self.entries.last_mut() {
                    text.push_str(&t);
                    self.touch(last);
                } else {
                    let t = t.trim_start_matches('\n').to_string();
                    if !t.is_empty() {
                        self.push(Entry::Answer(t));
                    }
                }
            }
            Msg::ToolStart {
                id,
                tool,
                title,
                diff,
            } => {
                self.phase = Phase::Tool(tool);
                let i = self.push(Entry::Tool(ToolRun {
                    tool,
                    title,
                    diff,
                    output: String::new(),
                    state: ToolState::Waiting,
                    parent: None,
                    activity: String::new(),
                }));
                self.tools.insert(id, i);
            }
            Msg::Sub { task, msg } => self.sub_message(task, *msg),
            Msg::Plan { overview, tasks } => {
                let marks: Vec<Mark> = tasks.lines().filter_map(|l| task_line(l).0).collect();
                let done = marks.iter().filter(|m| **m == Mark::Done).count();
                self.progress = (!marks.is_empty()).then_some((done, marks.len()));
                self.push(Entry::Plan { overview, tasks });
            }
            Msg::Pilot(on) => {
                self.pilot = on;
                self.progress = None;
                self.touch(0);
            }
            Msg::Permission { id, reply } => {
                let mode = self.mode;
                let Some((i, tool)) = self.tool(id) else {
                    let _ = reply.send(Decision::Deny(None));
                    return;
                };
                if mode.allows(tool.kind()) {
                    tool.state = ToolState::Running;
                    let _ = reply.send(Decision::Allow);
                } else {
                    tool.state = ToolState::Asking;
                    if self.permission.is_some() {
                        self.asks.push_back((id, reply));
                    } else {
                        self.permission = Some((id, reply));
                        self.phase = Phase::Asking;
                        // the change to approve must be in view
                        self.scroll = 0;
                    }
                }
                self.touch(i);
            }
            Msg::ToolOutput { id, text } => {
                if let Some((i, tool)) = self.tool(id) {
                    tool.state = ToolState::Running;
                    if tool.output.len() < MAX_TOOL_OUTPUT {
                        tool.output.push_str(&text);
                    }
                    self.touch(i);
                }
            }
            Msg::ToolDone { id, ok, summary } => {
                if let Some((i, tool)) = self.tool(id) {
                    tool.state = ToolState::Done { ok, summary };
                    if tool.tool == Some(Tool::Task) {
                        // its calls collapse
                        self.touch_all();
                    } else {
                        self.touch(i);
                    }
                }
            }
            Msg::Context(n) => self.used = n,
            Msg::TurnDone { stats, interrupted } => {
                if interrupted {
                    self.push(Entry::Info("interrupted".into()));
                }
                if !stats.is_empty() {
                    self.push(Entry::Stats(stats));
                }
            }
            Msg::Idle => {
                self.phase = Phase::Idle;
                self.busy_since = None;
                if let Some(next) = self.queue.pop_front() {
                    self.submit(next);
                }
            }
            Msg::Info(s) => {
                self.push(Entry::Info(s));
            }
            Msg::Error(s) => {
                self.push(Entry::Error(s));
            }
            Msg::Fatal(s) => {
                self.phase = Phase::Dead;
                self.status = "the model could not be loaded".into();
                self.touch(0);
                self.push(Entry::Error(s));
            }
            Msg::Shell { entry, ok, output } => {
                if let Some(Entry::Shell { cmd, output: out }) = self.entries.get_mut(entry) {
                    let context = format!(
                        "(I ran a shell command, `{cmd}`; its output:)\n```\n{}\n```",
                        clip(&output, 8000)
                    );
                    let _ = self.cmd.send(Cmd::Context(context));
                    *out = Some((ok, output));
                    self.touch(entry);
                }
            }
        }
    }

    /// A message of the subagent doing the task of call `task`: its calls go
    /// under the task, its text is shown as the task's activity
    fn sub_message(&mut self, task: usize, m: Msg) {
        let Some(&parent) = self.tools.get(&task) else {
            return;
        };
        let activity = |app: &mut App, f: &dyn Fn(&mut String)| {
            if let Some(Entry::Tool(t)) = app.entries.get_mut(parent) {
                f(&mut t.activity);
                if t.activity.len() > 4000 {
                    let mut cut = t.activity.len() - 2000;
                    while !t.activity.is_char_boundary(cut) {
                        cut += 1;
                    }
                    t.activity.drain(..cut);
                }
                app.touch(parent);
            }
        };
        match m {
            Msg::ToolStart {
                id,
                tool,
                title,
                diff,
            } => {
                self.phase = Phase::Tool(tool);
                let i = self.push(Entry::Tool(ToolRun {
                    tool,
                    title,
                    diff,
                    output: String::new(),
                    state: ToolState::Waiting,
                    parent: Some(parent),
                    activity: String::new(),
                }));
                self.tools.insert(id, i);
                activity(self, &|a| a.clear());
            }
            m @ (Msg::Permission { .. } | Msg::ToolOutput { .. } | Msg::ToolDone { .. }) => {
                self.message(m)
            }
            Msg::Generating => {
                self.phase = Phase::Prompt;
                activity(self, &|a| a.clear());
            }
            Msg::Thinking(t) => {
                self.phase = Phase::Thinking;
                activity(self, &|a| a.push_str(&t));
            }
            Msg::ThinkingDone => activity(self, &|a| a.clear()),
            Msg::Answer(t) => {
                self.phase = Phase::Writing;
                activity(self, &|a| a.push_str(&t));
            }
            Msg::Info(s) | Msg::Error(s) => activity(self, &|a| *a = s.clone()),
            _ => {}
        }
    }

    /// The next call waiting for approval, once the last is answered: those
    /// the mode now allows run, the first it doesn't is asked; all are denied
    /// when the user stopped
    fn next_permission(&mut self) {
        let stopped = self.stop.load(Ordering::SeqCst);
        while let Some((id, reply)) = self.asks.pop_front() {
            let mode = self.mode;
            let Some((i, tool)) = self.tool(id) else {
                let _ = reply.send(Decision::Deny(None));
                continue;
            };
            if stopped {
                let _ = reply.send(Decision::Deny(None));
            } else if mode.allows(tool.kind()) {
                tool.state = ToolState::Running;
                let _ = reply.send(Decision::Allow);
            } else {
                self.permission = Some((id, reply));
                self.phase = Phase::Asking;
                self.scroll = 0;
                self.touch(i);
                return;
            }
            self.touch(i);
        }
        self.phase = Phase::Tool(None);
    }

    fn interrupt(&mut self) {
        self.stop.store(true, Ordering::SeqCst);
        if let Some((_, reply)) = self.permission.take() {
            let _ = reply.send(Decision::Deny(None));
        }
        for (_, reply) in self.asks.drain(..) {
            let _ = reply.send(Decision::Deny(None));
        }
        // queued messages come back to the input, to edit or send again
        if !self.queue.is_empty() {
            let mut texts: Vec<String> = self.queue.drain(..).collect();
            if !self.input.is_empty() {
                texts.push(self.input.text.clone());
            }
            self.input.set(&texts.join("\n"));
        }
    }

    fn key(&mut self, k: KeyEvent) {
        if k.kind == KeyEventKind::Release {
            return;
        }
        let ctrl = k.modifiers.contains(KeyModifiers::CONTROL);
        let alt = k.modifiers.contains(KeyModifiers::ALT);
        let shift = k.modifiers.contains(KeyModifiers::SHIFT);

        if let Some((id, _)) = &self.permission {
            let id = *id;
            let decision = match k.code {
                KeyCode::Char('y') | KeyCode::Enter => Some(Decision::Allow),
                KeyCode::Char('a') => {
                    let kind = self.tool(id).map(|(_, t)| t.kind());
                    self.mode = match kind {
                        Some(Kind::Edit) if self.mode == Mode::Ask => Mode::AcceptEdits,
                        _ => Mode::AcceptAll,
                    };
                    self.notify(format!("mode: {}", self.mode.label()));
                    Some(Decision::Allow)
                }
                KeyCode::Char('n') => Some(Decision::Deny(None)),
                KeyCode::Esc => {
                    self.stop.store(true, Ordering::SeqCst);
                    Some(Decision::Deny(Some(
                        "No. Stop and wait for my instructions.".into(),
                    )))
                }
                KeyCode::Char('c') if ctrl => {
                    self.stop.store(true, Ordering::SeqCst);
                    Some(Decision::Deny(None))
                }
                _ => None,
            };
            if let Some(d) = decision {
                if let Some((_, reply)) = self.permission.take() {
                    if let Some((i, tool)) = self.tool(id) {
                        tool.state = ToolState::Running;
                        self.touch(i);
                    }
                    let _ = reply.send(d);
                }
                self.next_permission();
            }
            return;
        }

        match k.code {
            KeyCode::Char('c') if ctrl => {
                if self.busy() && self.phase != Phase::Loading {
                    self.interrupt();
                } else if !self.input.is_empty() {
                    self.input.clear();
                } else if self
                    .quit_armed
                    .is_some_and(|t| t.elapsed() < Duration::from_secs(2))
                {
                    self.quit = true;
                } else {
                    self.quit_armed = Some(Instant::now());
                    self.notify("press ctrl+c again to quit");
                }
            }
            KeyCode::Char('d') if ctrl => {
                if self.input.is_empty() {
                    self.quit = true;
                } else {
                    self.input.delete();
                }
            }
            KeyCode::Esc => {
                if self.busy() && self.phase != Phase::Loading {
                    self.interrupt();
                } else if self.scroll > 0 {
                    self.scroll = 0;
                }
            }
            KeyCode::Enter if shift || alt => self.input.insert("\n"),
            KeyCode::Char('j') if ctrl => self.input.insert("\n"),
            KeyCode::Enter => {
                if self.input.text.ends_with('\\') {
                    self.input.backspace();
                    self.input.insert("\n");
                } else {
                    let text = self.input.take();
                    if !text.trim().is_empty() {
                        self.scroll = 0;
                        self.submit(text);
                    }
                }
            }
            KeyCode::Tab => {
                if let Some((name, _)) = self.suggestions().first() {
                    self.input.set(&format!("{name} "));
                }
            }
            KeyCode::BackTab => {
                self.mode = self.mode.next();
                self.notify(format!("mode: {}", self.mode.label()));
            }
            KeyCode::Char('t') if ctrl => {
                self.show_thinking = !self.show_thinking;
                self.touch_all();
            }
            KeyCode::Char('o') if ctrl => {
                self.verbose = !self.verbose;
                self.touch_all();
            }
            KeyCode::PageUp => self.scroll += self.page / 2,
            KeyCode::PageDown => self.scroll = self.scroll.saturating_sub(self.page / 2),
            KeyCode::Up => {
                if !self.input.up() {
                    self.input.history_prev();
                }
            }
            KeyCode::Down => {
                if !self.input.down() {
                    self.input.history_next();
                }
            }
            KeyCode::Left if ctrl || alt => self.input.word_left(),
            KeyCode::Right if ctrl || alt => self.input.word_right(),
            KeyCode::Char('b') if alt => self.input.word_left(),
            KeyCode::Char('f') if alt => self.input.word_right(),
            KeyCode::Left => self.input.left(),
            KeyCode::Right => self.input.right(),
            KeyCode::Home => self.input.home(),
            KeyCode::End => self.input.end(),
            KeyCode::Char('a') if ctrl => self.input.home(),
            KeyCode::Char('e') if ctrl => self.input.end(),
            KeyCode::Char('b') if ctrl => self.input.left(),
            KeyCode::Char('f') if ctrl => self.input.right(),
            KeyCode::Backspace if alt || ctrl => self.input.delete_word(),
            KeyCode::Char('w') if ctrl => self.input.delete_word(),
            KeyCode::Char('h') if ctrl => self.input.backspace(),
            KeyCode::Backspace => self.input.backspace(),
            KeyCode::Delete => self.input.delete(),
            KeyCode::Char('u') if ctrl => self.input.kill_line_before(),
            KeyCode::Char('k') if ctrl => self.input.kill_line_after(),
            KeyCode::Char(c) if !ctrl => {
                let mut buf = [0u8; 4];
                self.input.insert(c.encode_utf8(&mut buf));
            }
            _ => {}
        }
    }

    /// The commands matching what is typed
    fn suggestions(&self) -> Vec<(&'static str, &'static str)> {
        let t = self.input.text.as_str();
        if !t.starts_with('/') || t.contains(char::is_whitespace) {
            return Vec::new();
        }
        COMMANDS
            .iter()
            .copied()
            .filter(|(c, _)| c.starts_with(t) && *c != t)
            .collect()
    }

    fn submit(&mut self, text: String) {
        let trimmed = text.trim();
        if let Some(cmd) = trimmed.strip_prefix('!') {
            self.shell(cmd.trim().to_string());
            return;
        }
        if trimmed.starts_with('/') {
            self.command(trimmed);
            return;
        }
        if self.phase != Phase::Idle {
            if self.phase == Phase::Dead {
                self.notify("no model loaded");
                return;
            }
            self.queue.push_back(text);
            return;
        }
        self.push(Entry::User(trimmed.to_string()));
        self.phase = Phase::Prompt;
        self.busy_since = Some(Instant::now());
        let _ = self.cmd.send(Cmd::Send(text));
    }

    fn command(&mut self, line: &str) {
        let name = line.split_whitespace().next().unwrap_or("");
        let idle = self.phase == Phase::Idle;
        match name {
            "/quit" | "/exit" => self.quit = true,
            "/help" => {
                let mut s = String::from("commands:\n");
                for (c, d) in COMMANDS.iter().take(COMMANDS.len() - 1) {
                    s.push_str(&format!("  {c:<10} {d}\n"));
                }
                s.push_str(HELP_KEYS);
                self.push(Entry::Info(s));
            }
            "/clear" | "/reset" | "/new" if idle => {
                self.entries.truncate(1);
                self.cache.truncate(1);
                self.tools.clear();
                self.scroll = 0;
                let _ = self.cmd.send(Cmd::Clear);
                self.phase = Phase::Prompt;
                self.push(Entry::Info("new conversation".into()));
            }
            "/pilot" if idle => {
                self.entries.truncate(1);
                self.cache.truncate(1);
                self.tools.clear();
                self.scroll = 0;
                self.progress = None;
                self.phase = Phase::Prompt;
                self.busy_since = Some(Instant::now());
                let _ = self.cmd.send(Cmd::Pilot(!self.pilot));
            }
            "/pilot" => self.notify("wait for the answer, or esc to interrupt it"),
            "/compact" if idle => {
                self.phase = Phase::Prompt;
                self.busy_since = Some(Instant::now());
                let _ = self.cmd.send(Cmd::Compact);
            }
            "/clear" | "/reset" | "/new" | "/compact" => {
                self.notify("wait for the answer, or esc to interrupt it")
            }
            "/think" | "/nothink" => {
                self.thinking = name == "/think";
                let _ = self.cmd.send(Cmd::Thinking(self.thinking));
                self.notify(if self.thinking {
                    "reasoning on"
                } else {
                    "reasoning off"
                });
            }
            "/mode" => {
                self.mode = self.mode.next();
                self.notify(format!("mode: {}", self.mode.label()));
            }
            "/context" => {
                let s = format!(
                    "context: {} of {} tokens ({:.0}%) · {} · {}",
                    self.used,
                    self.context,
                    100.0 * self.used as f64 / self.context.max(1) as f64,
                    self.model,
                    self.backend
                );
                self.push(Entry::Info(s));
            }
            _ => self.notify(format!("unknown command {name}, /help lists them")),
        }
    }

    /// A `!` command: run here, its output kept for the next message
    fn shell(&mut self, cmd: String) {
        if cmd.is_empty() {
            return;
        }
        let entry = self.push(Entry::Shell {
            cmd: cmd.clone(),
            output: None,
        });
        let own = self.own.clone();
        let cwd = self.cwd.clone();
        std::thread::spawn(move || {
            let out = crate::tools::shell(&cwd, &cmd);
            let _ = own.send(Msg::Shell {
                entry,
                ok: out.ok,
                output: out.text,
            });
        });
    }

    // drawing

    fn draw(&mut self, f: &mut Frame) {
        let area = f.area();
        let w = area.width as usize;
        let input_cols = w.saturating_sub(4);
        let (rows, cursor) = self.input.layout(input_cols);
        let input_h = if self.permission.is_some() {
            4
        } else {
            rows.len().clamp(1, 10) as u16 + 2
        };
        let suggestions = self.suggestions();
        let queue_h = self.queue.len().min(3) as u16;
        let [transcript, queue_area, spinner, popup, input_area, status] = Layout::vertical([
            Constraint::Min(1),
            Constraint::Length(queue_h),
            Constraint::Length(1),
            Constraint::Length(suggestions.len() as u16),
            Constraint::Length(input_h),
            Constraint::Length(1),
        ])
        .areas(area);

        self.draw_transcript(f, transcript);

        let queued: Vec<Line> = self
            .queue
            .iter()
            .take(3)
            .map(|q| {
                Line::styled(
                    format!(
                        "  ⏵ queued: {}",
                        clip(q.lines().next().unwrap_or(""), w.saturating_sub(14))
                    ),
                    DIM,
                )
            })
            .collect();
        f.render_widget(Paragraph::new(queued), queue_area);

        f.render_widget(Paragraph::new(self.spinner_line()), spinner);

        let pop: Vec<Line> = suggestions
            .iter()
            .map(|(c, d)| {
                Line::from(vec![
                    Span::styled(format!("  {c:<10} "), Style::new().fg(Color::Cyan)),
                    Span::styled(*d, DIM),
                ])
            })
            .collect();
        f.render_widget(Paragraph::new(pop), popup);

        if self.permission.is_some() {
            self.draw_permission(f, input_area);
        } else {
            let block = Block::new()
                .borders(Borders::TOP | Borders::BOTTOM)
                .border_style(DIM);
            let inner = block.inner(input_area);
            f.render_widget(block, input_area);
            let visible = inner.height as usize;
            let first = cursor.0.saturating_sub(visible.saturating_sub(1));
            let lines: Vec<Line> = rows
                .iter()
                .enumerate()
                .skip(first)
                .take(visible)
                .map(|(i, r)| {
                    let p = if i == 0 {
                        Span::styled(
                            "› ",
                            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        )
                    } else {
                        Span::raw("  ")
                    };
                    Line::from(vec![p, Span::raw(r.clone())])
                })
                .collect();
            let placeholder = self.input.is_empty() && self.entries.len() <= 1;
            if placeholder {
                f.render_widget(
                    Paragraph::new(Line::from(vec![
                        Span::styled(
                            "› ",
                            Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                        ),
                        Span::styled("ask for a change, a fix, an explanation… (/help)", DIM),
                    ])),
                    inner,
                );
            } else {
                f.render_widget(Paragraph::new(lines), inner);
            }
            f.set_cursor_position((
                inner.x + 2 + cursor.1 as u16,
                inner.y + (cursor.0 - first) as u16,
            ));
        }

        f.render_widget(Paragraph::new(self.status_line(w)), status);
    }

    fn spinner_line(&self) -> Line<'static> {
        if let Some((n, _)) = &self.notice {
            return Line::styled(format!("  {n}"), Style::new().fg(Color::Yellow));
        }
        let what = match &self.phase {
            Phase::Idle | Phase::Dead => return Line::raw(""),
            Phase::Loading => self.status.clone(),
            Phase::Prompt => "Reading".into(),
            Phase::Thinking => "Thinking".into(),
            Phase::Writing => "Writing".into(),
            Phase::Tool(None) => "Working".into(),
            Phase::Tool(Some(t)) => format!("Running {}", t.name()),
            Phase::Asking => "Waiting for your approval".into(),
        };
        const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
        let t = Instant::now().duration_since(self.busy_since.unwrap_or_else(Instant::now));
        let ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_millis());
        let frame = FRAMES[(ms / 100) as usize % FRAMES.len()];
        let mut spans = vec![
            Span::styled(format!("  {frame} "), Style::new().fg(Color::Magenta)),
            Span::styled(format!("{what}…"), Style::new().fg(Color::Magenta)),
        ];
        if self.busy_since.is_some() {
            spans.push(Span::styled(
                format!(" {}s · esc to interrupt", t.as_secs()),
                DIM,
            ));
        }
        Line::from(spans)
    }

    fn status_line(&self, w: usize) -> Line<'static> {
        let mode_style = match self.mode {
            Mode::Ask => DIM,
            Mode::AcceptEdits => Style::new().fg(Color::Yellow),
            Mode::AcceptAll => Style::new().fg(Color::Red),
        };
        let pct = 100.0 * self.used as f64 / self.context.max(1) as f64;
        let ctx_style = if pct > 80.0 {
            Style::new().fg(Color::Red)
        } else if pct > 60.0 {
            Style::new().fg(Color::Yellow)
        } else {
            DIM
        };
        let ctx = Span::styled(
            format!(
                "ctx {} / {} ({pct:.0}%)  ",
                kilo(self.used),
                kilo(self.context)
            ),
            ctx_style,
        );
        let model = Span::styled(
            format!(
                "{}{} · ",
                if self.model.is_empty() {
                    "loading"
                } else {
                    &self.model
                },
                if self.thinking { " · reasoning" } else { "" }
            ),
            DIM,
        );
        let pilot = Span::styled(
            match (self.pilot, self.progress) {
                (false, _) => String::new(),
                (true, Some((done, all))) => format!("pilot · plan {done}/{all} · "),
                (true, None) => "pilot · ".into(),
            },
            Style::new().fg(Color::Magenta),
        );
        let mode = Span::styled(format!("  ⏵ {}", self.mode.label()), mode_style);
        let hint = Span::styled(" (shift+tab)", DIM);
        // what fits, dropping the least useful first
        let candidates = [
            (
                vec![mode.clone(), hint.clone()],
                vec![pilot.clone(), model.clone(), ctx.clone()],
            ),
            (vec![mode.clone()], vec![pilot.clone(), model, ctx.clone()]),
            (vec![mode.clone()], vec![pilot.clone(), ctx.clone()]),
            (vec![], vec![pilot, ctx]),
        ];
        let span_w = |v: &[Span]| v.iter().map(|s| width(&s.content)).sum::<usize>();
        let (left, right) = candidates
            .iter()
            .find(|(l, r)| span_w(l) + span_w(r) <= w)
            .cloned()
            .unwrap_or_default();
        let mut spans = left.clone();
        spans.push(Span::raw(
            " ".repeat(w.saturating_sub(span_w(&left) + span_w(&right))),
        ));
        spans.extend(right);
        Line::from(spans)
    }

    fn draw_permission(&self, f: &mut Frame, area: Rect) {
        let Some((id, _)) = &self.permission else {
            return;
        };
        let Some(Entry::Tool(tool)) = self.tools.get(id).and_then(|&i| self.entries.get(i)) else {
            return;
        };
        let (question, always) = match tool.kind() {
            Kind::Exec => ("Run this command?", "yes, and don't ask again for commands"),
            _ => ("Make this change?", "yes, and accept edits from now on"),
        };
        let block = Block::bordered()
            .padding(ratatui::widgets::Padding::horizontal(1))
            .border_style(Style::new().fg(Color::Yellow))
            .title(Span::styled(
                format!(" {question} "),
                Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(area);
        f.render_widget(block, area);
        let key = |k: &str| {
            Span::styled(
                k.to_string(),
                Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
            )
        };
        let lines = vec![
            Line::from(Span::styled(
                clip(
                    tool.title.lines().next().unwrap_or(""),
                    area.width.saturating_sub(4) as usize,
                ),
                Style::new().add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                key("y"),
                Span::raw(" yes  "),
                key("a"),
                Span::raw(format!(" {always}  ")),
                key("n"),
                Span::raw(" no  "),
                key("esc"),
                Span::raw(" no, and stop"),
            ]),
        ];
        f.render_widget(Paragraph::new(lines), inner);
    }

    fn draw_transcript(&mut self, f: &mut Frame, area: Rect) {
        let w = area.width as usize;
        let h = area.height as usize;
        self.page = h;
        for i in 0..self.entries.len() {
            if !matches!(&self.cache[i], Some((cw, _)) if *cw == w) {
                let lines = self.render_entry(i, w);
                self.cache[i] = Some((w, lines));
            }
        }
        let total: usize = self
            .cache
            .iter()
            .map(|c| c.as_ref().map_or(0, |(_, l)| l.len()))
            .sum();
        // keep the view in place while scrolled back
        if self.scroll > 0 && total > self.last_total {
            self.scroll += total - self.last_total;
        }
        self.last_total = total;
        self.scroll = self.scroll.min(total.saturating_sub(h));
        let end = total - self.scroll;
        let start = end.saturating_sub(h);
        let mut lines = Vec::with_capacity(h);
        let mut pos = 0;
        for c in self.order.iter().filter_map(|&i| self.cache[i].as_ref()) {
            let n = c.1.len();
            if pos + n > start && pos < end {
                let a = start.saturating_sub(pos);
                let b = (end - pos).min(n);
                lines.extend(c.1[a..b].iter().cloned());
            }
            pos += n;
            if pos >= end {
                break;
            }
        }
        f.render_widget(Paragraph::new(lines), area);
        if self.scroll > 0 {
            let tag = format!(" ↓ {} more lines (esc) ", self.scroll);
            let tw = width(&tag) as u16;
            if area.width > tw {
                let r = Rect::new(
                    area.x + area.width - tw - 1,
                    area.y + area.height - 1,
                    tw,
                    1,
                );
                f.render_widget(
                    Paragraph::new(Span::styled(
                        tag,
                        Style::new().fg(Color::Black).bg(Color::Yellow),
                    )),
                    r,
                );
            }
        }
    }

    /// The lines of an entry, with the blank line before it
    fn render_entry(&self, i: usize, w: usize) -> Vec<Line<'static>> {
        let mut out = Vec::new();
        let entry = &self.entries[i];
        // a subagent's call: indented under its task, hidden once the task is done
        if let Entry::Tool(t) = entry
            && let Some(p) = t.parent
        {
            let task_done = matches!(
                self.entries.get(p),
                Some(Entry::Tool(ToolRun {
                    state: ToolState::Done { .. },
                    ..
                }))
            );
            if task_done && !self.verbose {
                return out;
            }
            self.render_tool(t, w.saturating_sub(4), &mut out);
            return out
                .into_iter()
                .map(|l| {
                    let mut spans = vec![Span::raw("    ")];
                    spans.extend(l.spans);
                    Line::from(spans)
                })
                .collect();
        }
        if i > 0 && !matches!(entry, Entry::Stats(_)) {
            out.push(Line::raw(""));
        }
        let indent = [Span::raw("  ")];
        match entry {
            Entry::Banner => {
                let accent = Style::new().fg(Color::Magenta).add_modifier(Modifier::BOLD);
                out.push(Line::from(vec![
                    Span::styled(" ✻ vcode", accent),
                    Span::styled("  a coding agent on your machine, running on biv", DIM),
                ]));
                let model = if self.model.is_empty() {
                    format!(
                        "   {}",
                        if self.status.is_empty() {
                            "loading"
                        } else {
                            &self.status
                        }
                    )
                } else {
                    format!(
                        "   {} · {} · {} tokens of context",
                        self.model, self.backend, self.context
                    )
                };
                out.extend(wrap_plain(&model, DIM, w, &[], &[Span::raw("     ")]));
                out.push(Line::styled(format!("   {}", self.cwd.display()), DIM));
                if self.pilot {
                    out.push(Line::styled(
                        "   pilot mode: the agent plans and directs subagents (/pilot to leave)",
                        Style::new().fg(Color::Magenta),
                    ));
                }
                out.push(Line::styled("   /help for commands · esc interrupts · shift+tab changes what needs approval", DIM));
            }
            Entry::User(t) => {
                let style = Style::new().add_modifier(Modifier::BOLD);
                out.extend(wrap_plain(
                    t,
                    style,
                    w,
                    &[Span::styled(
                        "› ",
                        Style::new().fg(Color::Cyan).add_modifier(Modifier::BOLD),
                    )],
                    &indent,
                ));
            }
            Entry::Thinking { text, done, .. } => {
                let style = DIM.add_modifier(Modifier::ITALIC);
                let text = text.trim();
                match done {
                    Some(secs) if !self.show_thinking => {
                        out.push(Line::styled(
                            format!("∴ Thought for {secs:.0}s (ctrl+t to show)"),
                            style,
                        ));
                    }
                    _ => {
                        let lines =
                            wrap_plain(text, style, w, &[Span::styled("∴ ", style)], &indent);
                        if self.show_thinking || lines.len() <= 8 {
                            out.extend(lines);
                        } else {
                            out.push(Line::styled("∴ Thinking… (ctrl+t to show all)", style));
                            let n = lines.len();
                            out.extend(lines.into_iter().skip(n - 6));
                        }
                    }
                }
            }
            Entry::Answer(t) => {
                out.extend(markdown(
                    t.trim_end(),
                    Style::new(),
                    w,
                    &[Span::styled("● ", Style::new().fg(Color::White))],
                    &indent,
                ));
            }
            Entry::Tool(t) => self.render_tool(t, w, &mut out),
            Entry::Shell { cmd, output } => {
                let style = Style::new().fg(Color::Magenta);
                out.extend(wrap_plain(
                    cmd,
                    Style::new().add_modifier(Modifier::BOLD),
                    w,
                    &[Span::styled("! ", style)],
                    &indent,
                ));
                match output {
                    None => out.push(Line::styled("  ⎿  running…", DIM)),
                    Some((ok, text)) => {
                        let lines: Vec<&str> = text.lines().collect();
                        let n = lines.len();
                        let shown = if self.verbose { n } else { n.min(10) };
                        let st = if *ok {
                            DIM
                        } else {
                            Style::new().fg(Color::Red)
                        };
                        for (j, l) in lines.iter().skip(n - shown).enumerate() {
                            let p = if j == 0 { "  ⎿  " } else { "     " };
                            out.extend(wrap_plain(
                                l,
                                st,
                                w,
                                &[Span::styled(p, DIM)],
                                &[Span::raw("     ")],
                            ));
                        }
                        if shown < n {
                            out.push(Line::styled(
                                format!("     … {} earlier lines (ctrl+o)", n - shown),
                                DIM,
                            ));
                        }
                    }
                }
            }
            Entry::Info(t) => {
                out.extend(wrap_plain(
                    t,
                    DIM,
                    w,
                    &[Span::styled("  · ", DIM)],
                    &[Span::raw("    ")],
                ));
            }
            Entry::Error(t) => {
                let red = Style::new().fg(Color::Red);
                out.extend(wrap_plain(
                    t,
                    red,
                    w,
                    &[Span::styled("  ✗ ", red)],
                    &[Span::raw("    ")],
                ));
            }
            Entry::Stats(t) => {
                out.push(Line::styled(
                    format!("  {t}"),
                    Style::new().fg(Color::DarkGray).add_modifier(Modifier::DIM),
                ));
            }
            Entry::Plan { overview, tasks } => self.render_plan(overview, tasks, w, &mut out),
        }
        out
    }

    fn render_tool(&self, t: &ToolRun, w: usize, out: &mut Vec<Line<'static>>) {
        let (dot, dot_style) = match &t.state {
            ToolState::Done { ok: true, .. } => ("⏺ ", Style::new().fg(Color::Green)),
            ToolState::Done { ok: false, .. } => ("⏺ ", Style::new().fg(Color::Red)),
            ToolState::Asking => ("⏺ ", Style::new().fg(Color::Yellow)),
            _ => ("⏺ ", Style::new().fg(Color::Gray)),
        };
        let bold = Style::new().add_modifier(Modifier::BOLD);
        let head = [Span::styled(dot, dot_style)];
        let pad = [Span::raw("  ")];
        if t.tool == Some(Tool::Bash) {
            let mut lines = wrap(
                &[("Bash ".to_string(), bold), (t.title.clone(), Style::new())],
                w,
                &head,
                &pad,
            );
            if !self.verbose && lines.len() > 4 {
                let n = lines.len();
                lines.truncate(3);
                lines.push(Line::styled(format!("  … {} more lines", n - 3), DIM));
            }
            out.extend(lines);
        } else {
            let (verb, rest) = t.title.split_once(' ').unwrap_or((&t.title, ""));
            out.extend(wrap(
                &[(verb.to_string(), bold), (format!(" {rest}"), Style::new())],
                w,
                &head,
                &pad,
            ));
        }

        let branch = |s: String, style: Style| {
            wrap_plain(
                &s,
                style,
                w,
                &[Span::styled("  ⎿  ", DIM)],
                &[Span::raw("     ")],
            )
        };
        let task = t.tool == Some(Tool::Task);
        match &t.state {
            ToolState::Waiting | ToolState::Running if task => {
                let last = t.activity.lines().rev().find(|l| !l.trim().is_empty());
                let line = clip(
                    last.unwrap_or("working…").trim(),
                    w.saturating_sub(8).max(10),
                );
                out.extend(branch(line, DIM.add_modifier(Modifier::ITALIC)));
            }
            ToolState::Done { ok, summary } if task && !self.verbose => {
                let style = if *ok {
                    DIM
                } else {
                    Style::new().fg(Color::Red)
                };
                out.extend(branch(format!("{summary} (ctrl+o shows its calls)"), style));
            }
            ToolState::Waiting => {}
            ToolState::Asking => out.extend(branch(
                "waiting for approval".into(),
                Style::new().fg(Color::Yellow),
            )),
            ToolState::Running => {}
            ToolState::Done { ok, summary } => {
                let style = if *ok {
                    DIM
                } else {
                    Style::new().fg(Color::Red)
                };
                out.extend(branch(summary.clone(), style));
            }
        }
        let denied =
            matches!(&t.state, ToolState::Done { ok: false, summary } if summary == "denied");
        if let Some(diff) = &t.diff
            && !denied
        {
            render_diff(diff, w, self.verbose, out);
        }
        if !t.output.is_empty() {
            let text = crate::tools::strip_ansi(&t.output);
            let lines: Vec<&str> = text.trim_end().lines().collect();
            let n = lines.len();
            let shown = if self.verbose { n } else { n.min(OUTPUT_LINES) };
            if shown < n {
                out.push(Line::styled(
                    format!("     … {} lines above (ctrl+o)", n - shown),
                    DIM,
                ));
            }
            for l in lines.iter().skip(n - shown) {
                let l = if self.verbose {
                    l.to_string()
                } else {
                    clip(l, w.saturating_sub(6).max(10))
                };
                out.extend(wrap_plain(
                    &l,
                    DIM,
                    w,
                    &[Span::raw("     ")],
                    &[Span::raw("     ")],
                ));
            }
        }
    }
}

/// The state of a task of the plan
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mark {
    Todo,
    Doing,
    Done,
    Blocked,
}

/// A line of the plan's tasks: its mark if it is a task, and its text
fn task_line(line: &str) -> (Option<Mark>, &str) {
    let mut t = line.trim();
    for bullet in ["- ", "* ", "+ "] {
        if let Some(rest) = t.strip_prefix(bullet) {
            t = rest.trim_start();
        }
    }
    if let Some((n, rest)) = t.split_once(". ")
        && !n.is_empty()
        && n.chars().all(|c| c.is_ascii_digit())
    {
        t = rest.trim_start();
    }
    for (prefix, mark) in [
        ("[ ]", Mark::Todo),
        ("[~]", Mark::Doing),
        ("[x]", Mark::Done),
        ("[X]", Mark::Done),
        ("[!]", Mark::Blocked),
    ] {
        if let Some(rest) = t.strip_prefix(prefix) {
            return (Some(mark), rest.trim_start());
        }
    }
    (None, line.trim())
}

impl App {
    fn render_plan(&self, overview: &str, tasks: &str, w: usize, out: &mut Vec<Line<'static>>) {
        let bold = Style::new().add_modifier(Modifier::BOLD);
        let marks: Vec<_> = tasks.lines().filter_map(|l| task_line(l).0).collect();
        let done = marks.iter().filter(|m| **m == Mark::Done).count();
        out.push(Line::from(vec![
            Span::styled("⏺ ", Style::new().fg(Color::Magenta)),
            Span::styled("Plan", bold),
            Span::styled(format!("  {done}/{} done", marks.len()), DIM),
        ]));
        let overview = overview.trim();
        if !overview.is_empty() {
            let lines = wrap_plain(
                overview,
                DIM.add_modifier(Modifier::ITALIC),
                w,
                &[Span::styled("  ⎿  ", DIM)],
                &[Span::raw("     ")],
            );
            let n = lines.len();
            let shown = if self.verbose { n } else { n.min(3) };
            out.extend(lines.into_iter().take(shown));
            if shown < n {
                out.push(Line::styled(
                    format!("     … {} more lines (ctrl+o)", n - shown),
                    DIM,
                ));
            }
        }
        for line in tasks.lines().filter(|l| !l.trim().is_empty()) {
            let (mark, text) = task_line(line);
            let (sym, style) = match mark {
                Some(Mark::Done) => ("✓ ", Style::new().fg(Color::Green)),
                Some(Mark::Doing) => (
                    "◐ ",
                    Style::new().fg(Color::Yellow).add_modifier(Modifier::BOLD),
                ),
                Some(Mark::Blocked) => ("✗ ", Style::new().fg(Color::Red)),
                Some(Mark::Todo) => ("○ ", Style::new()),
                None => ("  ", DIM),
            };
            let text_style = match mark {
                Some(Mark::Done) => DIM,
                _ => style,
            };
            out.extend(wrap(
                &[(text.to_string(), text_style)],
                w,
                &[Span::raw("     "), Span::styled(sym, style)],
                &[Span::raw("       ")],
            ));
        }
    }
}

fn render_diff(diff: &[DiffLine], w: usize, verbose: bool, out: &mut Vec<Line<'static>>) {
    let num_w = diff
        .iter()
        .map(|l| match l {
            DiffLine::Same(n, _) | DiffLine::Removed(n, _) | DiffLine::Added(n, _) => {
                n.to_string().len()
            }
            DiffLine::Gap => 1,
        })
        .max()
        .unwrap_or(1);
    let limit = if verbose { usize::MAX } else { DIFF_LINES };
    for l in diff.iter().take(limit) {
        let (n, sign, text, style) = match l {
            DiffLine::Same(n, t) => (n.to_string(), ' ', t, DIM),
            DiffLine::Removed(n, t) => (n.to_string(), '-', t, Style::new().fg(Color::Red)),
            DiffLine::Added(n, t) => (n.to_string(), '+', t, Style::new().fg(Color::Green)),
            DiffLine::Gap => {
                out.push(Line::styled(format!("     {:>num_w$}  ⋮", ""), DIM));
                continue;
            }
        };
        let p = [
            Span::styled(format!("     {n:>num_w$} "), DIM),
            Span::styled(format!("{sign} "), style),
        ];
        let r = [Span::raw(" ".repeat(5 + num_w + 3))];
        out.extend(wrap(&[(text.replace('\t', "    "), style)], w, &p, &r));
    }
    if diff.len() > limit {
        out.push(Line::styled(
            format!("     … {} more lines (ctrl+o)", diff.len() - limit),
            DIM,
        ));
    }
}

fn kilo(n: usize) -> String {
    if n >= 1000 {
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ratatui::backend::TestBackend;
    use ratatui::crossterm::event::KeyEvent;
    use std::sync::mpsc::channel;

    fn screen(app: &mut App, w: u16, h: u16) -> String {
        let mut t = ratatui::Terminal::new(TestBackend::new(w, h)).unwrap();
        t.draw(|f| app.draw(f)).unwrap();
        let buf = t.backend().buffer();
        (0..h)
            .map(|y| {
                (0..w)
                    .map(|x| buf[(x, y)].symbol().to_string())
                    .collect::<String>()
                    .trim_end()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    fn typed(app: &mut App, s: &str) {
        for c in s.chars() {
            app.key(KeyEvent::new(KeyCode::Char(c), KeyModifiers::NONE));
        }
    }

    #[test]
    fn a_session() {
        let (cmd, cmds) = channel();
        let (own, _) = channel();
        let opts = Options {
            cwd: "/work/proj".into(),
            thinking: true,
            accept_all: false,
            mouse: false,
        };
        let mut app = App::new(opts, cmd, own, Arc::new(AtomicBool::new(false)));
        app.message(Msg::Ready {
            model: "Qwen3.6-35B-A3B".into(),
            backend: "metal".into(),
            context: 65536,
        });
        app.message(Msg::Idle);
        typed(&mut app, "fix the tests");
        app.key(KeyEvent::new(KeyCode::Enter, KeyModifiers::NONE));
        assert!(matches!(cmds.try_recv(), Ok(Cmd::Send(t)) if t == "fix the tests"));
        app.message(Msg::Generating);
        app.message(Msg::Thinking("The user wants the tests fixed.".into()));
        app.message(Msg::ThinkingDone);
        app.message(Msg::Answer("Let me run the **tests** first.".into()));
        app.message(Msg::ToolStart {
            id: 0,
            tool: Some(Tool::Bash),
            title: "cargo test".into(),
            diff: None,
        });
        let (reply, answer) = channel();
        app.message(Msg::Permission { id: 0, reply });
        let asking = screen(&mut app, 80, 24);
        println!("{asking}\n{}", "=".repeat(80));
        assert!(asking.contains("Run this command?"));
        app.key(KeyEvent::new(KeyCode::Char('y'), KeyModifiers::NONE));
        assert_eq!(answer.try_recv(), Ok(Decision::Allow));
        app.message(Msg::ToolOutput {
            id: 0,
            text: "running 2 tests\ntest a ... FAILED\n".into(),
        });
        app.message(Msg::ToolDone {
            id: 0,
            ok: false,
            summary: "exit code 101".into(),
        });
        app.message(Msg::ToolStart {
            id: 1,
            tool: Some(Tool::EditFile),
            title: "Edit src/lib.rs".into(),
            diff: Some(crate::tools::diff(
                "a\nb\nc\nlet x = 1;\nd\n",
                "a\nb\nc\nlet x = 2;\nd\n",
            )),
        });
        let (reply, answer) = channel();
        app.message(Msg::Permission { id: 1, reply });
        app.key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        assert_eq!(answer.try_recv(), Ok(Decision::Allow));
        assert!(app.mode == Mode::AcceptEdits);
        app.message(Msg::ToolDone {
            id: 1,
            ok: true,
            summary: "+1 -1".into(),
        });
        app.message(Msg::Answer(
            "Fixed:\n\n- `x` is now 2\n\n```rust\nlet x = 2;\n```".into(),
        ));
        app.message(Msg::TurnDone {
            stats: "prompt 10 tok · answer 20 tok".into(),
            interrupted: false,
        });
        app.message(Msg::Context(3000));
        app.message(Msg::Idle);
        typed(&mut app, "/c");
        let done = screen(&mut app, 80, 30);
        println!("{done}");
        assert!(done.contains("Fixed:"));
        assert!(done.contains("/compact"));
    }

    #[test]
    fn a_task() {
        let (cmd, _cmds) = channel();
        let (own, _) = channel();
        let opts = Options {
            cwd: "/work/proj".into(),
            thinking: false,
            accept_all: false,
            mouse: false,
        };
        let mut app = App::new(opts, cmd, own, Arc::new(AtomicBool::new(false)));
        app.message(Msg::Idle);
        let sub = |m: Msg| Msg::Sub {
            task: 0,
            msg: Box::new(m),
        };
        app.message(Msg::ToolStart {
            id: 0,
            tool: Some(Tool::Task),
            title: "Task find the median code".into(),
            diff: None,
        });
        app.message(sub(Msg::Generating));
        app.message(sub(Msg::ToolStart {
            id: 1,
            tool: Some(Tool::Grep),
            title: "Grep \"median\" in .".into(),
            diff: None,
        }));
        app.message(sub(Msg::ToolDone {
            id: 1,
            ok: true,
            summary: "3 matches in 1 files".into(),
        }));
        app.message(sub(Msg::Generating));
        app.message(sub(Msg::Answer("It is in\nsrc/lib.rs:10".into())));
        let running = screen(&mut app, 60, 16);
        println!("{running}\n{}", "=".repeat(60));
        assert!(running.contains("    ⏺ Grep \"median\""));
        assert!(running.contains("⎿  src/lib.rs:10"));
        // the subagent's text is not an answer of the main agent
        assert!(!app.entries.iter().any(|e| matches!(e, Entry::Answer(_))));

        app.message(Msg::ToolDone {
            id: 0,
            ok: true,
            summary: "1 tool call · 4s".into(),
        });
        let done = screen(&mut app, 60, 16);
        println!("{done}");
        assert!(done.contains("1 tool call · 4s"));
        assert!(!done.contains("Grep"));
        app.verbose = true;
        app.touch_all();
        assert!(screen(&mut app, 60, 16).contains("Grep"));
    }

    #[test]
    fn tasks_at_once() {
        let (cmd, _cmds) = channel();
        let (own, _) = channel();
        let opts = Options {
            cwd: "/work/proj".into(),
            thinking: false,
            accept_all: false,
            mouse: false,
        };
        let mut app = App::new(opts, cmd, own, Arc::new(AtomicBool::new(false)));
        app.message(Msg::Idle);
        let sub = |task: usize, m: Msg| Msg::Sub {
            task,
            msg: Box::new(m),
        };
        let start = |id: usize, tool: Tool, title: &str| Msg::ToolStart {
            id,
            tool: Some(tool),
            title: title.into(),
            diff: None,
        };
        app.message(start(0, Tool::Task, "Verify run the tests"));
        app.message(start(1, Tool::Task, "Review check the diff"));
        // their calls come in mixed
        app.message(sub(0, start(2, Tool::Bash, "cargo test")));
        app.message(sub(1, start(3, Tool::Bash, "git diff")));
        app.message(sub(0, start(4, Tool::Grep, "Grep \"fn median\" in .")));
        let (reply0, answer0) = channel();
        app.message(sub(
            0,
            Msg::Permission {
                id: 2,
                reply: reply0,
            },
        ));
        let (reply1, answer1) = channel();
        app.message(sub(
            1,
            Msg::Permission {
                id: 3,
                reply: reply1,
            },
        ));

        let out = screen(&mut app, 70, 20);
        println!("{out}");
        // each call under its own task
        let at = |s: &str| out.find(s).unwrap_or_else(|| panic!("{s} not shown"));
        assert!(at("Verify run the tests") < at("cargo test"));
        assert!(at("cargo test") < at("fn median"));
        assert!(at("fn median") < at("Review check the diff"));
        assert!(at("Review check the diff") < at("git diff"));

        // asked one after the other; allowing all commands lets the second through
        assert!(matches!(app.permission, Some((2, _))));
        app.key(KeyEvent::new(KeyCode::Char('a'), KeyModifiers::NONE));
        assert_eq!(answer0.try_recv(), Ok(Decision::Allow));
        assert_eq!(answer1.try_recv(), Ok(Decision::Allow));
        assert!(app.permission.is_none() && app.asks.is_empty());

        // an interrupt denies every call waiting
        app.mode = Mode::Ask;
        let (reply0, answer0) = channel();
        let (reply1, answer1) = channel();
        app.message(sub(0, start(5, Tool::Bash, "cargo build")));
        app.message(sub(1, start(6, Tool::Bash, "git log")));
        app.message(sub(
            0,
            Msg::Permission {
                id: 5,
                reply: reply0,
            },
        ));
        app.message(sub(
            1,
            Msg::Permission {
                id: 6,
                reply: reply1,
            },
        ));
        app.interrupt();
        assert_eq!(answer0.try_recv(), Ok(Decision::Deny(None)));
        assert_eq!(answer1.try_recv(), Ok(Decision::Deny(None)));
        assert!(app.permission.is_none() && app.asks.is_empty());
    }

    #[test]
    fn a_plan() {
        assert_eq!(
            task_line("- [x] read the code"),
            (Some(Mark::Done), "read the code")
        );
        assert_eq!(
            task_line("2. [~] change it"),
            (Some(Mark::Doing), "change it")
        );
        assert_eq!(task_line("[!] blocked"), (Some(Mark::Blocked), "blocked"));
        assert_eq!(task_line("notes"), (None, "notes"));

        let (cmd, _cmds) = channel();
        let (own, _) = channel();
        let opts = Options {
            cwd: "/work/proj".into(),
            thinking: false,
            accept_all: false,
            mouse: false,
        };
        let mut app = App::new(opts, cmd, own, Arc::new(AtomicBool::new(false)));
        app.message(Msg::Pilot(true));
        app.message(Msg::Idle);
        app.message(Msg::Plan {
            overview: "A Rust crate of statistics functions in src/lib.rs.".into(),
            tasks: "- [x] explore the crate\n- [~] add variance\n- [ ] verify\n- [ ] document"
                .into(),
        });
        let out = screen(&mut app, 70, 16);
        println!("{out}");
        assert!(out.contains("Plan  1/4 done"));
        assert!(out.contains("◐ add variance"));
        assert!(out.contains("pilot · plan 1/4"));
    }
}

//! The agent: the model and the conversation, on a thread of their own.
//!
//! The thread owns everything biv needs (the mapped GGUF, the engine, the
//! chat), takes [`Cmd`]s from the interface and streams [`Msg`]s back. A turn
//! runs the model, then the tools it calls, until it answers without calling
//! any. Calls that change files or run commands ask the interface first.
//!
//! The `task` tool gives a task to a subagent: a chat of its own on the same
//! engine, which runs its own tool loop; its final answer is the result of
//! the call. Its messages reach the interface wrapped in [`Msg::Sub`].
//!
//! Subagents have roles ([`Role`]: explore, implement, verify...), each with
//! its instructions and tools. The system prompt of each role is processed
//! once and saved, and a task's chat starts from its role's.
//!
//! With [`Config::parallel`] above 1 the engine is served by biv's batch
//! scheduler ([`batch::serve`]), and the tasks called together in one turn
//! run at the same time, each on a thread of its own: the scheduler runs
//! their chats' tokens in shared passes, the weights read once for all of
//! them. Only tasks whose roles don't change files run together; the others
//! run alone.
//!
//! In pilot mode the main agent has no tools but `task` and `update_plan`:
//! it directs subagents and keeps the view of the project and of the plan.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use biv::batch::{self, BatchOptions};
use biv::chat::{Chat, ChatOptions, Event, Prefix, ReasoningEffort, ToolCall, TurnStats};
use biv::engine::Engine;
use biv::model::Model;
use biv::sampler::{Sampler, SamplerConfig};
use biv::tokenizer::Tokenizer;

use crate::roles::Role;
use crate::tools::{self, DiffLine, Env, Kind, Tool};

/// What the agent is started with
#[derive(Clone)]
pub struct Config {
    /// a GGUF path or a Hugging Face cache name
    pub model: String,
    pub cpu: bool,
    pub context: usize,
    /// of a subagent, 0 for no subagents
    pub sub_context: usize,
    /// the main agent only directs subagents
    pub pilot: bool,
    pub max_tokens: usize,
    pub thinking: bool,
    pub effort: ReasoningEffort,
    pub draft: Option<usize>,
    pub lookup: usize,
    pub temperature: Option<f32>,
    pub cwd: PathBuf,
    /// subagents working at the same time at most, their passes batched; 1
    /// for one after the other, on the engine itself
    pub parallel: usize,
}

/// From the interface to the agent
pub enum Cmd {
    /// a message of the user
    Send(String),
    /// output to give the model with the next message, e.g. of a `!` command
    Context(String),
    /// forget the conversation
    Clear,
    /// replace the conversation with a summary of it
    Compact,
    Thinking(bool),
    /// pilot mode on or off, in a new conversation
    Pilot(bool),
}

/// The user's answer to a tool call that needs approval
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Deny(Option<String>),
}

/// From the agent to the interface
pub enum Msg {
    /// loading progress
    Status(String),
    Ready {
        model: String,
        backend: String,
        context: usize,
    },
    /// a turn of the model starts: its prompt is being processed
    Generating,
    PromptDone,
    Thinking(String),
    ThinkingDone,
    Answer(String),
    ToolStart {
        id: usize,
        /// None for a function that doesn't exist
        tool: Option<Tool>,
        title: String,
        diff: Option<Vec<DiffLine>>,
    },
    /// a call needs the user's approval, answered on `reply`
    Permission {
        id: usize,
        reply: Sender<Decision>,
    },
    ToolOutput {
        id: usize,
        text: String,
    },
    ToolDone {
        id: usize,
        ok: bool,
        summary: String,
    },
    /// tokens of context in use
    Context(usize),
    /// the model's turn is over
    TurnDone {
        stats: String,
        interrupted: bool,
    },
    /// the user's request is answered, or given up
    Idle,
    Info(String),
    Error(String),
    /// the model could not be loaded
    Fatal(String),
    /// a `!` command of the user is done (sent by the interface to itself)
    Shell {
        entry: usize,
        ok: bool,
        output: String,
    },
    /// a message of the subagent doing the task of call `task`
    Sub {
        task: usize,
        msg: Box<Msg>,
    },
    /// the pilot's view of the project, and its tasks one per line
    Plan {
        overview: String,
        tasks: String,
    },
    /// pilot mode is on or off
    Pilot(bool),
}

/// Tool calls answered in one request before stopping
const MAX_STEPS: usize = 60;
/// Tasks given in one request by the pilot before stopping
const MAX_PILOT_STEPS: usize = 100;
/// Characters of a subagent's report given to the main agent at most
const MAX_REPORT: usize = 20_000;
/// Context used past which the conversation is compacted before a message
const COMPACT_AT: f64 = 0.8;

const COMPACT_PROMPT: &str = "Stop working on the task for now. Write a summary of our conversation so far, \
to continue from it after the conversation is cleared. Include: what I asked for, the decisions made, the \
files read and changed (with paths and what changed), commands run and their results, what is done and \
what is left to do, and anything else needed to continue. Don't call tools. Be complete but concise.";

pub fn spawn(cfg: Config, rx: Receiver<Cmd>, tx: Sender<Msg>, stop: Arc<AtomicBool>) {
    std::thread::Builder::new()
        .name("agent".into())
        .spawn(move || {
            if let Err(e) = run(cfg, rx, &tx, stop) {
                let _ = tx.send(Msg::Fatal(e));
            }
        })
        .expect("spawn the agent thread");
}

fn make_engine<'m>(model: &'m Model<'m>, cpu: bool) -> Result<Box<dyn Engine + 'm>, String> {
    if cpu {
        return Ok(Box::new(model));
    }
    #[cfg(target_os = "macos")]
    {
        biv::metal::MetalEngine::new(model, 256)
            .map(|e| Box::new(e) as Box<dyn Engine + 'm>)
            .map_err(|e| e.to_string())
    }
    #[cfg(not(target_os = "macos"))]
    {
        Ok(Box::new(model))
    }
}

/// The definitions of `tools`
fn definitions(tools: impl IntoIterator<Item = Tool>) -> Vec<String> {
    tools
        .into_iter()
        .map(|t| t.definition().to_string())
        .collect()
}

/// The options of the main conversation, from those all chats share: in
/// pilot mode or not, with subagents or not, working at the same time or not
fn main_options(
    base: &ChatOptions,
    cwd: &Path,
    pilot: bool,
    subagents: bool,
    parallel: bool,
) -> ChatOptions {
    let (system, tools): (String, Vec<Tool>) = if pilot {
        (
            crate::prompt::pilot(cwd, parallel),
            vec![Tool::Task, Tool::UpdatePlan],
        )
    } else {
        let tools = Tool::ALL
            .into_iter()
            .filter(|&t| t != Tool::UpdatePlan && (subagents || t != Tool::Task))
            .collect();
        (crate::prompt::system(cwd, subagents, parallel), tools)
    };
    ChatOptions {
        system: Some(system),
        tools: definitions(tools),
        ..base.clone()
    }
}

fn run(
    cfg: Config,
    rx: Receiver<Cmd>,
    tx: &Sender<Msg>,
    stop: Arc<AtomicBool>,
) -> Result<(), String> {
    let status = |s: String| {
        let _ = tx.send(Msg::Status(s));
    };
    status(format!("finding {}", cfg.model));
    let path = biv_cli::hf::resolve(&cfg.model)?;
    status(format!(
        "loading {}",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    let gguf = biv_cli::open_gguf(&path).map_err(|e| e.to_string())?;
    let tok = Tokenizer::from_gguf(&gguf).map_err(|e| e.to_string())?;
    let model = Model::load(&gguf).map_err(|e| e.to_string())?;
    let engine = make_engine(&model, cfg.cpu)?;
    let name = gguf
        .metadata
        .general
        .name
        .clone()
        .unwrap_or_else(|| "model".into());

    let sampling = &gguf.metadata.general.sampling;
    let defaults = SamplerConfig::default();
    let sampling = SamplerConfig {
        temperature: cfg
            .temperature
            .or(sampling.temp)
            .unwrap_or(defaults.temperature),
        top_k: sampling.top_k.map(|v| v as usize).unwrap_or(defaults.top_k),
        top_p: sampling.top_p.unwrap_or(defaults.top_p),
        seed: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(1),
        ..defaults
    };

    if cfg.pilot && cfg.sub_context == 0 {
        return Err("pilot mode needs subagents (--sub-ctx > 0)".into());
    }
    let setup = Setup {
        tok: &tok,
        name,
        max_context: model.cfg.context_length,
        sampling,
    };
    if cfg.sub_context > 0 && cfg.parallel > 1 {
        // the scheduler on this thread, the agent on another, with the handle
        batch::serve(engine.as_ref(), &BatchOptions::default(), |h| {
            session(&cfg, &h, Some(&h), setup, rx, tx, stop)
        })
        .map_err(|e| e.to_string())?
    } else {
        session(&cfg, engine.as_ref(), None, setup, rx, tx, stop)
    }
}

/// What the session needs of the loaded model, but the engine
struct Setup<'m> {
    tok: &'m Tokenizer,
    /// of the model
    name: String,
    /// tokens of the model's context, at most
    max_context: usize,
    sampling: SamplerConfig,
}

/// The agent on `engine` until the interface is gone. `shared` is the same
/// engine when subagents can work on it at the same time.
fn session<'m>(
    cfg: &Config,
    engine: &'m dyn Engine,
    shared: Option<&'m (dyn Engine + Sync)>,
    setup: Setup<'m>,
    rx: Receiver<Cmd>,
    tx: &Sender<Msg>,
    stop: Arc<AtomicBool>,
) -> Result<(), String> {
    let tok = setup.tok;
    let subagents = cfg.sub_context > 0;
    let parallel = shared.is_some();
    let base = ChatOptions {
        system: None,
        thinking: cfg.thinking,
        effort: cfg.effort,
        max_tokens: cfg.max_tokens,
        context: cfg.context.min(setup.max_context),
        draft: cfg.draft,
        lookup: cfg.lookup,
        tools: Vec::new(),
        ..Default::default()
    };
    let sub_base = subagents.then(|| ChatOptions {
        context: cfg.sub_context.min(setup.max_context),
        ..base.clone()
    });
    let opts = main_options(&base, &cfg.cwd, cfg.pilot, subagents, parallel);
    let context = opts.context;
    let mut chat = Chat::new(engine, tok, opts).map_err(|e| e.to_string())?;
    let _ = tx.send(Msg::Ready {
        model: setup.name,
        backend: engine.description(),
        context,
    });

    // the system prompt and the tools, processed once for every conversation
    let _ = tx.send(Msg::Status("reading the system prompt".into()));
    let t = Instant::now();
    let prefix = chat.save().map_err(|e| e.to_string())?;
    let _ = tx.send(Msg::Context(chat.n_past()));
    let _ = tx.send(Msg::Status(String::new()));
    let _ = tx.send(Msg::Info(format!(
        "system prompt: {} tokens in {:.1}s",
        prefix.n_past(),
        t.elapsed().as_secs_f64()
    )));
    if parallel {
        let _ = tx.send(Msg::Info(format!(
            "subagents: up to {} at a time, in batched passes",
            cfg.parallel
        )));
    }
    let _ = tx.send(Msg::Pilot(cfg.pilot));
    let _ = tx.send(Msg::Idle);

    let mut prefixes = [None, None];
    prefixes[cfg.pilot as usize] = Some(prefix);
    let mut agent = Agent {
        chat,
        prefixes,
        pilot: cfg.pilot,
        base,
        subagents,
        carry: Vec::new(),
        worker: Worker {
            engine,
            shared,
            parallel: cfg.parallel.max(1),
            tok,
            sampler: Sampler::new(setup.sampling.clone()),
            sampling: setup.sampling,
            tx: tx.clone(),
            env: Env {
                cwd: cfg.cwd.clone(),
                stop: stop.clone(),
            },
            stop,
            ids: Arc::new(AtomicUsize::new(0)),
            sub_base,
            roles: HashMap::new(),
            task: None,
            plan: None,
            pilot: cfg.pilot,
            plan_fresh: false,
            seen: Vec::new(),
        },
    };
    for cmd in rx {
        match cmd {
            Cmd::Send(text) => {
                agent.worker.stop.store(false, Ordering::SeqCst);
                if agent.used() > COMPACT_AT {
                    agent
                        .worker
                        .info("context almost full, compacting the conversation".into());
                    agent.compact();
                }
                agent.request(text);
                let _ = tx.send(Msg::Idle);
            }
            Cmd::Context(text) => agent.carry.push(text),
            Cmd::Clear => {
                agent.clear();
                agent.carry.clear();
                let _ = tx.send(Msg::Idle);
            }
            Cmd::Compact => {
                agent.worker.stop.store(false, Ordering::SeqCst);
                agent.compact();
                let _ = tx.send(Msg::Idle);
            }
            Cmd::Thinking(on) => {
                agent.base.thinking = on;
                agent.chat.opts.thinking = on;
                if let Some(o) = &mut agent.worker.sub_base {
                    o.thinking = on;
                }
            }
            Cmd::Pilot(on) => {
                agent.worker.stop.store(false, Ordering::SeqCst);
                agent.switch(on);
                let _ = tx.send(Msg::Pilot(agent.pilot));
                let _ = tx.send(Msg::Idle);
            }
        }
    }
    Ok(())
}

/// The main conversation
struct Agent<'m> {
    chat: Chat<'m>,
    /// the conversation with only the system prompt, without and with pilot
    /// mode, each made when first used
    prefixes: [Option<Prefix>; 2],
    pilot: bool,
    /// the options all chats share
    base: ChatOptions,
    subagents: bool,
    /// text to put before the next message: a summary, command outputs
    carry: Vec<String>,
    worker: Worker<'m>,
}

/// What runs turns and tools, for the main conversation and the subagents
struct Worker<'m> {
    engine: &'m dyn Engine,
    /// the same engine, for subagents working at the same time on threads of
    /// their own; None when they work one after the other
    shared: Option<&'m (dyn Engine + Sync)>,
    /// subagents working at the same time at most
    parallel: usize,
    tok: &'m Tokenizer,
    /// what the sampler was made with, for the subagents' samplers
    sampling: SamplerConfig,
    sampler: Sampler,
    tx: Sender<Msg>,
    env: Env,
    stop: Arc<AtomicBool>,
    /// ids of the tool calls, the subagents' included
    ids: Arc<AtomicUsize>,
    /// the subagents' options but their system prompt and tools, None
    /// without subagents, and in a subagent
    sub_base: Option<ChatOptions>,
    /// the system prompt of each role, where its tasks start, each made for
    /// the first task of its role
    roles: HashMap<Role, Prefix>,
    /// in a subagent, the call of its task: its messages are tagged with it
    task: Option<usize>,
    /// the pilot's overview and tasks
    plan: Option<(String, String)>,
    /// the main agent is the pilot: its task results get reminders
    pilot: bool,
    /// update_plan was called since the last task ended
    plan_fresh: bool,
    /// the read calls of the current conversation since its last edit or
    /// command, to catch the model repeating itself
    seen: Vec<String>,
}

/// What a subagent's [`Worker`] is made of, but the engine and the
/// tokenizer: sent to the thread of its task
struct Seed {
    tx: Sender<Msg>,
    env: Env,
    ids: Arc<AtomicUsize>,
    sampling: SamplerConfig,
    task: usize,
}

/// A task given, and shown: what the subagent works on
struct Started {
    /// of the call
    id: usize,
    role: Role,
    prompt: String,
    start: Instant,
}

/// The result of a call not run because the user interrupted
const NOT_RUN: &str = "Not run: interrupted by the user.";

/// The result of a read call made again with nothing changed since
const REPEATED: &str = "You already made this exact call, and nothing changed since: its result is \
the same as before. Don't repeat calls. Use what you have; if it is enough, write your answer.";

enum Input {
    User(String),
    Tools(Vec<String>),
}

/// How a conversation's tool loop ended
enum End {
    /// with this answer, without tool calls
    Answered(String),
    /// by the user, with the results of the calls of the last turn if any
    Stopped(Option<Vec<String>>),
    Failed(String),
    /// after too many calls, with the text of the last turn
    TooManySteps(String),
}

impl<'m> Worker<'m> {
    /// `m`, tagged as the subagent's while it works
    fn tagged(task: Option<usize>, m: Msg) -> Msg {
        match task {
            Some(task) => Msg::Sub {
                task,
                msg: Box::new(m),
            },
            None => m,
        }
    }

    fn send(&self, m: Msg) {
        let _ = self.tx.send(Self::tagged(self.task, m));
    }

    fn info(&self, s: String) {
        self.send(Msg::Info(s));
    }

    fn stopped(&self) -> bool {
        self.stop.load(Ordering::SeqCst)
    }

    /// One turn of the model in `chat`, streamed to the interface: its stats
    /// and the text of its answer
    fn generate(
        &mut self,
        chat: &mut Chat<'_>,
        input: Input,
    ) -> Result<(TurnStats, String), String> {
        self.send(Msg::Generating);
        let tx = self.tx.clone();
        let task = self.task;
        let stop = self.stop.clone();
        let t0 = Instant::now();
        let mut t1 = None;
        let mut answer = String::new();
        let on_event = |ev: Event<'_>| {
            let m = match ev {
                Event::PromptDone => {
                    t1 = Some(Instant::now());
                    Msg::PromptDone
                }
                Event::Thinking(t) => Msg::Thinking(t.to_string()),
                Event::ThinkingDone => Msg::ThinkingDone,
                Event::Answer(t) => {
                    answer.push_str(t);
                    Msg::Answer(t.to_string())
                }
                // handled once the turn is over
                Event::ToolCall(_) => return,
            };
            let _ = tx.send(Self::tagged(task, m));
        };
        let should_stop = || stop.load(Ordering::SeqCst);
        let result = match &input {
            Input::User(text) => chat.send(text, &mut self.sampler, on_event, should_stop),
            Input::Tools(results) => {
                chat.send_tool_responses(results, &mut self.sampler, on_event, should_stop)
            }
        };
        if self.task.is_none() {
            self.send(Msg::Context(chat.n_past()));
        }
        let stats = result.map_err(|e| e.to_string())?;
        let t1 = t1.unwrap_or(t0);
        let rate = |n: usize, s: f64| n as f64 / s.max(1e-6);
        let prompt_s = (t1 - t0).as_secs_f64();
        let gen_s = t1.elapsed().as_secs_f64();
        self.send(Msg::TurnDone {
            stats: format!(
                "prompt {} tok {:.0} tok/s · answer {} tok {:.1} tok/s{}",
                stats.prompt_tokens,
                rate(stats.prompt_tokens, prompt_s),
                stats.gen_tokens,
                rate(stats.gen_tokens, gen_s),
                if stats.truncated && !self.stopped() {
                    " · truncated"
                } else {
                    ""
                }
            ),
            interrupted: self.stopped(),
        });
        Ok((stats, answer))
    }

    /// A message to the model of `chat`, then as many tool calls as it takes,
    /// `max_steps` at most: how it ended, and the number of calls
    fn converse(&mut self, chat: &mut Chat<'_>, text: String, max_steps: usize) -> (End, usize) {
        let mut input = Input::User(text);
        let mut steps = 0;
        loop {
            let (stats, answer) = match self.generate(chat, input) {
                Ok(s) => s,
                Err(e) => return (End::Failed(e), steps),
            };
            if self.stopped() {
                return (End::Stopped(None), steps);
            }
            if stats.tool_calls.is_empty() {
                return (End::Answered(answer), steps);
            }
            if steps >= max_steps {
                return (End::TooManySteps(answer), steps);
            }
            steps += stats.tool_calls.len();
            let results = self.answer(&stats.tool_calls);
            if self.stopped() {
                return (End::Stopped(Some(results)), steps);
            }
            input = Input::Tools(results);
        }
    }

    /// The results of a turn's calls, in their order. Tasks called one after
    /// the other whose roles don't change files are given together, and their
    /// subagents work at the same time.
    fn answer(&mut self, calls: &[ToolCall]) -> Vec<String> {
        let mut results = Vec::with_capacity(calls.len());
        let mut rest = calls;
        while !rest.is_empty() {
            if self.stopped() {
                results.extend(rest.iter().map(|_| NOT_RUN.to_string()));
                break;
            }
            let n = self.together(rest);
            if n > 0 {
                results.extend(self.delegate(&rest[..n]));
            } else {
                results.push(self.call(&rest[0]));
            }
            rest = &rest[n.max(1)..];
        }
        results
    }

    /// How many of the first `calls` are tasks to give together: 0 if the
    /// first isn't a task for a subagent, 1 if it is one to give alone
    fn together(&self, calls: &[ToolCall]) -> usize {
        let is_task = |c: &ToolCall| Tool::from_name(&c.name) == Some(Tool::Task);
        if self.sub_base.is_none() || !is_task(&calls[0]) {
            return 0;
        }
        if self.shared.is_none() {
            return 1;
        }
        let reads_only = |c: &ToolCall| {
            is_task(c) && tools::task_request(c).is_ok_and(|(role, ..)| !role.changes_files())
        };
        calls
            .iter()
            .take(self.parallel)
            .take_while(|c| reads_only(c))
            .count()
            .max(1)
    }

    fn next_id(&self) -> usize {
        self.ids.fetch_add(1, Ordering::Relaxed)
    }

    /// Run a call, after asking the user if it needs to: the result for the model
    fn call(&mut self, call: &ToolCall) -> String {
        let tool = Tool::from_name(&call.name);
        if tool == Some(Tool::UpdatePlan) && self.task.is_none() {
            return match tools::plan_request(call) {
                Ok((overview, tasks)) => {
                    self.plan_fresh = true;
                    self.send(Msg::Plan {
                        overview: overview.clone(),
                        tasks: tasks.clone(),
                    });
                    self.plan = Some((overview, tasks));
                    "Plan recorded.".into()
                }
                Err(e) => format!("Error: {e}"),
            };
        }
        // the same read again, with no edit nor command since, is the model looping
        let key = format!("{}{:?}", call.name, call.arguments);
        match tool.map(Tool::kind) {
            Some(Kind::Read) if self.seen.contains(&key) => return REPEATED.into(),
            Some(Kind::Read) => self.seen.push(key),
            Some(Kind::Edit | Kind::Exec) => self.seen.clear(),
            _ => {}
        }
        let id = self.next_id();
        let plan = match self.env.plan(call) {
            Ok(p) => p,
            Err(e) => {
                self.send(Msg::ToolStart {
                    id,
                    tool,
                    title: describe(call),
                    diff: None,
                });
                self.send(Msg::ToolDone {
                    id,
                    ok: false,
                    summary: tools::clip(&e, 200),
                });
                return format!("Error: {e}");
            }
        };
        self.send(Msg::ToolStart {
            id,
            tool: Some(plan.tool),
            title: plan.title.clone(),
            diff: plan.diff.clone(),
        });
        if matches!(plan.tool.kind(), Kind::Edit | Kind::Exec) {
            let (reply, answer) = channel();
            self.send(Msg::Permission { id, reply });
            match answer.recv().unwrap_or(Decision::Deny(None)) {
                Decision::Allow => {}
                Decision::Deny(why) => {
                    self.send(Msg::ToolDone {
                        id,
                        ok: false,
                        summary: "denied".into(),
                    });
                    return match why {
                        Some(w) => format!("The user denied this call and said: {w}"),
                        None => "The user denied this call. Don't retry it; ask the user how to proceed if you need to.".into(),
                    };
                }
            }
        }
        let tx = self.tx.clone();
        let task = self.task;
        let out = self.env.run(plan, &mut |text| {
            let m = Msg::ToolOutput {
                id,
                text: text.to_string(),
            };
            let _ = tx.send(Self::tagged(task, m));
        });
        self.send(Msg::ToolDone {
            id,
            ok: out.ok,
            summary: out.summary,
        });
        out.text
    }

    /// Calls of `task`, each done by a subagent of its role, at the same time
    /// when there are several: their reports are the results
    fn delegate(&mut self, calls: &[ToolCall]) -> Vec<String> {
        let mut results: Vec<Option<String>> = vec![None; calls.len()];
        let mut ready = Vec::new();
        let mut any = false;
        for (i, call) in calls.iter().enumerate() {
            let task = match self.start_task(call) {
                Ok(t) => t,
                Err(result) => {
                    results[i] = Some(result);
                    continue;
                }
            };
            any = true;
            // the system prompt of each role is processed before the tasks start
            match self.role_prefix(task.role, task.id) {
                Ok(()) => ready.push((i, task)),
                Err(e) => results[i] = Some(self.finish_task(&task, End::Failed(e), 0).1),
            }
        }
        let ends = self.run_tasks(&ready);
        let mut implemented = false;
        for ((i, task), (end, steps)) in ready.iter().zip(ends) {
            let (ok, text) = self.finish_task(task, end, steps);
            implemented |= ok && task.role == Role::Implement;
            results[*i] = Some(text);
        }
        // the subagents may have changed what the agent read
        if any {
            self.seen.clear();
        }
        let mut results: Vec<String> = results.into_iter().map(Option::unwrap_or_default).collect();
        if self.pilot
            && any
            && let Some(last) = results.last_mut()
        {
            *last = self.remind(std::mem::take(last), implemented);
        }
        results
    }

    /// A call of `task` shown, and read: the task, or the result of the call
    /// when it is not one
    fn start_task(&mut self, call: &ToolCall) -> Result<Started, String> {
        let id = self.next_id();
        let (role, description, prompt) = match tools::task_request(call) {
            Ok(r) => r,
            Err(e) => {
                self.send(Msg::ToolStart {
                    id,
                    tool: Some(Tool::Task),
                    title: describe(call),
                    diff: None,
                });
                self.send(Msg::ToolDone {
                    id,
                    ok: false,
                    summary: e.clone(),
                });
                return Err(format!("Error: {e}"));
            }
        };
        self.send(Msg::ToolStart {
            id,
            tool: Some(Tool::Task),
            title: format!("{} {description}", role.label()),
            diff: None,
        });
        Ok(Started {
            id,
            role,
            prompt,
            start: Instant::now(),
        })
    }

    /// The options of a chat of `role`'s subagents
    fn task_options(&self, role: Role) -> Result<ChatOptions, String> {
        let base = self.sub_base.clone().ok_or("no subagents")?;
        Ok(ChatOptions {
            system: Some(crate::prompt::subagent(&self.env.cwd, role)),
            tools: definitions(role.tools()),
            ..base
        })
    }

    /// The system prompt of `role` processed and saved, if this is its first
    /// task; the news goes to the task of call `id`
    fn role_prefix(&mut self, role: Role, id: usize) -> Result<(), String> {
        if self.roles.contains_key(&role) {
            return Ok(());
        }
        let t = Instant::now();
        let mut chat = Chat::new(self.engine, self.tok, self.task_options(role)?)
            .map_err(|e| e.to_string())?;
        let prefix = chat.save().map_err(|e| e.to_string())?;
        let memory = chat.memory().allocated() as f64 / 1e9;
        let _ = self.tx.send(Self::tagged(
            Some(id),
            Msg::Info(format!(
                "{} subagent ready: system prompt {} tokens in {:.1}s, session {memory:.1} GB",
                role.name(),
                prefix.n_past(),
                t.elapsed().as_secs_f64()
            )),
        ));
        self.roles.insert(role, prefix);
        Ok(())
    }

    /// What the worker of the subagent doing the task of call `task` is made of
    fn seed(&self, task: usize) -> Seed {
        Seed {
            tx: self.tx.clone(),
            env: Env {
                cwd: self.env.cwd.clone(),
                stop: self.stop.clone(),
            },
            ids: self.ids.clone(),
            sampling: SamplerConfig {
                // each subagent samples on its own
                seed: self.sampling.seed ^ (task as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                ..self.sampling.clone()
            },
            task,
        }
    }

    /// The tasks done, each on a chat of its own from its role's prefix: on
    /// threads of their own when there are several and the engine is shared,
    /// one after the other otherwise
    fn run_tasks(&self, tasks: &[(usize, Started)]) -> Vec<(End, usize)> {
        let jobs: Vec<_> = tasks
            .iter()
            .map(|(_, t)| {
                let opts = self
                    .task_options(t.role)
                    .expect("tasks only with subagents");
                (self.seed(t.id), opts, &self.roles[&t.role], t)
            })
            .collect();
        let shared = self.shared.filter(|_| jobs.len() > 1);
        let Some(engine) = shared else {
            return jobs
                .into_iter()
                .map(|(seed, opts, prefix, t)| {
                    Worker::of_task(self.engine, self.tok, seed).work(opts, prefix, t)
                })
                .collect();
        };
        let tok = self.tok;
        std::thread::scope(|s| {
            let threads: Vec<_> = jobs
                .into_iter()
                .map(|(seed, opts, prefix, t)| {
                    std::thread::Builder::new()
                        .name(format!("task {}", t.id))
                        .spawn_scoped(s, move || {
                            Worker::of_task(engine, tok, seed).work(opts, prefix, t)
                        })
                })
                .collect();
            threads
                .into_iter()
                .map(|t| match t.map(|t| t.join()) {
                    Ok(Ok(r)) => r,
                    Ok(Err(_)) => (End::Failed("the subagent crashed".into()), 0),
                    Err(e) => (End::Failed(format!("no thread for the subagent: {e}")), 0),
                })
                .collect()
        })
    }

    /// The task's end shown, and its report: whether it succeeded, and the
    /// result of its call
    fn finish_task(&self, task: &Started, end: End, steps: usize) -> (bool, String) {
        let took = format!(
            "{steps} tool call{} · {:.0}s",
            if steps == 1 { "" } else { "s" },
            task.start.elapsed().as_secs_f64()
        );
        let (ok, summary, text) = match end {
            End::Answered(a) if !a.trim().is_empty() => {
                (true, took, tools::clip(a.trim(), MAX_REPORT))
            }
            End::Answered(_) => (
                false,
                format!("no report · {took}"),
                "The subagent finished without a report.".into(),
            ),
            End::Stopped(_) => (
                false,
                format!("interrupted · {took}"),
                "The subagent was interrupted by the user.".into(),
            ),
            End::Failed(e) => (
                false,
                format!("failed: {e}"),
                format!("The subagent failed: {e}"),
            ),
            End::TooManySteps(a) => (
                false,
                format!("stopped · {took}"),
                format!(
                    "The subagent stopped after {steps} tool calls without finishing. Its last message:\n{}",
                    tools::clip(a.trim(), MAX_REPORT)
                ),
            ),
        };
        self.send(Msg::ToolDone {
            id: task.id,
            ok,
            summary,
        });
        (ok, text)
    }

    /// The pilot's instructions after tasks' reports, where it decides what to
    /// do next
    fn remind(&mut self, text: String, implemented: bool) -> String {
        let mut reminders = Vec::new();
        if implemented {
            reminders.push(
                "a change is done only once a verify task has passed: give it one, saying what to check",
            );
        }
        if !std::mem::take(&mut self.plan_fresh) {
            reminders.push("record the state of the tasks with update_plan");
        }
        if reminders.is_empty() {
            text
        } else {
            format!("{text}\n\n(Reminder: {}.)", reminders.join("; "))
        }
    }

    /// The worker of a subagent, which gives no tasks
    fn of_task(engine: &'m dyn Engine, tok: &'m Tokenizer, seed: Seed) -> Worker<'m> {
        Worker {
            engine,
            shared: None,
            parallel: 1,
            tok,
            sampler: Sampler::new(seed.sampling.clone()),
            sampling: seed.sampling,
            tx: seed.tx,
            stop: seed.env.stop.clone(),
            env: seed.env,
            ids: seed.ids,
            sub_base: None,
            roles: HashMap::new(),
            task: Some(seed.task),
            plan: None,
            pilot: false,
            plan_fresh: false,
            seen: Vec::new(),
        }
    }

    /// The subagent's work on `task`, in a chat started from its role's prefix
    fn work(mut self, opts: ChatOptions, prefix: &Prefix, task: &Started) -> (End, usize) {
        let mut chat = match Chat::new(self.engine, self.tok, opts) {
            Ok(c) => c,
            Err(e) => return (End::Failed(e.to_string()), 0),
        };
        if let Err(e) = chat.restore(prefix) {
            return (End::Failed(e.to_string()), 0);
        }
        self.converse(&mut chat, task.prompt.clone(), task.role.max_steps())
    }
}

impl Agent<'_> {
    /// The conversation with only the system prompt, in the current mode
    fn prefix(&self) -> &Prefix {
        self.prefixes[self.pilot as usize]
            .as_ref()
            .expect("the prefix of the current mode is made when switching to it")
    }

    /// The part of the context in use
    fn used(&self) -> f64 {
        self.chat.n_past() as f64 / self.chat.opts.context as f64
    }

    fn clear(&mut self) {
        self.worker.plan = None;
        let prefix = self.prefixes[self.pilot as usize]
            .as_ref()
            .expect("the prefix of the current mode is made when switching to it");
        if let Err(e) = self.chat.restore(prefix) {
            self.worker
                .send(Msg::Error(format!("clearing the conversation: {e}")));
        }
        self.worker.send(Msg::Context(self.chat.n_past()));
    }

    /// Pilot mode on or off: a new conversation, with that mode's system prompt
    fn switch(&mut self, pilot: bool) {
        if pilot && !self.subagents {
            self.worker.send(Msg::Error(
                "pilot mode needs subagents (--sub-ctx > 0)".into(),
            ));
            return;
        }
        let cwd = self.worker.env.cwd.clone();
        let parallel = self.worker.shared.is_some();
        self.carry.clear();
        if pilot != self.pilot {
            self.chat.opts = main_options(&self.base, &cwd, pilot, self.subagents, parallel);
            if self.prefixes[pilot as usize].is_none() {
                self.worker.info("reading the system prompt".into());
                match self.chat.reset().and_then(|()| self.chat.save()) {
                    Ok(p) => self.prefixes[pilot as usize] = Some(p),
                    Err(e) => {
                        self.worker.send(Msg::Error(format!("switching mode: {e}")));
                        self.chat.opts =
                            main_options(&self.base, &cwd, self.pilot, self.subagents, parallel);
                        self.clear();
                        return;
                    }
                }
            }
            self.pilot = pilot;
            self.worker.pilot = pilot;
        }
        self.clear();
        self.worker.info(if self.pilot {
            "pilot mode: the agent plans and directs subagents".into()
        } else {
            "pilot mode off".into()
        });
    }

    /// A message of the user, answered with as many tool calls as it takes
    fn request(&mut self, text: String) {
        let mut text = text;
        if !self.carry.is_empty() {
            let carry = std::mem::take(&mut self.carry).join("\n\n");
            text = format!("{carry}\n\n{text}");
        }
        let max = if self.pilot {
            MAX_PILOT_STEPS
        } else {
            MAX_STEPS
        };
        let (end, steps) = self.worker.converse(&mut self.chat, text, max);
        match end {
            End::Answered(_) | End::Stopped(None) => {}
            // the results go with the next message, with the user's say
            End::Stopped(Some(results)) => self.carry.push(format!(
                "(I interrupted you. Results of your last tool calls:)\n{}",
                results.join("\n\n")
            )),
            End::Failed(e) => self
                .worker
                .send(Msg::Error(format!("{e}; /compact or /clear to go on"))),
            End::TooManySteps(_) => self.worker.info(format!(
                "stopped after {steps} tool calls; send a message to continue"
            )),
        }
    }

    /// Replace the conversation by a summary of it, given with the next
    /// message, with the pilot's plan
    fn compact(&mut self) {
        if self.chat.n_past() <= self.prefix().n_past() {
            self.worker.info("nothing to compact".into());
            return;
        }
        let before = self.chat.n_past();
        let thinking = self.chat.opts.thinking;
        let max_tokens = self.chat.opts.max_tokens;
        self.chat.opts.thinking = false;
        self.chat.opts.max_tokens = 2048;
        self.worker.info("summarizing the conversation".into());
        let mut summary = String::new();
        let tx = self.worker.tx.clone();
        let stop = self.worker.stop.clone();
        let result = self.chat.send(
            COMPACT_PROMPT,
            &mut self.worker.sampler,
            |ev| {
                if let Event::Answer(t) = ev {
                    summary.push_str(t);
                    let _ = tx.send(Msg::Answer(t.to_string()));
                }
            },
            || stop.load(Ordering::SeqCst),
        );
        self.chat.opts.thinking = thinking;
        self.chat.opts.max_tokens = max_tokens;
        let stopped = self.worker.stopped();
        self.worker.send(Msg::TurnDone {
            stats: String::new(),
            interrupted: stopped,
        });
        match result {
            Ok(_) if !stopped && !summary.trim().is_empty() => {
                let plan = self.worker.plan.take();
                self.clear();
                let mut carry = format!(
                    "(Our earlier conversation was cleared to free context. Your summary of it:)\n\n{}",
                    summary.trim()
                );
                if let Some((overview, tasks)) = &plan {
                    carry.push_str(&format!(
                        "\n\n(Your plan, as last recorded with update_plan:)\n{overview}\n\n{tasks}"
                    ));
                }
                carry.push_str("\n\n(Continue from there.)");
                self.carry.insert(0, carry);
                self.worker.plan = plan;
                self.worker.info(format!(
                    "compacted: {before} tokens of conversation replaced by a summary"
                ));
            }
            Ok(_) => self
                .worker
                .info("compaction interrupted; the conversation is unchanged".into()),
            Err(e) => {
                // no room even for the summary: start over
                self.clear();
                self.worker.send(Msg::Error(format!(
                    "could not summarize ({e}); the conversation was cleared"
                )));
            }
        }
    }
}

/// A call that could not be planned, as shown
fn describe(call: &ToolCall) -> String {
    let args: Vec<String> = call
        .arguments
        .iter()
        .map(|(k, v)| format!("{k}={}", tools::clip(v.lines().next().unwrap_or(""), 40)))
        .collect();
    format!("{}({})", call.name, args.join(", "))
}

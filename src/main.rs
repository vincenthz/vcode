//! vcode: a terminal coding agent, with a Qwen model running locally on biv.

mod agent;
mod prompt;
mod roles;
mod tools;
mod trace;
mod ui;

use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::mpsc::channel;

use agent::{Cmd, Config, Decision, Msg};
use biv::chat::ReasoningEffort;

/// The models used without `-m`
const DEFAULT_MODELS: [&str; 3] = [
    "hf.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q8_0",
    "hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1",
    "hf.co/unsloth/Qwen3.8-27B-GGUF:Q8_0",
];

fn default_model() -> String {
    DEFAULT_MODELS
        .iter()
        .find(|m| biv_cli::hf::resolve(m).is_ok())
        .unwrap_or(&DEFAULT_MODELS[1])
        .to_string()
}

const USAGE: &str = "\
usage: vcode [options] [directory]

  -m, --model <path|name>  GGUF file, or a model of the Hugging Face cache
                           [default: hf.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF:Q8_0
                           if downloaded, else hf.co/unsloth/Qwen3.8-27B-GGUF:Q4_1]
  -p, --prompt <text>      do one task without the interface, printing the
                           transcript, then exit
  -y, --yolo               run commands and edit files without asking
  -P, --pilot              pilot mode: the agent only plans and directs
                           subagents (explore, implement, verify, review,
                           formalize, document), keeping the overall view
      --no-think           answer without reasoning first (faster)
      --effort <level>     reasoning effort: low, medium, xhigh [default: medium]
  -c, --ctx <n>            context size in tokens, e.g. 262144 or 256k; /context
                           changes it while running [default: 256k]
      --sub-ctx <n>        context size of the subagents the agent gives tasks
                           to, 0 for no subagents [default: 128k]
      --parallel <n>       subagents working at the same time at most, their
                           passes batched on the engine; 1 for one at a time
                           [default: 4]
  -n, --max-tokens <n>     tokens per answer, reasoning included [default: 16384]
      --cpu                run on the CPU instead of the GPU
  -d, --draft <n>          tokens drafted per step by the MTP head [default: engine's]
  -l, --lookup <n>         tokens proposed per step from the conversation, which
                           speeds up edits that copy code, 0 to disable [default: 7]
      --temp <f>           sampling temperature [default: the model's]
      --no-mouse           leave the mouse to the terminal (text selection)
      --trace <file>       trace the prompts, tokens and tool calls of the
                           agents to a file (also /debug <file>, /debug off)
  -h, --help

The directory, the current one by default, is where the agent works. Its
AGENTS.md (or VCODE.md, CLAUDE.md) goes into the system prompt.";

struct Args {
    cfg: Config,
    prompt: Option<String>,
    yolo: bool,
    mouse: bool,
    trace: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut a = Args {
        cfg: Config {
            model: String::new(),
            cpu: !cfg!(target_os = "macos"),
            context: 256 << 10,
            sub_context: 128 << 10,
            pilot: false,
            max_tokens: 16384,
            thinking: true,
            effort: ReasoningEffort::Medium,
            draft: None,
            lookup: 7,
            temperature: None,
            cwd: std::env::current_dir().map_err(|e| e.to_string())?,
            parallel: 4,
        },
        prompt: None,
        yolo: false,
        mouse: true,
        trace: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut val = || it.next().ok_or_else(|| format!("{flag} needs a value"));
        fn num<T: std::str::FromStr>(flag: &str, v: String) -> Result<T, String> {
            v.parse()
                .map_err(|_| format!("{flag}: invalid value '{v}'"))
        }
        fn tokens(flag: &str, v: String) -> Result<usize, String> {
            agent::parse_tokens(&v).ok_or_else(|| format!("{flag}: invalid size '{v}'"))
        }
        match flag.as_str() {
            "-m" | "--model" => a.cfg.model = val()?,
            "-p" | "--prompt" => a.prompt = Some(val()?),
            "-y" | "--yolo" => a.yolo = true,
            "-P" | "--pilot" => a.cfg.pilot = true,
            "--no-think" => a.cfg.thinking = false,
            "--effort" => {
                let v = val()?;
                a.cfg.effort = ReasoningEffort::parse(&v)
                    .ok_or_else(|| format!("--effort: unknown level '{v}'"))?
            }
            "-c" | "--ctx" => a.cfg.context = tokens(&flag, val()?)?,
            "--sub-ctx" => a.cfg.sub_context = tokens(&flag, val()?)?,
            "--parallel" => {
                a.cfg.parallel = num(&flag, val()?)?;
                if a.cfg.parallel == 0 {
                    return Err("--parallel: at least 1".into());
                }
            }
            "-n" | "--max-tokens" => a.cfg.max_tokens = num(&flag, val()?)?,
            "--cpu" => a.cfg.cpu = true,
            "-d" | "--draft" => a.cfg.draft = Some(num(&flag, val()?)?),
            "-l" | "--lookup" => a.cfg.lookup = num(&flag, val()?)?,
            "--temp" => a.cfg.temperature = Some(num(&flag, val()?)?),
            "--no-mouse" => a.mouse = false,
            "--trace" => a.trace = Some(PathBuf::from(val()?)),
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            f if f.starts_with('-') => return Err(format!("unknown option {f}\n\n{USAGE}")),
            dir => {
                let path = PathBuf::from(dir);
                a.cfg.cwd = path.canonicalize().map_err(|e| format!("{dir}: {e}"))?;
                if !a.cfg.cwd.is_dir() {
                    return Err(format!("{dir}: not a directory"));
                }
            }
        }
    }
    if a.cfg.model.is_empty() {
        a.cfg.model = default_model();
    }
    Ok(a)
}

fn main() {
    let args = match parse_args() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            std::process::exit(2);
        }
    };
    if let Some(path) = &args.trace
        && let Err(e) = trace::start(path)
    {
        eprintln!("error: --trace {e}");
        std::process::exit(2);
    }
    let stop = Arc::new(AtomicBool::new(false));
    let (cmd_tx, cmd_rx) = channel();
    let (msg_tx, msg_rx) = channel();
    agent::spawn(args.cfg.clone(), cmd_rx, msg_tx.clone(), stop.clone());

    if let Some(prompt) = args.prompt {
        let status = headless(&prompt, args.yolo, cmd_tx, msg_rx);
        trace::stop();
        std::process::exit(status);
    }
    let opts = ui::Options {
        cwd: args.cfg.cwd,
        thinking: args.cfg.thinking,
        accept_all: args.yolo,
        mouse: args.mouse,
        theme: ui::themes::Theme::default(),
    };
    let result = ui::run(opts, cmd_tx, msg_tx, msg_rx, stop);
    // the tokens not written yet
    trace::stop();
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
    // the agent thread may be in the middle of an answer: don't wait for it
    std::process::exit(0);
}

/// One task, its transcript on stdout: for scripts and tests
fn headless(
    prompt: &str,
    yolo: bool,
    cmd: std::sync::mpsc::Sender<Cmd>,
    msgs: std::sync::mpsc::Receiver<Msg>,
) -> i32 {
    let mut out = std::io::stdout();
    // the text of each subagent's current turn: subagents working at the
    // same time would mix their streams
    let mut texts: std::collections::HashMap<usize, String> = Default::default();
    let mut tasks = std::collections::HashSet::new();
    let mut sent = false;
    let mut status = 0;
    let mut mid_line = false;
    let nl = |out: &mut std::io::Stdout, mid: &mut bool| {
        if std::mem::take(mid) {
            let _ = writeln!(out);
        }
    };
    for m in msgs {
        // a subagent's messages: its calls indented, its text dimmed
        let (task, m) = match m {
            Msg::Sub { task, msg } => (Some(task), *msg),
            m => (None, m),
        };
        let sub = task.is_some();
        let indent = task.map_or(String::new(), |t| format!("   #{t} | "));
        match m {
            Msg::Thinking(_) | Msg::ThinkingDone if sub => {}
            Msg::Answer(t) if let Some(task) = task => texts.entry(task).or_default().push_str(&t),
            Msg::TurnDone { stats, .. } if let Some(task) = task => {
                nl(&mut out, &mut mid_line);
                let text = texts.remove(&task).unwrap_or_default();
                for line in text.trim().lines() {
                    let _ = writeln!(out, "{indent}\x1b[2m{line}\x1b[0m");
                }
                eprintln!("{indent}[{stats}]");
            }
            Msg::Status(s) if !s.is_empty() => eprintln!("[{s}]"),
            Msg::Ready {
                model,
                backend,
                context,
                sub_context,
            } => eprintln!("[{model} · {backend} · {context} tokens, {sub_context} for subagents]"),
            Msg::Info(s) => {
                nl(&mut out, &mut mid_line);
                eprintln!("{indent}[{s}]");
            }
            Msg::Idle if !sent => {
                sent = true;
                let _ = cmd.send(Cmd::Send(prompt.into()));
            }
            Msg::Idle => break,
            Msg::Thinking(t) => {
                let _ = write!(out, "\x1b[2m{t}\x1b[0m");
                mid_line = !t.ends_with('\n');
            }
            Msg::ThinkingDone => {
                let _ = writeln!(out, "\n");
                mid_line = false;
            }
            Msg::Answer(t) => {
                let _ = write!(out, "{t}");
                mid_line = !t.ends_with('\n');
            }
            Msg::ToolStart {
                id,
                tool,
                title,
                diff,
            } => {
                nl(&mut out, &mut mid_line);
                let name = tool.map_or("unknown function", tools::Tool::name);
                let tag = if tool == Some(tools::Tool::Task) {
                    tasks.insert(id);
                    format!(" #{id}")
                } else {
                    String::new()
                };
                let _ = writeln!(out, "{indent}-> {name}{tag}: {title}");
                if let Some(d) = diff {
                    let (a, r) = tools::count_changes(&d);
                    let _ = writeln!(out, "{indent}   (+{a} -{r})");
                }
            }
            Msg::Permission { reply, .. } => {
                let _ = reply.send(if yolo {
                    Decision::Allow
                } else {
                    Decision::Deny(Some(
                        "Not allowed in this mode: only read and search.".into(),
                    ))
                });
            }
            Msg::ToolDone { id, ok, summary } => {
                nl(&mut out, &mut mid_line);
                let _ = writeln!(
                    out,
                    "{indent}<- {}{summary}{}",
                    if ok { "" } else { "FAILED " },
                    if tasks.contains(&id) {
                        format!(" (#{id})")
                    } else {
                        String::new()
                    }
                );
            }
            Msg::Plan { overview, tasks } => {
                nl(&mut out, &mut mid_line);
                let _ = writeln!(out, "== plan\n{overview}\n\n{tasks}\n==");
            }
            Msg::TurnDone { stats, .. } => {
                nl(&mut out, &mut mid_line);
                eprintln!("{indent}[{stats}]");
            }
            Msg::Error(e) => {
                nl(&mut out, &mut mid_line);
                eprintln!("{indent}error: {e}");
                status = 1;
            }
            Msg::Fatal(e) => {
                eprintln!("error: {e}");
                return 1;
            }
            _ => {}
        }
        let _ = out.flush();
    }
    let _ = writeln!(out);
    status
}

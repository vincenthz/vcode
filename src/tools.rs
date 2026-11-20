//! The tools the model can call: their definitions, as the chat template
//! expects them, and how they run.
//!
//! A call first becomes a [`Plan`] (arguments checked, the change of an edit
//! computed), which the user may have to approve, then runs.

use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime};

use biv::chat::ToolCall;

use crate::procs::Procs;
use crate::roles::Role;
use crate::sandbox::{Mark, Sandbox};
use similar::{ChangeTag, TextDiff};

/// The functions the model can call
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Tool {
    ReadFile,
    EditFile,
    WriteFile,
    Bash,
    Grep,
    Glob,
    ListDir,
    /// a subagent: run by the agent, not by [`Env`]
    Task,
    /// the pilot's view of the project and its tasks: kept by the agent
    UpdatePlan,
}

impl Tool {
    pub const ALL: [Tool; 9] = [
        Tool::ReadFile,
        Tool::EditFile,
        Tool::WriteFile,
        Tool::Bash,
        Tool::Grep,
        Tool::Glob,
        Tool::ListDir,
        Tool::Task,
        Tool::UpdatePlan,
    ];

    /// The name the model calls it by
    pub fn name(self) -> &'static str {
        match self {
            Tool::ReadFile => "read_file",
            Tool::EditFile => "edit_file",
            Tool::WriteFile => "write_file",
            Tool::Bash => "bash",
            Tool::Grep => "grep",
            Tool::Glob => "glob",
            Tool::ListDir => "list_dir",
            Tool::Task => "task",
            Tool::UpdatePlan => "update_plan",
        }
    }

    pub fn from_name(name: &str) -> Option<Tool> {
        Tool::ALL.into_iter().find(|t| t.name() == name.trim())
    }

    /// What approving a call of it allows
    pub fn kind(self) -> Kind {
        match self {
            Tool::ReadFile | Tool::Grep | Tool::Glob | Tool::ListDir => Kind::Read,
            Tool::EditFile | Tool::WriteFile => Kind::Edit,
            Tool::Bash => Kind::Exec,
            Tool::Task => Kind::Delegate,
            Tool::UpdatePlan => Kind::Read,
        }
    }

    /// Its function definition, written as the chat template's `tojson` writes it
    pub fn definition(self) -> &'static str {
        match self {
            Tool::ReadFile => {
                r#"{"type": "function", "function": {"name": "read_file", "description": "Read a text file. Returns its lines prefixed with their line numbers (the prefix is not part of the file). Reads up to 400 lines from offset; read further with a larger offset.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path, absolute or relative to the working directory"}, "offset": {"type": "integer", "description": "First line to read, from 1"}, "limit": {"type": "integer", "description": "Number of lines to read"}}, "required": ["path"]}}}"#
            }
            Tool::EditFile => {
                r#"{"type": "function", "function": {"name": "edit_file", "description": "Replace exact text in a file. old_string must appear in the file exactly as written (whitespace and indentation included, line number prefixes excluded) and only once, unless replace_all is true. Include enough surrounding lines to make it unique. Read the file before editing it.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}, "old_string": {"type": "string", "description": "The exact text to replace"}, "new_string": {"type": "string", "description": "The text to put instead"}, "replace_all": {"type": "boolean", "description": "Replace every occurrence"}}, "required": ["path", "old_string", "new_string"]}}}"#
            }
            Tool::WriteFile => {
                r#"{"type": "function", "function": {"name": "write_file", "description": "Create a file, or replace all of its content. Parent directories are created. Prefer edit_file to change part of an existing file.", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "File path"}, "content": {"type": "string", "description": "The whole content of the file"}}, "required": ["path", "content"]}}}"#
            }
            Tool::Bash => {
                r#"{"type": "function", "function": {"name": "bash", "description": "Run a shell command with bash in the working directory and return its output (stdout and stderr) and exit code. Each call starts a new shell in the working directory. Use it to build, run tests, use git, and so on. Don't run interactive programs.", "parameters": {"type": "object", "properties": {"command": {"type": "string", "description": "The command"}, "timeout": {"type": "integer", "description": "Seconds before the command is killed, 120 by default, 600 at most"}}, "required": ["command"]}}}"#
            }
            Tool::Grep => {
                r#"{"type": "function", "function": {"name": "grep", "description": "Search file contents with a regular expression (Rust regex syntax), skipping files ignored by git. Returns path:line: text for each matching line.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The regular expression"}, "path": {"type": "string", "description": "File or directory to search, the working directory by default"}, "include": {"type": "string", "description": "Only files whose path matches this glob, e.g. *.rs or src/**/*.ts"}, "ignore_case": {"type": "boolean"}}, "required": ["pattern"]}}}"#
            }
            Tool::Glob => {
                r#"{"type": "function", "function": {"name": "glob", "description": "Find files whose path matches a glob pattern, e.g. **/*.rs or src/*.toml, skipping files ignored by git. Most recently modified first.", "parameters": {"type": "object", "properties": {"pattern": {"type": "string", "description": "The glob, relative to path"}, "path": {"type": "string", "description": "Directory to search, the working directory by default"}}, "required": ["pattern"]}}}"#
            }
            Tool::ListDir => {
                r#"{"type": "function", "function": {"name": "list_dir", "description": "List the entries of a directory; directories end with /", "parameters": {"type": "object", "properties": {"path": {"type": "string", "description": "Directory, the working directory by default"}}, "required": []}}}"#
            }
            Tool::Task => {
                r#"{"type": "function", "function": {"name": "task", "description": "Give a task to a subagent, which starts knowing nothing of this conversation and works in a context of its own; its final report is the result. Kinds of subagents: explore investigates the code and answers questions, changing nothing; implement makes one change, then builds and tests it; verify checks a change or a claim independently (builds, tests, runs) and reports PASS or FAIL with evidence; review reviews changes for bugs and differences from what was asked; formalize makes requirements precise (invariants, edge cases, examples) and encodes them as tests; document writes or updates documentation; general does anything else. The prompt must be complete: the goal, what is already known (files, facts, decisions), constraints, and what to report.", "parameters": {"type": "object", "properties": {"agent": {"type": "string", "enum": ["explore", "implement", "verify", "review", "formalize", "document", "general"], "description": "The kind of subagent, general by default"}, "description": {"type": "string", "description": "What the subagent does, in 3 to 6 words, shown to the user"}, "prompt": {"type": "string", "description": "The task, complete and self-contained"}}, "required": ["agent", "description", "prompt"]}}}"#
            }
            Tool::UpdatePlan => {
                r#"{"type": "function", "function": {"name": "update_plan", "description": "Record your view of the project and the plan of the work, replacing the previous ones. Call it once you know enough to plan, and again whenever a task starts, ends, fails or changes. The user sees it, and it is kept when the conversation is compacted.", "parameters": {"type": "object", "properties": {"overview": {"type": "string", "description": "What matters about the project for this work, in a few lines: what it is, its structure and key files, conventions, decisions made"}, "tasks": {"type": "string", "description": "The tasks, one per line, each starting with [ ] to do, [~] in progress, [x] done, or [!] blocked or failed"}}, "required": ["overview", "tasks"]}}}"#
            }
        }
    }
}

/// Characters of a tool result sent to the model at most
const MAX_OUTPUT: usize = 24_000;
/// Lines of a file read by default
const READ_LINES: usize = 400;
/// Characters of a line shown to the model at most
const MAX_LINE: usize = 500;
/// Matches of grep, files of glob, entries of list_dir
const MAX_RESULTS: usize = 200;
/// How long the output of a command whose shell exited may still come,
/// before what holds it open is taken for processes left in the background
const BACKGROUND_AFTER: Duration = Duration::from_millis(300);
/// Files larger than this aren't searched in a sandbox (they would be copied)
const MAX_GREP_FILE: u64 = 4 << 20;
/// Files a glob finds in a sandbox past which they aren't sorted by time
const MAX_SORTED: usize = 5000;

/// What approving a call allows
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Kind {
    /// reads, searches, notes: never asked
    Read,
    /// changes files
    Edit,
    /// runs a command
    Exec,
    /// starts a subagent, whose own calls are approved one by one
    Delegate,
}

/// A checked call, ready to run
pub struct Plan {
    pub tool: Tool,
    /// what it does in a few words, e.g. `Edit src/main.rs`
    pub title: String,
    /// the change of an edit, shown before it is approved
    pub diff: Option<Vec<DiffLine>>,
    op: Op,
}

enum Op {
    Read {
        path: PathBuf,
        offset: usize,
        limit: usize,
    },
    Write {
        path: PathBuf,
        content: String,
        created: bool,
    },
    Bash {
        command: String,
        timeout: Duration,
    },
    Grep {
        re: regex::Regex,
        root: PathBuf,
        include: Option<globset::GlobMatcher>,
    },
    Glob {
        glob: globset::GlobMatcher,
        root: PathBuf,
    },
    List {
        path: PathBuf,
    },
}

/// A line of a diff
#[derive(Debug, Clone)]
pub enum DiffLine {
    /// line numbers in the old and the new file
    Same(usize, String),
    Removed(usize, String),
    Added(usize, String),
    /// lines left out between hunks
    Gap,
}

/// The result of a call
pub struct Output {
    /// for the model
    pub text: String,
    /// for the user: a short summary of the result, e.g. `42 lines`
    pub summary: String,
    pub ok: bool,
}

/// Where tools run
pub struct Env {
    pub cwd: PathBuf,
    /// set to interrupt a running command
    pub stop: Arc<AtomicBool>,
    /// where its commands are listed while they run
    pub procs: Procs,
    /// who runs them, as listed: `agent`, `explore #12`, `you`
    pub owner: String,
    /// where the tools run when not here: `cwd` is then its work directory
    pub sandbox: Option<Arc<Sandbox>>,
}

fn arg<'a>(call: &'a ToolCall, name: &str) -> Option<&'a str> {
    call.arguments
        .iter()
        .find(|(k, _)| k == name)
        .map(|(_, v)| v.as_str())
}

fn required<'a>(call: &'a ToolCall, name: &str) -> Result<&'a str, String> {
    arg(call, name).ok_or_else(|| format!("missing parameter {name}"))
}

fn number(call: &ToolCall, name: &str) -> Result<Option<usize>, String> {
    match arg(call, name).map(str::trim) {
        None | Some("") | Some("null") => Ok(None),
        Some(v) => v
            .parse::<f64>()
            .map(|n| Some(n.max(0.0) as usize))
            .map_err(|_| format!("{name}: not a number: {v}")),
    }
}

fn boolean(call: &ToolCall, name: &str) -> bool {
    matches!(arg(call, name).map(str::trim), Some("true" | "True" | "1"))
}

impl Env {
    /// A path the model gave: `~` expanded, relative to the working directory
    fn resolve(&self, path: &str) -> PathBuf {
        let path = path.trim();
        let path = path.strip_prefix("file://").unwrap_or(path);
        if path.is_empty() || path == "." {
            return self.cwd.clone();
        }
        if let Some(rest) = path.strip_prefix("~/") {
            if let Some(s) = &self.sandbox {
                return s.home.join(rest);
            }
            if let Some(home) = std::env::var_os("HOME") {
                return PathBuf::from(home).join(rest);
            }
        }
        self.cwd.join(path)
    }

    /// The content of a file, here or in the sandbox
    fn load(&self, path: &Path) -> std::io::Result<Vec<u8>> {
        match &self.sandbox {
            Some(s) => s.read(path),
            None => std::fs::read(path),
        }
    }

    fn load_text(&self, path: &Path) -> std::io::Result<String> {
        String::from_utf8(self.load(path)?).map_err(|_| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "stream did not contain valid UTF-8",
            )
        })
    }

    /// A path as shown: relative to the working directory when inside it
    pub fn show(&self, path: &Path) -> String {
        match path.strip_prefix(&self.cwd) {
            Ok(p) if p.as_os_str().is_empty() => ".".into(),
            Ok(p) => p.display().to_string(),
            Err(_) => path.display().to_string(),
        }
    }

    /// Check a call and prepare it
    pub fn plan(&self, call: &ToolCall) -> Result<Plan, String> {
        let tool =
            Tool::from_name(&call.name).ok_or_else(|| format!("unknown function {}", call.name))?;
        let plan = |title: String, op| Plan {
            tool,
            title,
            diff: None,
            op,
        };
        Ok(match tool {
            Tool::ReadFile => {
                let path = self.resolve(required(call, "path")?);
                let offset = number(call, "offset")?.unwrap_or(1).max(1);
                let limit = number(call, "limit")?.unwrap_or(READ_LINES).max(1);
                let title = format!("Read {}", self.show(&path));
                plan(
                    title,
                    Op::Read {
                        path,
                        offset,
                        limit,
                    },
                )
            }
            Tool::WriteFile => {
                let path = self.resolve(required(call, "path")?);
                let content = required(call, "content")?.to_string();
                let old = self.load_text(&path).ok();
                let created = old.is_none();
                let diff = diff(old.as_deref().unwrap_or(""), &content);
                Plan {
                    tool,
                    title: format!(
                        "{} {}",
                        if created { "Create" } else { "Write" },
                        self.show(&path)
                    ),
                    diff: Some(diff),
                    op: Op::Write {
                        path,
                        content,
                        created,
                    },
                }
            }
            Tool::EditFile => {
                let path = self.resolve(required(call, "path")?);
                let old_string = required(call, "old_string")?;
                let new_string = required(call, "new_string")?;
                let all = boolean(call, "replace_all");
                let old = self
                    .load_text(&path)
                    .map_err(|e| format!("{}: {e}", self.show(&path)))?;
                if old_string.is_empty() {
                    return Err("old_string is empty: use write_file to create a file".into());
                }
                if old_string == new_string {
                    return Err("old_string and new_string are the same".into());
                }
                let content = replace(&old, old_string, new_string, all)
                    .map_err(|e| format!("{}: {e}", self.show(&path)))?;
                Plan {
                    tool,
                    title: format!("Edit {}", self.show(&path)),
                    diff: Some(diff(&old, &content)),
                    op: Op::Write {
                        path,
                        content,
                        created: false,
                    },
                }
            }
            Tool::Bash => {
                let command = required(call, "command")?.trim().to_string();
                if command.is_empty() {
                    return Err("empty command".into());
                }
                let secs = number(call, "timeout")?.unwrap_or(120).clamp(1, 600);
                plan(
                    command.clone(),
                    Op::Bash {
                        command,
                        timeout: Duration::from_secs(secs as u64),
                    },
                )
            }
            Tool::Grep => {
                let pattern = required(call, "pattern")?;
                let re = regex::RegexBuilder::new(pattern)
                    .case_insensitive(boolean(call, "ignore_case"))
                    .build()
                    .map_err(|e| format!("pattern: {e}"))?;
                let root = self.resolve(arg(call, "path").unwrap_or("."));
                let include = match arg(call, "include").map(str::trim) {
                    Some(g) if !g.is_empty() => Some(glob_matcher(g)?),
                    _ => None,
                };
                let title = format!("Grep {pattern:?} in {}", self.show(&root));
                plan(title, Op::Grep { re, root, include })
            }
            Tool::Glob => {
                let pattern = required(call, "pattern")?;
                let root = self.resolve(arg(call, "path").unwrap_or("."));
                let title = format!("Glob {pattern} in {}", self.show(&root));
                let glob = glob_matcher(pattern)?;
                plan(title, Op::Glob { glob, root })
            }
            Tool::Task | Tool::UpdatePlan => {
                return Err(format!("{} is not available here", tool.name()));
            }
            Tool::ListDir => {
                let path = self.resolve(arg(call, "path").unwrap_or("."));
                let title = format!("List {}", self.show(&path));
                plan(title, Op::List { path })
            }
        })
    }

    /// Run a plan. `progress` gets the output of a command as it comes.
    pub fn run(&self, plan: Plan, progress: &mut dyn FnMut(&str)) -> Output {
        let result = match plan.op {
            Op::Read {
                path,
                offset,
                limit,
            } => self.read(&path, offset, limit),
            Op::Write {
                path,
                content,
                created,
            } => (|| {
                match &self.sandbox {
                    Some(s) => s.write(&path, content.as_bytes())?,
                    None => {
                        if let Some(dir) = path.parent() {
                            std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                        }
                        std::fs::write(&path, &content).map_err(|e| e.to_string())?;
                    }
                }
                let (added, removed) = plan.diff.as_deref().map(count_changes).unwrap_or((0, 0));
                let summary = if created {
                    format!("{} lines", content.lines().count())
                } else {
                    format!("+{added} -{removed}")
                };
                Ok(Output {
                    text: format!(
                        "{} {}",
                        if created { "Created" } else { "Updated" },
                        self.show(&path)
                    ),
                    summary,
                    ok: true,
                })
            })(),
            Op::Bash { command, timeout } => Ok(self.bash(&command, timeout, progress)),
            Op::Grep { re, root, include } => Ok(self.grep(&re, &root, include.as_ref())),
            Op::Glob { glob, root } => Ok(self.glob(&glob, &root)),
            Op::List { path } => self.list(&path),
        };
        result.unwrap_or_else(failure)
    }

    fn read(&self, path: &Path, offset: usize, limit: usize) -> Result<Output, String> {
        let shown = self.show(path);
        let is_dir = || format!("{shown} is a directory, use list_dir");
        if self.sandbox.is_none() && path.is_dir() {
            return Err(is_dir());
        }
        let bytes = match self.load(path) {
            Ok(b) => b,
            Err(e) if e.kind() == std::io::ErrorKind::IsADirectory => return Err(is_dir()),
            Err(e) => return Err(format!("{shown}: {e}")),
        };
        if bytes.iter().take(8192).any(|&b| b == 0) {
            return Err(format!("{shown}: binary file, {} bytes", bytes.len()));
        }
        let text = String::from_utf8_lossy(&bytes);
        let total = text.lines().count();
        if total == 0 {
            return Ok(Output {
                text: format!("{shown} is empty"),
                summary: "empty".into(),
                ok: true,
            });
        }
        if offset > total {
            return Err(format!(
                "{shown} has {total} lines, offset {offset} is past the end"
            ));
        }
        let mut out = String::new();
        let mut last = offset - 1;
        for (i, line) in text.lines().enumerate().skip(offset - 1).take(limit) {
            if out.len() > MAX_OUTPUT {
                break;
            }
            let _ = writeln!(out, "{:>6}\t{}", i + 1, clip(line, MAX_LINE));
            last = i + 1;
        }
        if last < total {
            let _ = write!(
                out,
                "\n({shown} has {total} lines; read more with offset={})",
                last + 1
            );
        }
        Ok(Output {
            text: out,
            summary: if offset == 1 && last == total {
                format!("{total} lines")
            } else {
                format!("lines {offset}-{last} of {total}")
            },
            ok: true,
        })
    }

    fn bash(&self, command: &str, timeout: Duration, progress: &mut dyn FnMut(&str)) -> Output {
        let (mut cmd, what) = match &self.sandbox {
            Some(s) => (s.shell(&self.cwd, command), "ssh"),
            None => {
                let shell = if Path::new("/bin/bash").exists() {
                    "/bin/bash"
                } else {
                    "/bin/sh"
                };
                let mut cmd = Command::new(shell);
                cmd.arg("-c")
                    .arg(command)
                    .current_dir(&self.cwd)
                    .env("PAGER", "cat")
                    .env("GIT_PAGER", "cat")
                    .env("NO_COLOR", "1")
                    .env("TERM", "dumb");
                (cmd, shell)
            }
        };
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // its own process group, to kill what it starts too
            cmd.process_group(0);
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return failure(format!("{what}: {e}")),
        };
        let job = self.procs.add(command, &self.owner, child.id());
        // in a sandbox, the shell's exit status there, from its mark: ssh
        // lasts while what it left in the background holds the output
        let exit: Arc<std::sync::Mutex<Option<i32>>> = Default::default();
        let mut readers = Vec::new();
        if let Some(mut pipe) = child.stdout.take() {
            let job = job.clone();
            readers.push(std::thread::spawn(move || {
                let mut buf = [0u8; 8192];
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    job.out.lock().unwrap().push(&buf[..n]);
                }
            }));
        }
        if let Some(mut pipe) = child.stderr.take() {
            let job = job.clone();
            let sandbox = self.sandbox.clone();
            let exit = exit.clone();
            readers.push(std::thread::spawn(move || {
                let mut demux = sandbox.as_ref().map(|s| s.demux());
                let mut buf = [0u8; 8192];
                let mut marks = Vec::new();
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    let Some(d) = demux.as_mut() else {
                        job.out.lock().unwrap().push(&buf[..n]);
                        continue;
                    };
                    let data = d.feed(&buf[..n], &mut marks);
                    job.out.lock().unwrap().push(&data);
                    for m in marks.drain(..) {
                        match (m, &sandbox) {
                            (Mark::Pid(pgid), Some(s)) => {
                                let s = s.clone();
                                job.on_kill(Box::new(move || s.kill_group(pgid)));
                            }
                            (Mark::Exit(code), _) => *exit.lock().unwrap() = Some(code),
                            _ => {}
                        }
                    }
                }
                if let Some(d) = demux.as_mut() {
                    job.out.lock().unwrap().push(&d.finish());
                }
            }));
        }

        let start = job.start;
        let mut sent = 0;
        let mut stream = |progress: &mut dyn FnMut(&str)| {
            let out = job.out.lock().unwrap();
            let head = out.head();
            if head.len() > sent {
                progress(&String::from_utf8_lossy(&head[sent..]));
                sent = head.len();
            }
        };
        let mut killed = None;
        // the shell's exit code, None if killed by a signal; whether its
        // connection was lost, in a sandbox
        let (code, lost) = loop {
            match child.try_wait() {
                Ok(Some(status)) => {
                    // its output may still come with its mark
                    let mut marked = *exit.lock().unwrap();
                    if self.sandbox.is_some() && marked.is_none() {
                        let t = Instant::now();
                        while !readers.iter().all(|r| r.is_finished())
                            && t.elapsed() < BACKGROUND_AFTER
                        {
                            std::thread::sleep(Duration::from_millis(10));
                        }
                        marked = *exit.lock().unwrap();
                    }
                    break match (&self.sandbox, marked) {
                        (Some(_), Some(c)) => (Some(c), false),
                        (Some(_), None) => (status.code(), killed.is_none()),
                        (None, _) => (status.code(), false),
                    };
                }
                Ok(None) => {
                    if let Some(c) = *exit.lock().unwrap() {
                        break (Some(c), false);
                    }
                }
                Err(_) => break (None, false),
            }
            if killed.is_none() {
                if self.stop.load(Ordering::SeqCst) {
                    killed = Some("interrupted by the user");
                } else if job.killed() {
                    killed = Some("killed by the user");
                } else if start.elapsed() > timeout {
                    killed = Some("timed out");
                }
                if killed.is_some() {
                    job.kill();
                    // the shell itself, in case it left its group
                    let _ = child.kill();
                }
            }
            stream(progress);
            std::thread::sleep(Duration::from_millis(30));
        };
        // killed from the list as it exited
        if killed.is_none() && job.killed() {
            killed = Some("killed by the user");
        }
        // the shell is gone, but processes it started may hold its output
        // open (`server &`): they go on in the background, still listed
        let grace = Instant::now();
        while !readers.iter().all(|r| r.is_finished()) && grace.elapsed() < BACKGROUND_AFTER {
            std::thread::sleep(Duration::from_millis(10));
        }
        let background = !readers.iter().all(|r| r.is_finished());
        if background {
            job.set_background();
            let procs = self.procs.clone();
            let job = job.clone();
            std::thread::spawn(move || {
                let mut since_kill = None;
                while !readers.iter().all(|r| r.is_finished()) {
                    // killed, yet the output still held: by a process that
                    // left the group; it isn't listed any longer
                    if job.killed()
                        && since_kill.get_or_insert_with(Instant::now).elapsed()
                            > Duration::from_secs(2)
                    {
                        let _ = child.kill();
                        break;
                    }
                    std::thread::sleep(Duration::from_millis(100));
                }
                // ssh, in a sandbox, ends with the output
                let _ = child.wait();
                procs.remove(job.id);
            });
        } else {
            for r in readers {
                let _ = r.join();
            }
            let _ = child.wait();
            self.procs.remove(job.id);
        }
        stream(progress);
        let text = strip_ansi(&job.out.lock().unwrap().text());
        let text = text.trim_end();
        let mut result = middle_out(text, MAX_OUTPUT);
        let ok = killed.is_none() && code == Some(0);
        let mut status = match (killed, code) {
            (Some(why), _) => format!("[{why} after {:.0}s]", start.elapsed().as_secs_f64()),
            (None, _) if lost => "[the connection to the sandbox was lost]".into(),
            (None, Some(0)) => String::new(),
            (None, Some(c)) => format!("[exit code {c}]"),
            (None, None) => "[killed by a signal]".into(),
        };
        let lines = text.lines().count();
        let mut summary = match (status.is_empty(), lines) {
            (true, 0) => "no output".into(),
            (true, n) => format!("{n} lines"),
            (false, _) => status.trim_matches(['[', ']']).to_string(),
        };
        if background && killed.is_none() {
            if !status.is_empty() {
                status.push('\n');
            }
            status.push_str(
                "[processes it started are still running in the background; their later output \
                 is not shown; the user can list and stop them]",
            );
            summary.push_str(" · still running in the background (ctrl+p)");
        }
        if !status.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&status);
        }
        if result.is_empty() {
            result = "(no output)".into();
        }
        Output {
            summary,
            text: result,
            ok,
        }
    }

    fn grep(
        &self,
        re: &regex::Regex,
        root: &Path,
        include: Option<&globset::GlobMatcher>,
    ) -> Output {
        let included = |path: &Path| match include {
            None => true,
            Some(g) => {
                let rel = path.strip_prefix(root).unwrap_or(path);
                g.is_match(rel) || path.file_name().is_some_and(|n| g.is_match(n))
            }
        };
        let mut out = String::new();
        let mut matches = 0;
        let mut files = 0;
        // false once there are enough matches
        let mut search = |path: &Path, bytes: &[u8]| {
            if bytes.iter().take(8192).any(|&b| b == 0) {
                return true;
            }
            let text = String::from_utf8_lossy(bytes);
            let mut found = false;
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    if !found {
                        found = true;
                        files += 1;
                    }
                    matches += 1;
                    if matches > MAX_RESULTS {
                        return false;
                    }
                    let _ = writeln!(
                        out,
                        "{}:{}: {}",
                        self.show(path),
                        i + 1,
                        clip(line.trim_end(), 300)
                    );
                }
            }
            true
        };
        match &self.sandbox {
            None => {
                for entry in walk(root) {
                    let path = entry.path();
                    if !included(path) {
                        continue;
                    }
                    let Ok(bytes) = std::fs::read(path) else {
                        continue;
                    };
                    if !search(path, &bytes) {
                        break;
                    }
                }
            }
            // the files there copied here, to search them the same way
            Some(s) => {
                let paths: Vec<PathBuf> = match s.files(root) {
                    Ok(p) => p.into_iter().filter(|p| included(p)).collect(),
                    Err(e) => return failure(format!("{}: {e}", self.show(root))),
                };
                if let Err(e) = s.fetch(&paths, MAX_GREP_FILE, &mut |p, e| search(&p, &e.data)) {
                    return failure(e);
                }
            }
        }
        if matches == 0 {
            return Output {
                text: "No matches".into(),
                summary: "no matches".into(),
                ok: true,
            };
        }
        if matches > MAX_RESULTS {
            let _ = write!(
                out,
                "(more than {MAX_RESULTS} matches, only the first shown; narrow the search)"
            );
        }
        Output {
            text: middle_out(&out, MAX_OUTPUT),
            summary: if matches > MAX_RESULTS {
                format!("{MAX_RESULTS}+ matches")
            } else {
                format!("{matches} matches in {files} files")
            },
            ok: true,
        }
    }

    fn glob(&self, glob: &globset::GlobMatcher, root: &Path) -> Output {
        if let Some(s) = &self.sandbox {
            let found: Vec<PathBuf> = match s.files(root) {
                Ok(p) => p
                    .into_iter()
                    .filter(|p| glob.is_match(p.strip_prefix(root).unwrap_or(p)))
                    .collect(),
                Err(e) => return failure(format!("{}: {e}", self.show(root))),
            };
            let found = if found.len() <= MAX_SORTED {
                s.newest_first(found)
            } else {
                found
            };
            return self.found(found.into_iter());
        }
        let mut found: Vec<(SystemTime, PathBuf)> = walk(root)
            .filter(|e| {
                let rel = e.path().strip_prefix(root).unwrap_or(e.path());
                glob.is_match(rel)
            })
            .map(|e| {
                let t = e
                    .metadata()
                    .ok()
                    .and_then(|m| m.modified().ok())
                    .unwrap_or(SystemTime::UNIX_EPOCH);
                (t, e.into_path())
            })
            .collect();
        found.sort_by_key(|f| std::cmp::Reverse(f.0));
        self.found(found.into_iter().map(|(_, p)| p))
    }

    /// The files a glob found, most recent first
    fn found(&self, found: impl ExactSizeIterator<Item = PathBuf>) -> Output {
        let n = found.len();
        let mut out: String = found
            .take(MAX_RESULTS)
            .map(|p| self.show(&p) + "\n")
            .collect();
        if n > MAX_RESULTS {
            let _ = write!(out, "({n} files, only the {MAX_RESULTS} most recent shown)");
        }
        if n == 0 {
            out = "No files found".into();
        }
        Output {
            text: out,
            summary: format!("{n} files"),
            ok: true,
        }
    }

    fn list(&self, path: &Path) -> Result<Output, String> {
        let shown = self.show(path);
        let mut entries: Vec<String> = match &self.sandbox {
            Some(s) => s.list(path).map_err(|e| format!("{shown}: {e}"))?,
            None => std::fs::read_dir(path)
                .map_err(|e| format!("{shown}: {e}"))?
                .filter_map(Result::ok)
                .map(|e| {
                    let name = e.file_name().to_string_lossy().into_owned();
                    if e.path().is_dir() { name + "/" } else { name }
                })
                .collect(),
        };
        entries.sort();
        let n = entries.len();
        let mut out = entries
            .iter()
            .take(MAX_RESULTS * 2)
            .map(|e| e.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        if n > MAX_RESULTS * 2 {
            let _ = write!(
                out,
                "\n({n} entries, only the first {} shown)",
                MAX_RESULTS * 2
            );
        }
        if n == 0 {
            out = format!("{shown} is empty");
        }
        Ok(Output {
            text: out,
            summary: format!("{n} entries"),
            ok: true,
        })
    }
}

/// The kind of subagent, the description and the prompt of a call of [`Tool::Task`]
pub fn task_request(call: &ToolCall) -> Result<(Role, String, String), String> {
    let prompt = required(call, "prompt")?.trim();
    if prompt.is_empty() {
        return Err("empty prompt".into());
    }
    let role = match arg(call, "agent").map(str::trim) {
        None | Some("") => Role::General,
        Some(name) => Role::from_name(name).ok_or_else(|| {
            let names: Vec<_> = Role::ALL.iter().map(|r| r.name()).collect();
            format!("unknown agent {name}, one of {}", names.join(", "))
        })?,
    };
    let description = match arg(call, "description").map(str::trim) {
        Some(d) if !d.is_empty() => d.to_string(),
        _ => clip(prompt.lines().next().unwrap_or_default(), 40),
    };
    Ok((role, description, prompt.to_string()))
}

/// The overview and the tasks of a call of [`Tool::UpdatePlan`]
pub fn plan_request(call: &ToolCall) -> Result<(String, String), String> {
    let tasks = required(call, "tasks")?.trim().to_string();
    let overview = arg(call, "overview").unwrap_or_default().trim().to_string();
    Ok((overview, tasks))
}

/// A command the user runs (`!`), in `cwd`, or in the sandbox's work
/// directory: listed in `procs`, where it can be killed
pub fn shell(cwd: &Path, command: &str, procs: &Procs, sandbox: Option<Arc<Sandbox>>) -> Output {
    let env = Env {
        cwd: sandbox.as_ref().map_or_else(|| cwd.to_path_buf(), |s| s.root.clone()),
        stop: Arc::new(AtomicBool::new(false)),
        procs: procs.clone(),
        owner: "you".into(),
        sandbox,
    };
    env.bash(command, Duration::from_secs(600), &mut |_| {})
}

fn failure(e: String) -> Output {
    Output {
        summary: clip(e.lines().next().unwrap_or_default(), 120),
        text: format!("Error: {e}"),
        ok: false,
    }
}

fn glob_matcher(pattern: &str) -> Result<globset::GlobMatcher, String> {
    let pattern = pattern.trim().trim_start_matches("./");
    globset::GlobBuilder::new(pattern)
        .literal_separator(true)
        .build()
        .map(|g| g.compile_matcher())
        .map_err(|e| format!("glob: {e}"))
}

/// The files under `root`, without those git ignores and the `.git` directory
fn walk(root: &Path) -> impl Iterator<Item = ignore::DirEntry> {
    ignore::WalkBuilder::new(root)
        .hidden(false)
        .filter_entry(|e| e.file_name() != ".git")
        .build()
        .filter_map(Result::ok)
        .filter(|e| e.file_type().is_some_and(|t| t.is_file()))
}

/// `s` replaced by `with` in `text`: once, unless `all`
fn replace(text: &str, s: &str, with: &str, all: bool) -> Result<String, String> {
    let n = text.matches(s).count();
    if n == 1 || (n > 1 && all) {
        return Ok(text.replace(s, with));
    }
    if n > 1 {
        return Err(format!(
            "old_string appears {n} times; include more surrounding lines to make it unique, or set replace_all"
        ));
    }
    // the model often gets trailing whitespace or the file's line endings wrong
    if text.contains("\r\n") && !s.contains("\r\n") {
        let crlf = |t: &str| t.replace('\n', "\r\n");
        if let Ok(r) = replace(text, &crlf(s), &crlf(with), all) {
            return Ok(r);
        }
    }
    let trimmed: String = s.lines().map(str::trim_end).collect::<Vec<_>>().join("\n");
    if trimmed != s && text.matches(&trimmed).count() == 1 {
        return Ok(text.replacen(&trimmed, with.trim_end_matches('\n'), 1));
    }
    Err("old_string not found; read the file again and copy the text exactly, without the line number prefixes".into())
}

/// The changed lines between `old` and `new`, with 3 lines of context
pub fn diff(old: &str, new: &str) -> Vec<DiffLine> {
    let d = TextDiff::from_lines(old, new);
    let mut out = Vec::new();
    for (i, group) in d.grouped_ops(3).iter().enumerate() {
        if i > 0 {
            out.push(DiffLine::Gap);
        }
        for op in group {
            for c in d.iter_changes(op) {
                let line = c.value().trim_end_matches(['\n', '\r']).to_string();
                out.push(match c.tag() {
                    ChangeTag::Equal => DiffLine::Same(c.new_index().unwrap_or(0) + 1, line),
                    ChangeTag::Delete => DiffLine::Removed(c.old_index().unwrap_or(0) + 1, line),
                    ChangeTag::Insert => DiffLine::Added(c.new_index().unwrap_or(0) + 1, line),
                });
            }
        }
    }
    out
}

/// Lines added and removed
pub fn count_changes(diff: &[DiffLine]) -> (usize, usize) {
    diff.iter().fold((0, 0), |(a, r), l| match l {
        DiffLine::Added(..) => (a + 1, r),
        DiffLine::Removed(..) => (a, r + 1),
        _ => (a, r),
    })
}

/// At most `n` characters of `s`
pub fn clip(s: &str, n: usize) -> String {
    match s.char_indices().nth(n) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

/// `s` cut to about `max` bytes by leaving out its middle
pub fn middle_out(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let floor = |mut i: usize| {
        while !s.is_char_boundary(i) {
            i -= 1;
        }
        i
    };
    let head = floor(max / 3);
    let tail = floor(s.len() - max * 2 / 3);
    let omitted = s[head..tail].lines().count();
    format!(
        "{}\n[... {omitted} lines omitted ...]\n{}",
        &s[..head],
        &s[tail..]
    )
}

/// Text without terminal escape sequences, carriage returns resolved
pub fn strip_ansi(s: &str) -> String {
    static RE: std::sync::OnceLock<regex::Regex> = std::sync::OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(
            r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(\x07|\x1b\\)|\x1b[@-Z\\-_]",
        )
        .unwrap()
    });
    let s = re.replace_all(s, "");
    // progress bars: keep what follows the last carriage return of a line
    s.split('\n')
        .map(|l| {
            let l = l.strip_suffix('\r').unwrap_or(l);
            l.rsplit('\r').next().unwrap_or(l)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(name: &str, args: &[(&str, &str)]) -> ToolCall {
        ToolCall {
            name: name.into(),
            arguments: args
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }

    fn env() -> (Env, PathBuf) {
        let dir = std::env::temp_dir().join(format!(
            "vcode-test-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("src")).unwrap();
        std::fs::write(
            dir.join("src/a.rs"),
            "fn main() {\n    println!(\"hi\");\n}\n",
        )
        .unwrap();
        std::fs::write(dir.join("notes.txt"), "one\ntwo\nthree\n").unwrap();
        (
            Env {
                cwd: dir.clone(),
                stop: Arc::new(AtomicBool::new(false)),
                procs: Procs::default(),
                owner: "agent".into(),
                sandbox: None,
            },
            dir,
        )
    }

    fn run(env: &Env, c: ToolCall) -> Output {
        let plan = env.plan(&c).unwrap_or_else(|e| panic!("{e}"));
        env.run(plan, &mut |_| {})
    }

    #[test]
    fn definitions_are_json() {
        for t in Tool::ALL {
            assert_eq!(Tool::from_name(t.name()), Some(t));
            let d = t.definition();
            assert!(d.contains(&format!(r#""name": "{}""#, t.name())), "{d}");
            // balanced braces and quotes, the cheap check without a JSON parser
            assert_eq!(d.matches('{').count(), d.matches('}').count(), "{d}");
            assert_eq!(d.matches('"').count() % 2, 0, "{d}");
        }
    }

    #[test]
    fn read_edit_write() {
        let (env, dir) = env();
        let out = run(&env, call("read_file", &[("path", "src/a.rs")]));
        assert!(
            out.text.contains("     2\t    println!(\"hi\");"),
            "{}",
            out.text
        );
        assert_eq!(out.summary, "3 lines");

        let out = run(
            &env,
            call(
                "read_file",
                &[("path", "notes.txt"), ("offset", "2"), ("limit", "1")],
            ),
        );
        assert!(out.text.starts_with("     2\ttwo\n"), "{}", out.text);
        assert!(out.text.contains("offset=3"));

        let edit = call(
            "edit_file",
            &[
                ("path", "src/a.rs"),
                ("old_string", "println!(\"hi\");"),
                ("new_string", "println!(\"bye\");"),
            ],
        );
        let plan = env.plan(&edit).unwrap();
        assert_eq!(plan.tool.kind(), Kind::Edit);
        assert_eq!(count_changes(plan.diff.as_ref().unwrap()), (1, 1));
        let out = env.run(plan, &mut |_| {});
        assert!(out.ok);
        assert!(
            std::fs::read_to_string(dir.join("src/a.rs"))
                .unwrap()
                .contains("bye")
        );

        let missing = call(
            "edit_file",
            &[
                ("path", "src/a.rs"),
                ("old_string", "nope"),
                ("new_string", "x"),
            ],
        );
        assert!(env.plan(&missing).is_err());

        let out = run(
            &env,
            call(
                "write_file",
                &[("path", "new/b.txt"), ("content", "x\ny\n")],
            ),
        );
        assert!(out.ok);
        assert_eq!(
            std::fs::read_to_string(dir.join("new/b.txt")).unwrap(),
            "x\ny\n"
        );
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn edit_ambiguous_and_trailing_space() {
        let text = "a = 1\nb = 1\n";
        assert!(replace(text, "= 1", "= 2", false).is_err());
        assert_eq!(replace(text, "= 1", "= 2", true).unwrap(), "a = 2\nb = 2\n");
        assert_eq!(replace("x\ny\n", "x  \ny", "z", false).unwrap(), "z\n");
        assert_eq!(
            replace("a\r\nb\r\n", "a\nb", "c\nd", false).unwrap(),
            "c\r\nd\r\n"
        );
    }

    #[test]
    fn search() {
        let (env, dir) = env();
        let out = run(
            &env,
            call("grep", &[("pattern", "print"), ("include", "*.rs")]),
        );
        assert!(out.text.contains("src/a.rs:2:"), "{}", out.text);
        let out = run(&env, call("glob", &[("pattern", "**/*.rs")]));
        assert_eq!(out.text.trim(), "src/a.rs");
        let out = run(&env, call("list_dir", &[]));
        assert_eq!(out.text, "notes.txt\nsrc/");
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn bash() {
        let (env, dir) = env();
        let out = run(
            &env,
            call("bash", &[("command", "echo hello; echo err >&2; exit 3")]),
        );
        assert!(!out.ok);
        assert!(
            out.text.contains("hello")
                && out.text.contains("err")
                && out.text.contains("[exit code 3]"),
            "{}",
            out.text
        );
        let t = Instant::now();
        let out = run(
            &env,
            call("bash", &[("command", "sleep 30"), ("timeout", "1")]),
        );
        assert!(t.elapsed() < Duration::from_secs(5));
        assert!(out.text.contains("timed out"), "{}", out.text);
        let _ = std::fs::remove_dir_all(dir);
    }

    /// Waits for `cond` 5 s at most
    fn eventually(cond: impl Fn() -> bool) -> bool {
        let t = Instant::now();
        while t.elapsed() < Duration::from_secs(5) {
            if cond() {
                return true;
            }
            std::thread::sleep(Duration::from_millis(20));
        }
        false
    }

    #[test]
    fn background_processes() {
        let (env, dir) = env();
        // the shell exits, its sleep holds the output open
        let t = Instant::now();
        let out = run(
            &env,
            call("bash", &[("command", "sleep 30 & echo started")]),
        );
        assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
        assert!(out.ok, "{}", out.text);
        assert!(out.text.contains("started"), "{}", out.text);
        assert!(out.text.contains("background"), "{}", out.text);
        let list = env.procs.list();
        assert_eq!(list.len(), 1);
        assert!(list[0].background);
        assert_eq!(list[0].owner, "agent");
        assert_eq!(list[0].last, "started");
        // killed from the list, with its sleep
        assert!(env.procs.kill(list[0].id));
        assert!(eventually(|| env.procs.is_empty()));
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn killed_from_the_list() {
        let (env, dir) = env();
        let t = Instant::now();
        let out = std::thread::scope(|s| {
            let running = s.spawn(|| {
                run(
                    &env,
                    call("bash", &[("command", "echo going; sleep 30; echo never")]),
                )
            });
            assert!(eventually(|| env
                .procs
                .list()
                .first()
                .is_some_and(|p| p.last == "going")));
            let id = env.procs.list()[0].id;
            assert!(env.procs.kill(id));
            running.join().unwrap()
        });
        assert!(t.elapsed() < Duration::from_secs(5), "{:?}", t.elapsed());
        assert!(!out.ok);
        assert!(out.text.contains("going"), "{}", out.text);
        assert!(!out.text.contains("never"), "{}", out.text);
        assert!(out.text.contains("killed by the user"), "{}", out.text);
        assert!(env.procs.is_empty());
        let _ = std::fs::remove_dir_all(dir);
    }

    /// In a VM of vemu's, with the tools of an [`Env`]: run with
    /// `VCODE_TEST_IMAGE=golden.raw VCODE_TEST_KERNEL=Image cargo test in_a_vm -- --ignored`
    /// (`VCODE_VEMU` for a vemu binary not on the PATH)
    #[test]
    #[ignore]
    fn in_a_vm() {
        let image = std::env::var("VCODE_TEST_IMAGE").expect("VCODE_TEST_IMAGE");
        let kernel = std::env::var("VCODE_TEST_KERNEL").expect("VCODE_TEST_KERNEL");
        in_a_sandbox(
            &format!("vemu:t,image={image},kernel={kernel},memory=1G,cpus=2"),
            "vemu:t",
        );
    }

    /// On a host over ssh, reached without a password: run with
    /// `VCODE_TEST_HOST=[user@]host[:port] cargo test on_a_host -- --ignored`
    #[test]
    #[ignore]
    fn on_a_host() {
        let host = std::env::var("VCODE_TEST_HOST").expect("VCODE_TEST_HOST");
        let spec = format!("ssh:{host}");
        in_a_sandbox(&spec, &spec);
    }

    /// The tools in the sandbox `first`, then again in `again`: the same one
    fn in_a_sandbox(first: &str, again: &str) {
        use crate::sandbox::{Kind as Change, Sandbox, Spec};
        let base = std::env::temp_dir().join(format!("vcode-sb-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let local = base.join("proj");
        std::fs::create_dir_all(local.join("src")).unwrap();
        std::fs::create_dir_all(local.join("target/debug")).unwrap();
        // SAFETY: this test runs alone (--ignored, by name)
        unsafe { std::env::set_var("XDG_DATA_HOME", base.join("data")) };
        std::fs::write(local.join(".gitignore"), "target/\n").unwrap();
        std::fs::write(local.join("src/a.rs"), "fn main() {\n    println!(\"hi\");\n}\n").unwrap();
        std::fs::write(local.join("notes.txt"), "one\ntwo\n").unwrap();
        std::fs::write(local.join("target/debug/big"), "ignored").unwrap();
        let git = |args: &[&str]| {
            assert!(Command::new("git").args(args).current_dir(&local).output().unwrap().status.success())
        };
        git(&["init", "-q"]);
        git(&["add", "."]);
        git(&["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init"]);

        let spec = Spec::parse(first).unwrap();
        let say = &mut |s: &str| eprintln!("  {s}");
        let sb = Arc::new(Sandbox::start(&spec, &local, say).unwrap());
        let env = Env {
            cwd: sb.root.clone(),
            stop: Arc::new(AtomicBool::new(false)),
            procs: Procs::default(),
            owner: "agent".into(),
            sandbox: Some(sb.clone()),
        };
        let out = run(&env, call("read_file", &[("path", "src/a.rs")]));
        assert!(out.text.contains("println!(\"hi\")"), "{}", out.text);
        assert!(run(&env, call("read_file", &[("path", "src")])).text.contains("is a directory"));
        assert!(!run(&env, call("read_file", &[("path", "nope")])).ok);
        let out = run(&env, call("list_dir", &[]));
        assert_eq!(out.text, ".git/\n.gitignore\nnotes.txt\nsrc/", "{}", out.text);
        let out = run(&env, call("glob", &[("pattern", "**/*.rs")]));
        assert_eq!(out.text.trim(), "src/a.rs");
        let out = run(&env, call("grep", &[("pattern", r"print\w+")]));
        assert!(out.text.contains("src/a.rs:2:"), "{}", out.text);
        // an edit and a new file, there
        let out = run(
            &env,
            call(
                "edit_file",
                &[("path", "src/a.rs"), ("old_string", "\"hi\""), ("new_string", "\"bye\"")],
            ),
        );
        assert!(out.ok, "{}", out.text);
        assert!(run(&env, call("write_file", &[("path", "new/b.txt"), ("content", "b\n")])).ok);
        assert!(std::fs::read_to_string(local.join("src/a.rs")).unwrap().contains("hi"));
        // commands there
        let out = run(&env, call("bash", &[("command", "pwd; uname -s; cat new/b.txt; git status --short; exit 3")]));
        assert!(!out.ok);
        assert!(out.text.contains(&sb.root.display().to_string()), "{}", out.text);
        assert!(out.text.contains("Linux") && out.text.contains("b\n"), "{}", out.text);
        assert!(out.text.contains("M src/a.rs") && out.text.contains("[exit code 3]"), "{}", out.text);
        assert!(!out.text.contains('\u{1e}'), "marks left in: {:?}", out.text);
        let t = Instant::now();
        let out = run(&env, call("bash", &[("command", "sleep 30"), ("timeout", "1")]));
        assert!(t.elapsed() < Duration::from_secs(5) && out.text.contains("timed out"), "{}", out.text);
        // left in the background, then killed from the list: there too
        let t = Instant::now();
        let out = run(&env, call("bash", &[("command", "sleep 31 & echo started")]));
        assert!(t.elapsed() < Duration::from_secs(3), "{:?}", t.elapsed());
        assert!(out.ok && out.text.contains("started") && out.text.contains("background"), "{}", out.text);
        let list = env.procs.list();
        assert_eq!(list.len(), 1);
        assert!(env.procs.kill(list[0].id));
        assert!(eventually(|| env.procs.is_empty()));
        let ps = run(&env, call("bash", &[("command", "{ ps -eo args 2>/dev/null || ps -o args; } | grep -c '^sleep 3[01]' || true")]));
        assert_eq!(ps.text.trim(), "0", "{}", ps.text);

        // changed on both sides: a conflict, left alone
        assert!(run(&env, call("bash", &[("command", "echo there >> notes.txt; mkdir -p target && echo x > target/out")])).ok);
        std::fs::write(local.join("notes.txt"), "one\ntwo\nhere\n").unwrap();
        let changes = sb.changes().unwrap();
        let got: Vec<_> = changes.iter().map(|c| (c.path.as_str(), c.kind, c.conflict)).collect();
        assert_eq!(
            got,
            [
                ("new/b.txt", Change::Added, false),
                ("notes.txt", Change::Modified, true),
                ("src/a.rs", Change::Modified, false),
            ]
        );
        let diff = sb.diff().unwrap();
        let a = diff.iter().find(|d| d.change.path == "src/a.rs").unwrap();
        assert_eq!(count_changes(a.lines.as_ref().unwrap()), (1, 1));
        let report = sb.pull().unwrap();
        assert_eq!((report.files, report.conflicts.clone()), (2, vec!["notes.txt".to_string()]));
        assert!(std::fs::read_to_string(local.join("src/a.rs")).unwrap().contains("bye"));
        assert_eq!(std::fs::read_to_string(local.join("new/b.txt")).unwrap(), "b\n");
        assert_eq!(std::fs::read_to_string(local.join("notes.txt")).unwrap(), "one\ntwo\nhere\n");
        assert!(!local.join("target/out").exists());
        assert_eq!(sb.changes().unwrap().len(), 1);
        sb.stop();

        // kept: the next start copies only what changed here
        std::fs::write(local.join("c.txt"), "c\n").unwrap();
        std::fs::remove_file(local.join("new/b.txt")).unwrap();
        let mut said = Vec::new();
        let sb = Arc::new(Sandbox::start(&Spec::parse(again).unwrap(), &local, &mut |s| {
            eprintln!("  {s}");
            said.push(s.to_string())
        }).unwrap());
        assert!(said.iter().any(|s| s.contains("copied 1 file") && s.contains("1 deleted")), "{said:?}");
        assert!(said.iter().any(|s| s.contains("notes.txt")), "{said:?}");
        let env = Env { sandbox: Some(sb.clone()), cwd: sb.root.clone(), ..env };
        let out = run(&env, call("bash", &[("command", "cat c.txt notes.txt; ls new")]));
        assert!(out.text.contains("c\n") && out.text.contains("there"), "{}", out.text);
        assert!(!out.text.contains("b.txt"), "{}", out.text);
        // a host keeps its work directories: this one goes
        let root = sb.root.display().to_string();
        let out = sb.run(r#"rm -rf -- "$1" "$1.vcode-stamp""#, &[&root], None).unwrap();
        assert_eq!(out.code, Some(0), "{}", out.stderr);
        sb.stop();
        let _ = std::fs::remove_dir_all(base);
    }

    #[test]
    fn ansi() {
        assert_eq!(strip_ansi("\x1b[1;32mok\x1b[0m 10%\r50%\r100%\n"), "100%\n");
    }
}

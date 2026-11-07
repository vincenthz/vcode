//! The tools the model can call: their definitions, as the chat template
//! expects them, and how they run.
//!
//! A call first becomes a [`Plan`] (arguments checked, the change of an edit
//! computed), which the user may have to approve, then runs.

use std::fmt::Write as _;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime};

use biv::chat::ToolCall;

use crate::roles::Role;
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
        if let Some(rest) = path.strip_prefix("~/")
            && let Some(home) = std::env::var_os("HOME")
        {
            return PathBuf::from(home).join(rest);
        }
        self.cwd.join(path)
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
                let old = std::fs::read_to_string(&path).ok();
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
                let old = std::fs::read_to_string(&path)
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
                if let Some(dir) = path.parent() {
                    std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
                }
                std::fs::write(&path, &content).map_err(|e| e.to_string())?;
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
        if path.is_dir() {
            return Err(format!("{shown} is a directory, use list_dir"));
        }
        let bytes = std::fs::read(path).map_err(|e| format!("{shown}: {e}"))?;
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
        let shell = if Path::new("/bin/bash").exists() {
            "/bin/bash"
        } else {
            "/bin/sh"
        };
        let mut cmd = Command::new(shell);
        cmd.arg("-c")
            .arg(command)
            .current_dir(&self.cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PAGER", "cat")
            .env("GIT_PAGER", "cat")
            .env("NO_COLOR", "1")
            .env("TERM", "dumb");
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            // its own process group, to kill what it starts too
            cmd.process_group(0);
        }
        let mut child = match cmd.spawn() {
            Ok(c) => c,
            Err(e) => return failure(format!("{shell}: {e}")),
        };
        let output = Arc::new(Mutex::new(Vec::<u8>::new()));
        let readers: Vec<_> = [
            child
                .stdout
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
            child
                .stderr
                .take()
                .map(|s| Box::new(s) as Box<dyn Read + Send>),
        ]
        .into_iter()
        .flatten()
        .map(|mut pipe| {
            let output = output.clone();
            std::thread::spawn(move || {
                let mut buf = [0u8; 8192];
                while let Ok(n) = pipe.read(&mut buf) {
                    if n == 0 {
                        break;
                    }
                    output.lock().unwrap().extend_from_slice(&buf[..n]);
                }
            })
        })
        .collect();

        let start = Instant::now();
        let mut sent = 0;
        let mut killed = None;
        let status = loop {
            match child.try_wait() {
                Ok(Some(status)) => break Some(status),
                Ok(None) => {}
                Err(_) => break None,
            }
            if killed.is_none() {
                if self.stop.load(Ordering::SeqCst) {
                    killed = Some("interrupted by the user");
                } else if start.elapsed() > timeout {
                    killed = Some("timed out");
                }
                if killed.is_some() {
                    kill(&mut child);
                }
            }
            {
                let out = output.lock().unwrap();
                if out.len() > sent {
                    progress(&String::from_utf8_lossy(&out[sent..]));
                    sent = out.len();
                }
            }
            std::thread::sleep(Duration::from_millis(30));
        };
        for r in readers {
            let _ = r.join();
        }
        let out = output.lock().unwrap();
        if out.len() > sent {
            progress(&String::from_utf8_lossy(&out[sent..]));
        }
        let text = strip_ansi(&String::from_utf8_lossy(&out));
        let text = text.trim_end();
        let code = status.and_then(|s| s.code());
        let mut result = middle_out(text, MAX_OUTPUT);
        let ok = killed.is_none() && code == Some(0);
        let status = match (killed, code) {
            (Some(why), _) => format!("[{why} after {:.0}s]", start.elapsed().as_secs_f64()),
            (None, Some(0)) => String::new(),
            (None, Some(c)) => format!("[exit code {c}]"),
            (None, None) => "[killed by a signal]".into(),
        };
        if !status.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(&status);
        }
        if result.is_empty() {
            result = "(no output)".into();
        }
        let lines = text.lines().count();
        Output {
            summary: match (status.is_empty(), lines) {
                (true, 0) => "no output".into(),
                (true, n) => format!("{n} lines"),
                (false, _) => status.trim_matches(['[', ']']).to_string(),
            },
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
        let mut out = String::new();
        let mut matches = 0;
        let mut files = 0;
        'walk: for entry in walk(root) {
            let path = entry.path();
            if let Some(g) = include {
                let rel = path.strip_prefix(root).unwrap_or(path);
                if !g.is_match(rel) && !path.file_name().is_some_and(|n| g.is_match(n)) {
                    continue;
                }
            }
            let Ok(bytes) = std::fs::read(path) else {
                continue;
            };
            if bytes.iter().take(8192).any(|&b| b == 0) {
                continue;
            }
            let text = String::from_utf8_lossy(&bytes);
            let mut found = false;
            for (i, line) in text.lines().enumerate() {
                if re.is_match(line) {
                    if !found {
                        found = true;
                        files += 1;
                    }
                    matches += 1;
                    if matches > MAX_RESULTS {
                        break 'walk;
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
        let n = found.len();
        let mut out: String = found
            .iter()
            .take(MAX_RESULTS)
            .map(|(_, p)| self.show(p) + "\n")
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
        let mut entries: Vec<String> = std::fs::read_dir(path)
            .map_err(|e| format!("{shown}: {e}"))?
            .filter_map(Result::ok)
            .map(|e| {
                let name = e.file_name().to_string_lossy().into_owned();
                if e.path().is_dir() { name + "/" } else { name }
            })
            .collect();
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

/// A command the user runs (`!`), in `cwd`
pub fn shell(cwd: &Path, command: &str) -> Output {
    let env = Env {
        cwd: cwd.to_path_buf(),
        stop: Arc::new(AtomicBool::new(false)),
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

#[cfg(unix)]
fn kill(child: &mut std::process::Child) {
    // the whole process group, then the shell itself in case it left it
    let _ = Command::new("kill")
        .args(["-KILL", &format!("-{}", child.id())])
        .stderr(Stdio::null())
        .status();
    let _ = child.kill();
}

#[cfg(not(unix))]
fn kill(child: &mut std::process::Child) {
    let _ = child.kill();
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
fn middle_out(s: &str, max: usize) -> String {
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

    #[test]
    fn ansi() {
        assert_eq!(strip_ansi("\x1b[1;32mok\x1b[0m 10%\r50%\r100%\n"), "100%\n");
    }
}

//! The system prompts, of the agent and of its subagents: how to work, the
//! environment, the project's instructions.

use std::path::Path;
use std::process::Command;

use crate::roles::Role;

const BASE: &str = "\
You are vcode, a coding agent running in the user's terminal, on their machine. You help with \
software engineering tasks: understanding code, fixing bugs, adding features, refactoring, \
running builds and tests, explaining.

# How to work
- Look before you act: use grep, glob, list_dir and read_file to find and read the relevant code. \
Never guess what a file contains.
- Change files with edit_file (exact replacements of small parts) or write_file (new files, \
complete rewrites). Keep to the style of the surrounding code.
- After a change, check it when you can: build, run the tests or the program with bash, and fix \
what fails.
- Call tools until the task is done, then reply with a short summary of what you did and anything \
left to do. Don't stop halfway to ask whether to continue.
- Ask the user first before anything destructive or hard to undo (deleting files, rewriting git \
history, pushing). Don't commit unless asked.
- If a tool call fails, read the error and correct the call rather than repeating it.

# Replies
- Be brief and direct: no preamble, no flattery, no restating the question.
- Use Markdown. Put code in fenced blocks with a language. Refer to code as path:line.{TASK}";

/// How to delegate, in the agent's instructions when it has subagents
const TASK: &str = "

# Subagents
Your context is small and every file you read stays in it. When a question or a change needs \
more than two or three files read or several searches (how does X work, where is Y handled, \
what calls Z), don't read them yourself: give the investigation to a subagent with task. It works \
in a context of its own and returns a report, and you continue from the report. Its prompt must \
be complete, since it knows nothing of this conversation: the question, what you already know, \
and what to report (paths, line numbers, short excerpts). Read a file yourself only to change it \
or to check a detail of the report.";

const SUBAGENT: &str = "\
You are a subagent of vcode, a coding agent running in the user's terminal, on their machine. \
The main agent gave you a task. Do it on your own with the tools: nobody will answer questions.

# Your role: {ROLE}
{INSTRUCTIONS}

# How to work
- Look before you conclude: use grep, glob, list_dir and read_file. Never guess what a file \
contains.
- Be efficient: search before reading, and read the relevant parts of large files.
- If a tool call fails, read the error and correct the call rather than repeating it.

# Your final answer
Your last message, the one without tool calls, is all the main agent gets of your work. Make it a \
complete, self-contained report: paths with line numbers, short code excerpts only where they \
matter, and what you could not find or do. No preamble, and no more than about 500 words unless \
the task needs more.";

const PILOT: &str = "\
You are vcode in pilot mode, a coding agent running in the user's terminal, on their machine. You \
don't read or change files yourself: you direct subagents, each doing one well-defined piece of \
work in a context of its own, and you keep the overall view: what the project is, what the user \
wants, the plan, and where each task stands. Your context holds that view and the subagents' \
reports, not the code.

# Subagents (the task tool)
- explore: investigates the code and answers questions; changes nothing.
- implement: makes one change, then builds and tests it.
- verify: checks a change or a claim independently (builds, tests, runs it); reports PASS or FAIL.
- review: reviews changes for bugs and for differences from what was asked; changes nothing.
- formalize: makes requirements precise (invariants, edge cases, examples) and encodes them as tests.
- document: writes or updates documentation.
- general: anything else.
They know nothing of this conversation, and nothing of each other's work but what you tell \
them: every prompt must be complete: the goal, the files and facts you know from earlier \
reports, constraints and decisions, and what to report. Tasks run one at a time.

# How to work
1. Understand: unless the request is trivial, start with one or a few explore tasks on the parts \
of the project it involves.
2. Plan: record an overview of the project and the list of tasks with update_plan. Keep tasks \
small: one implement task makes one coherent change.
3. Work through the tasks one at a time. After an implement task, give the change to a verify \
task, and to a review task when it matters; if they find problems, give a new implement task \
with what they found. Update the plan as tasks start, finish or fail.
4. Use formalize when the requirements are subtle (algorithms, data formats, protocols, edge \
cases), before implementing or to test the result; use document when the change affects users \
or interfaces.
5. When the work is done, reply to the user with a short summary: what was done, what was \
verified and how, and what is left.

- Rely on reports, but not blindly: a change is done when a verify task passed.
- If the request is ambiguous or has several reasonable readings, ask the user before starting: \
answering without a tool call ends your turn.
- Ask before anything destructive or hard to undo (deleting files, rewriting history, pushing). \
Don't have subagents commit unless the user asked.
- Keep your own messages short. Use Markdown.";

/// Files read into the system prompt, the first found in the working directory
const INSTRUCTION_FILES: [&str; 3] = ["AGENTS.md", "VCODE.md", "CLAUDE.md"];

/// Bytes of the project's instructions kept at most
const MAX_INSTRUCTIONS: usize = 16_000;

/// The agent's, which may give tasks to subagents
pub fn system(cwd: &Path, subagents: bool) -> String {
    let mut s = BASE.replace("{TASK}", if subagents { TASK } else { "" });
    environment(cwd, &mut s);
    s
}

/// A subagent's, for its role
pub fn subagent(cwd: &Path, role: Role) -> String {
    let mut s = SUBAGENT
        .replace("{ROLE}", role.name())
        .replace("{INSTRUCTIONS}", role.instructions());
    environment(cwd, &mut s);
    s
}

/// The pilot's: an agent that only directs subagents
pub fn pilot(cwd: &Path) -> String {
    let mut s = String::from(PILOT);
    environment(cwd, &mut s);
    s
}

/// The working directory, the platform, the project's instructions
fn environment(cwd: &Path, s: &mut String) {
    s.push_str("\n\n# Environment\n");
    s.push_str(&format!("- Working directory: {}\n", cwd.display()));
    s.push_str(&format!(
        "- Platform: {} {}\n",
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    s.push_str(&format!("- Date: {}\n", today()));
    match git_branch(cwd) {
        Some(b) => s.push_str(&format!("- Git repository, on branch {b}\n")),
        None => s.push_str("- Not a git repository\n"),
    }
    let top = top_level(cwd);
    if !top.is_empty() {
        s.push_str(&format!("- Top-level entries: {top}\n"));
    }
    if let Some((name, text)) = instructions(cwd) {
        s.push_str(&format!(
            "\n# Project instructions ({name})\nFollow these instructions from the project.\n\n{}\n",
            text.trim()
        ));
    }
}

fn instructions(cwd: &Path) -> Option<(&'static str, String)> {
    INSTRUCTION_FILES.iter().find_map(|name| {
        let text = std::fs::read_to_string(cwd.join(name)).ok()?;
        let mut end = text.len().min(MAX_INSTRUCTIONS);
        while !text.is_char_boundary(end) {
            end -= 1;
        }
        Some((*name, text[..end].to_string()))
    })
}

fn git_branch(cwd: &Path) -> Option<String> {
    let out = Command::new("git")
        .args(["rev-parse", "--abbrev-ref", "HEAD"])
        .current_dir(cwd)
        .stderr(std::process::Stdio::null())
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}

/// The entries of the working directory, a few dozen at most
fn top_level(cwd: &Path) -> String {
    let Ok(rd) = std::fs::read_dir(cwd) else {
        return String::new();
    };
    let mut names: Vec<String> = rd
        .filter_map(Result::ok)
        .map(|e| {
            let n = e.file_name().to_string_lossy().into_owned();
            if e.path().is_dir() { n + "/" } else { n }
        })
        .filter(|n| n != ".git/" && n != ".DS_Store")
        .collect();
    names.sort();
    let n = names.len();
    names.truncate(40);
    let mut s = names.join(", ");
    if n > 40 {
        s.push_str(&format!(", and {} more", n - 40));
    }
    s
}

/// Today's date, YYYY-MM-DD in UTC
fn today() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // civil from days, Howard Hinnant's algorithm
    let z = (secs / 86400) as i64 + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!("{y:04}-{m:02}-{d:02}")
}

//! A trace of what the agents do, to a file: the prompts they are given, the
//! tokens they generate, the tool calls they make and their results.
//!
//! Every agent has a number, the main one 0 and each subagent the next, and
//! counts the prompts it is given (a message, the results of its calls): each
//! entry is tagged `[agent.prompt]`. The tokens are kept per agent and written
//! in chunks, when the stream changes (reasoning, answer), when anything else
//! of the agent is traced, or when a chunk gets long: agents working at the
//! same time don't mix their streams.
//!
//! Turned on with `--trace <file>` or `/debug <file>`, off with `/debug off`.
//! The file is appended to, and written as it goes (`tail -f` works).

use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Tracing is on: checked before taking the lock, for every token
static ON: AtomicBool = AtomicBool::new(false);
static TRACER: Mutex<Option<Tracer>> = Mutex::new(None);
/// The number of the next subagent: the main agent is 0
static AGENTS: AtomicUsize = AtomicUsize::new(1);

/// Bytes of a chunk of tokens past which it is written, at a line's end
const CHUNK: usize = 4096;

/// Who an entry is of: an agent, and the prompt it is answering
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tag {
    pub agent: usize,
    /// the prompts given to the agent so far, 0 before the first
    pub prompt: usize,
}

impl Tag {
    pub fn new(agent: usize) -> Tag {
        Tag { agent, prompt: 0 }
    }
}

impl fmt::Display for Tag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.agent, self.prompt)
    }
}

/// What tokens are of
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Stream {
    Thinking,
    Answer,
}

impl Stream {
    fn name(self) -> &'static str {
        match self {
            Stream::Thinking => "thinking",
            Stream::Answer => "answer",
        }
    }
}

/// Tokens of an agent not written yet
struct Chunk {
    tag: Tag,
    stream: Stream,
    /// seconds since the trace started, at the first token
    at: f64,
    text: String,
    /// the chunk follows one of the same stream, cut for its length
    more: bool,
}

struct Tracer {
    path: PathBuf,
    out: BufWriter<File>,
    start: Instant,
    /// per agent
    chunks: HashMap<usize, Chunk>,
}

impl Tracer {
    fn now(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    fn entry(&mut self, at: f64, tag: Option<Tag>, what: &str, body: &str) {
        let tag = tag.map_or(String::new(), |t| format!(" [{t}]"));
        let _ = writeln!(self.out, "{at:.3}s{tag} {what}");
        for line in body.trim_end().lines() {
            let _ = writeln!(self.out, "    {line}");
        }
        let _ = self.out.flush();
    }

    fn write_chunk(&mut self, c: Chunk) {
        if c.text.trim().is_empty() {
            return;
        }
        let what = format!(
            "{}{} ({} bytes)",
            c.stream.name(),
            if c.more { ", continued" } else { "" },
            c.text.len()
        );
        self.entry(c.at, Some(c.tag), &what, &c.text);
    }

    /// The agent's tokens not written yet, written
    fn flush(&mut self, agent: usize) {
        if let Some(c) = self.chunks.remove(&agent) {
            self.write_chunk(c);
        }
    }

    fn flush_all(&mut self) {
        let mut agents: Vec<usize> = self.chunks.keys().copied().collect();
        agents.sort();
        for a in agents {
            self.flush(a);
        }
    }
}

fn lock() -> MutexGuard<'static, Option<Tracer>> {
    TRACER.lock().unwrap_or_else(|e| e.into_inner())
}

/// Whether tracing is on
pub fn enabled() -> bool {
    ON.load(Ordering::Relaxed)
}

/// The file traced to, if tracing is on
pub fn path() -> Option<PathBuf> {
    lock().as_ref().map(|t| t.path.clone())
}

/// The number of a new subagent
pub fn next_agent() -> usize {
    AGENTS.fetch_add(1, Ordering::Relaxed)
}

/// Tracing on, to `path`, appended to; instead of the file traced to before
pub fn start(path: &Path) -> Result<(), String> {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut tracer = lock();
    if let Some(mut old) = tracer.take() {
        old.flush_all();
        old.entry(old.now(), None, "=== trace stopped", "");
    }
    let mut t = Tracer {
        path: path.to_path_buf(),
        out: BufWriter::new(file),
        start: Instant::now(),
        chunks: HashMap::new(),
    };
    let unix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    t.entry(
        0.0,
        None,
        &format!("=== trace started (unix time {unix}), entries tagged [agent.prompt]"),
        "",
    );
    *tracer = Some(t);
    ON.store(true, Ordering::Relaxed);
    Ok(())
}

/// Tracing off, the tokens not written yet written: the file traced to
pub fn stop() -> Option<PathBuf> {
    let mut tracer = lock();
    ON.store(false, Ordering::Relaxed);
    let mut t = tracer.take()?;
    t.flush_all();
    t.entry(t.now(), None, "=== trace stopped", "");
    Some(t.path)
}

/// Tokens generated by an agent, kept to be written in chunks
pub fn token(tag: Tag, stream: Stream, text: &str) {
    if !enabled() {
        return;
    }
    let mut tracer = lock();
    let Some(t) = tracer.as_mut() else { return };
    let same = t
        .chunks
        .get(&tag.agent)
        .is_some_and(|c| c.tag == tag && c.stream == stream);
    if !same {
        t.flush(tag.agent);
    }
    let at = t.now();
    let c = t.chunks.entry(tag.agent).or_insert_with(|| Chunk {
        tag,
        stream,
        at,
        text: String::new(),
        more: false,
    });
    c.text.push_str(text);
    if c.text.len() >= CHUNK {
        // cut at the end of a line if there is one, the rest kept
        let cut = c.text.rfind('\n').map_or(c.text.len(), |i| i + 1);
        let rest = c.text.split_off(cut);
        let done = std::mem::replace(
            c,
            Chunk {
                tag,
                stream,
                at,
                text: rest,
                more: true,
            },
        );
        t.write_chunk(done);
    }
}

/// Something an agent did, with its details: written after its tokens
pub fn event(tag: Tag, what: &str, body: &str) {
    if !enabled() {
        return;
    }
    let mut tracer = lock();
    let Some(t) = tracer.as_mut() else { return };
    t.flush(tag.agent);
    let at = t.now();
    t.entry(at, Some(tag), what, body);
}

/// The arguments of a call, one per line
pub fn arguments<'a>(args: impl IntoIterator<Item = &'a (String, String)>) -> String {
    let mut s = String::new();
    for (k, v) in args {
        if v.contains('\n') {
            s.push_str(&format!("{k}:\n"));
            for line in v.lines() {
                s.push_str(&format!("  {line}\n"));
            }
        } else {
            s.push_str(&format!("{k}: {v}\n"));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks() {
        let dir = std::env::temp_dir().join(format!("vcode-trace-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        let path = dir.join("trace.log");
        let _ = std::fs::remove_file(&path);
        start(&path).unwrap();
        let a = Tag { agent: 0, prompt: 1 };
        let b = Tag { agent: 1, prompt: 1 };
        event(a, "prompt", "hello");
        for w in ["let ", "me ", "think\n", "more"] {
            token(a, Stream::Thinking, w);
            token(b, Stream::Answer, w);
        }
        token(a, Stream::Answer, "the answer");
        event(a, "turn done", "");
        assert_eq!(stop(), Some(path.clone()));
        assert!(!enabled());
        token(a, Stream::Answer, "not traced");

        let log = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
        let entries: Vec<&str> = log
            .lines()
            .filter(|l| !l.starts_with("    "))
            .map(|l| l.split_once("s ").unwrap().1)
            .collect();
        assert!(entries[0].starts_with("=== trace started"));
        assert_eq!(
            entries[1..],
            [
                "[0.1] prompt",
                "[0.1] thinking (17 bytes)",
                "[0.1] answer (10 bytes)",
                "[0.1] turn done",
                "[1.1] answer (17 bytes)",
                "=== trace stopped",
            ]
        );
        assert!(log.contains("    let me think\n    more\n"));
        assert!(!log.contains("not traced"));
    }
}

//! The shell commands running: those of the agents' `bash` calls and the
//! user's `!` commands. The interface lists them (`/ps`, ctrl+p), and the
//! user can kill any of them, with the processes it started.
//!
//! A command whose shell exited but which left processes running that hold
//! its output (`server &`) is given back to the model then, and stays listed,
//! in the background, until those processes end or are killed.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Bytes of a command's output kept from its start
const HEAD: usize = 1 << 20;
/// Bytes kept from its end, past the start's
const TAIL: usize = 64 << 10;

/// The commands running, shared by the agents and the interface
#[derive(Clone, Default)]
pub struct Procs(Arc<Inner>);

#[derive(Default)]
struct Inner {
    next: AtomicUsize,
    jobs: Mutex<Vec<Arc<Job>>>,
}

/// A command running
pub struct Job {
    pub id: usize,
    pub command: String,
    /// who runs it: `agent`, `explore #12`, `you`
    pub owner: String,
    /// of its shell, which leads the process group of everything it starts
    pub pid: u32,
    pub start: Instant,
    killed: AtomicBool,
    background: AtomicBool,
    pub out: Mutex<Capture>,
}

/// A command as listed
pub struct Info {
    pub id: usize,
    pub command: String,
    pub owner: String,
    pub elapsed: Duration,
    /// its shell exited, processes it started still run
    pub background: bool,
    /// being killed
    pub killed: bool,
    /// the last line of its output
    pub last: String,
}

impl Procs {
    /// A command started, listed until [`Procs::remove`]
    pub fn add(&self, command: &str, owner: &str, pid: u32) -> Arc<Job> {
        let job = Arc::new(Job {
            id: self.0.next.fetch_add(1, Ordering::Relaxed) + 1,
            command: command.to_string(),
            owner: owner.to_string(),
            pid,
            start: Instant::now(),
            killed: AtomicBool::new(false),
            background: AtomicBool::new(false),
            out: Mutex::new(Capture::default()),
        });
        self.jobs().push(job.clone());
        job
    }

    pub fn remove(&self, id: usize) {
        self.jobs().retain(|j| j.id != id);
    }

    pub fn list(&self) -> Vec<Info> {
        self.jobs()
            .iter()
            .map(|j| Info {
                id: j.id,
                command: j.command.clone(),
                owner: j.owner.clone(),
                elapsed: j.start.elapsed(),
                background: j.background.load(Ordering::SeqCst),
                killed: j.killed(),
                last: j.out.lock().unwrap().last_line(),
            })
            .collect()
    }

    pub fn is_empty(&self) -> bool {
        self.jobs().is_empty()
    }

    /// Kill command `id` and what it started: false if it isn't running
    pub fn kill(&self, id: usize) -> bool {
        let job = self.jobs().iter().find(|j| j.id == id).cloned();
        job.map(|j| j.kill()).is_some()
    }

    /// Kill them all, when vcode exits
    pub fn kill_all(&self) {
        let jobs: Vec<_> = self.jobs().clone();
        for j in jobs {
            j.kill();
        }
    }

    fn jobs(&self) -> std::sync::MutexGuard<'_, Vec<Arc<Job>>> {
        self.0.jobs.lock().unwrap_or_else(|e| e.into_inner())
    }
}

impl Job {
    /// Kill its process group
    pub fn kill(&self) {
        self.killed.store(true, Ordering::SeqCst);
        kill_group(self.pid);
    }

    pub fn killed(&self) -> bool {
        self.killed.load(Ordering::SeqCst)
    }

    pub fn set_background(&self) {
        self.background.store(true, Ordering::SeqCst);
    }
}

/// Kill the process group led by `pid`. A process group's id isn't given to
/// a new process while the group has members, so this can't hit another.
#[cfg(unix)]
fn kill_group(pid: u32) {
    // `kill -0` would be our own group
    if pid == 0 {
        return;
    }
    let _ = std::process::Command::new("kill")
        .args(["-KILL", &format!("-{pid}")])
        .stderr(std::process::Stdio::null())
        .status();
}

#[cfg(not(unix))]
fn kill_group(pid: u32) {
    if pid == 0 {
        return;
    }
    let _ = std::process::Command::new("taskkill")
        .args(["/F", "/T", "/PID", &pid.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status();
}

/// A command's output, bounded: its first [`HEAD`] bytes, then its last
/// [`TAIL`] bytes or so
#[derive(Default)]
pub struct Capture {
    head: Vec<u8>,
    tail: Vec<u8>,
    /// bytes left out between the two
    dropped: usize,
}

impl Capture {
    pub fn push(&mut self, bytes: &[u8]) {
        let room = (HEAD - self.head.len()).min(bytes.len());
        self.head.extend_from_slice(&bytes[..room]);
        let rest = &bytes[room..];
        if rest.is_empty() {
            return;
        }
        self.tail.extend_from_slice(rest);
        if self.tail.len() > 2 * TAIL {
            let cut = self.tail.len() - TAIL;
            self.tail.drain(..cut);
            self.dropped += cut;
        }
    }

    /// The start of the output: what is streamed as it comes
    pub fn head(&self) -> &[u8] {
        &self.head
    }

    /// All that is kept of it
    pub fn text(&self) -> String {
        let mut s = String::from_utf8_lossy(&self.head).into_owned();
        if self.dropped > 0 {
            s.push_str(&format!("\n[... {} bytes omitted ...]\n", self.dropped));
        }
        s.push_str(&String::from_utf8_lossy(&self.tail));
        s
    }

    /// Its last line with text, as a terminal would show it
    pub fn last_line(&self) -> String {
        let end = if self.tail.is_empty() {
            &self.head
        } else {
            &self.tail
        };
        let from = end.len().saturating_sub(4096);
        let text = crate::tools::strip_ansi(&String::from_utf8_lossy(&end[from..]));
        text.lines()
            .rev()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or_default()
            .to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capture_is_bounded() {
        let mut c = Capture::default();
        c.push(b"start\n");
        let chunk = vec![b'x'; 100_000];
        for _ in 0..30 {
            c.push(&chunk);
        }
        c.push(b"\nthe end\n");
        assert_eq!(c.head().len(), HEAD);
        assert!(c.tail.len() <= 2 * TAIL);
        let text = c.text();
        assert!(text.starts_with("start\n"));
        assert!(text.contains("bytes omitted"));
        assert!(text.ends_with("the end\n"));
        assert_eq!(c.last_line(), "the end");
        assert_eq!(HEAD + c.dropped + c.tail.len(), 6 + 3_000_000 + 9);
    }

    #[test]
    fn listed_until_removed() {
        let procs = Procs::default();
        let a = procs.add("cargo test", "agent", 0);
        let b = procs.add("npm run dev", "you", 0);
        b.set_background();
        b.out.lock().unwrap().push(b"compiling\n\x1b[32mready on :3000\x1b[0m\n\n");
        let list = procs.list();
        assert_eq!(list.len(), 2);
        assert!(!list[0].background && list[1].background);
        assert_eq!(list[1].last, "ready on :3000");
        procs.remove(a.id);
        assert_eq!(procs.list().len(), 1);
        assert!(!procs.kill(a.id));
        procs.remove(b.id);
        assert!(procs.is_empty());
    }
}

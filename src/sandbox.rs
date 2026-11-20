//! Sandboxes: the agent's tools run on another machine, a host reached by
//! ssh or a VM run by [vemu](https://github.com/vincenthz/vemu), in a work
//! directory there that holds a copy of the project.
//!
//! The copy there is the one the tools see and change. It is made when the
//! sandbox starts, and kept between sessions: later starts copy only what
//! changed here since. Changes made there come back only when the user asks
//! (`/sandbox pull`).
//!
//! What was last agreed on by the two sides is kept here, in a manifest:
//! for each file, the CRC of its content (POSIX `cksum`'s, which the other
//! side can compute without copying anything), and its size and time of
//! modification here. A file changed on both sides since is a conflict, and
//! left as it is on both. Files changed there are found among those newer
//! than a stamp made at the first copy.
//!
//! Everything goes through one ssh connection (a ControlMaster), with only
//! POSIX tools on the other side (`sh`, `find`, `tar`, `cksum`, `ls`; `git`
//! when there is one), so that a minimal Alpine image will do.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use crate::tar;
use crate::tools::DiffLine;

/// Files larger than this aren't copied back to compare or show them
const MAX_FETCH: u64 = 8 << 20;
/// How long a VM may take to boot, until its ssh server answers
const BOOT_TIMEOUT: Duration = Duration::from_secs(120);

/// What to start, as given to `--sandbox`
#[derive(Debug, Clone, PartialEq)]
pub enum Spec {
    /// `ssh:[user@]host[:port][,dir=PATH]`
    Ssh {
        dest: String,
        port: Option<u16>,
        /// the work directory there, relative to the home directory unless
        /// absolute
        dir: Option<String>,
    },
    /// `vemu:NAME[,image=PATH,kernel=PATH,memory=SIZE,cpus=N,key=PATH,append=CMDLINE]`
    Vemu(VmSpec),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct VmSpec {
    pub name: String,
    pub image: Option<PathBuf>,
    pub kernel: Option<PathBuf>,
    pub memory: Option<String>,
    pub cpus: Option<u32>,
    pub key: Option<PathBuf>,
    pub append: Option<String>,
    /// the vemu binary
    pub vemu: Option<PathBuf>,
}

pub const SPEC_HELP: &str = "ssh:[user@]host[:port][,dir=PATH] or \
vemu:NAME[,image=PATH,kernel=PATH,memory=SIZE,cpus=N,key=PATH,vemu=PATH,append=CMDLINE]";

impl Spec {
    pub fn parse(s: &str) -> Result<Spec, String> {
        let bad = |why: &str| format!("{why}; expected {SPEC_HELP}");
        let (kind, rest) = s.split_once(':').ok_or_else(|| bad("no kind"))?;
        let (head, opts) = split_options(rest);
        if head.is_empty() {
            return Err(bad("no host or name"));
        }
        match kind {
            "ssh" => {
                let (dest, port) = match head.rsplit_once(':') {
                    Some((d, p)) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => {
                        (d.to_string(), Some(p.parse().map_err(|_| bad("invalid port"))?))
                    }
                    _ => (head.to_string(), None),
                };
                let mut dir = None;
                for (k, v) in opts {
                    match k.as_str() {
                        "dir" => dir = Some(v),
                        _ => return Err(bad(&format!("unknown option {k} for ssh"))),
                    }
                }
                Ok(Spec::Ssh { dest, port, dir })
            }
            "vemu" | "vm" => {
                if !head
                    .bytes()
                    .all(|b| b.is_ascii_alphanumeric() || b"._-".contains(&b))
                    || head.starts_with('.')
                {
                    return Err(bad("a VM's name is letters, digits, '.', '_' and '-'"));
                }
                let mut v = VmSpec {
                    name: head.to_string(),
                    ..Default::default()
                };
                for (k, val) in opts {
                    match k.as_str() {
                        "image" => v.image = Some(host_path(&val)),
                        "kernel" => v.kernel = Some(host_path(&val)),
                        "key" => v.key = Some(host_path(&val)),
                        "vemu" => v.vemu = Some(host_path(&val)),
                        "memory" | "mem" => v.memory = Some(val),
                        "cpus" => {
                            v.cpus = Some(
                                val.parse()
                                    .ok()
                                    .filter(|&n| n > 0)
                                    .ok_or_else(|| bad("cpus: a number"))?,
                            )
                        }
                        "append" => v.append = Some(val),
                        _ => return Err(bad(&format!("unknown option {k} for vemu"))),
                    }
                }
                Ok(Spec::Vemu(v))
            }
            _ => Err(bad(&format!("unknown kind {kind}"))),
        }
    }
}

/// `head,key=value,...`: `append=` takes the rest, commas included
fn split_options(s: &str) -> (&str, Vec<(String, String)>) {
    let (head, mut rest) = s.split_once(',').unwrap_or((s, ""));
    let mut opts = Vec::new();
    while !rest.is_empty() {
        if let Some(v) = rest.strip_prefix("append=") {
            opts.push(("append".into(), v.into()));
            break;
        }
        let (opt, more) = rest.split_once(',').unwrap_or((rest, ""));
        let (k, v) = opt.split_once('=').unwrap_or((opt, ""));
        opts.push((k.trim().to_string(), v.trim().to_string()));
        rest = more;
    }
    (head.trim(), opts)
}

/// A path of this machine given by the user: `~` expanded, made absolute
fn host_path(p: &str) -> PathBuf {
    let path = match p.strip_prefix("~/") {
        Some(rest) => home().join(rest),
        None => PathBuf::from(p),
    };
    std::path::absolute(&path).unwrap_or(path)
}

fn home() -> PathBuf {
    std::env::var_os("HOME").map_or_else(|| PathBuf::from("/"), PathBuf::from)
}

/// Where vcode keeps its VMs and what it knows of each work directory
fn data_dir() -> PathBuf {
    match std::env::var_os("XDG_DATA_HOME").filter(|d| !d.is_empty()) {
        Some(d) => PathBuf::from(d).join("vcode"),
        None => home().join(".local/share/vcode"),
    }
}

/// `s` quoted for a POSIX shell
pub fn sq(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

fn fnv(s: &str) -> u64 {
    s.bytes().fold(0xcbf2_9ce4_8422_2325, |h, b| {
        (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
    })
}

/// A name for the project at `local`: its directory's name, and a hash of
/// its path, so that two projects of the same name don't share one
fn project_id(local: &Path) -> String {
    let name: String = local
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "root".into())
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!(
        "{name}-{:06x}",
        fnv(&local.to_string_lossy()) & 0xff_ffff
    )
}

/// A sandbox started: where the tools run
pub struct Sandbox {
    /// as the user named it: `vemu:dev`, `ssh:devbox`
    pub label: String,
    /// a VM of its own, rather than a host shared with other things
    pub vm: bool,
    /// the work directory there, absolute
    pub root: PathBuf,
    /// the project's directory here
    pub local: PathBuf,
    /// the home directory there
    pub home: PathBuf,
    /// `uname -sm` there
    pub platform: String,
    /// ssh's options and destination
    ssh: Vec<OsString>,
    /// the ControlMaster's socket
    control: PathBuf,
    /// the manifest of the work directory, here
    manifest: PathBuf,
    /// the stamp file there: files newer may have changed
    stamp: String,
    /// marks the lines of the shell wrapping the agent's commands
    nonce: String,
    /// the rules of the project's .gitignore files, for the other side
    ignore: Mutex<Ignore>,
    vm_proc: Mutex<Option<Machine>>,
    /// one copy at a time
    sync: Mutex<()>,
    stopped: std::sync::atomic::AtomicBool,
}

impl Drop for Sandbox {
    /// A VM isn't left running, even when vcode panics
    fn drop(&mut self) {
        self.stop();
    }
}

/// A vemu process
struct Machine {
    child: Child,
    /// held open: vemu's console reads it
    _stdin: Option<ChildStdin>,
    pidfile: PathBuf,
}

/// The output of a script run there
pub struct Out {
    pub code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: String,
}

impl Sandbox {
    /// Start the sandbox, copy the project at `local` there, and say how it goes
    pub fn start(spec: &Spec, local: &Path, say: &mut dyn FnMut(&str)) -> Result<Sandbox, String> {
        let project = project_id(local);
        let control = control_path(&project);
        let base = vec![
            OsString::from("-o"),
            format!("ControlPath={}", control.display()).into(),
            "-o".into(),
            "ServerAliveInterval=15".into(),
            "-o".into(),
            "ServerAliveCountMax=4".into(),
            "-o".into(),
            "LogLevel=ERROR".into(),
        ];
        let (label, ssh, state, dir, machine) = match spec {
            Spec::Ssh { dest, port, dir } => {
                let mut ssh = base;
                if let Some(p) = port {
                    ssh.extend(["-p".into(), p.to_string().into()]);
                }
                ssh.push(dest.into());
                let label = match port {
                    Some(p) => format!("ssh:{dest}:{p}"),
                    None => format!("ssh:{dest}"),
                };
                let state = data_dir().join("hosts").join(safe(&label[4..]));
                std::fs::create_dir_all(&state).map_err(|e| format!("{}: {e}", state.display()))?;
                say(&format!("connecting to {dest}…"));
                // interactive: a password or a passphrase can be asked for
                connect(&ssh, &state.join("ssh.log"), true)?;
                let dir = dir.clone().unwrap_or(format!("vcode/{project}"));
                (label, ssh, state, dir, None)
            }
            Spec::Vemu(v) => {
                let (machine, ssh, state) = boot(v, base, say)?;
                let dir = format!("/work/{project}");
                (format!("vemu:{}", v.name), ssh, state, dir, Some(machine))
            }
        };
        let mut sb = Sandbox {
            label,
            vm: machine.is_some(),
            root: PathBuf::new(),
            local: local.to_path_buf(),
            home: PathBuf::new(),
            platform: String::new(),
            ssh,
            control,
            manifest: state.join(format!("{project}.manifest")),
            stamp: String::new(),
            nonce: format!(
                "vcode{:012x}",
                fnv(&format!("{:?}{}", Instant::now(), std::process::id())) & 0xffff_ffff_ffff
            ),
            ignore: Mutex::new(Ignore::default()),
            vm_proc: Mutex::new(machine),
            sync: Mutex::new(()),
            stopped: Default::default(),
        };
        // stopped if anything below fails
        let fail = |sb: &Sandbox, e: String| {
            sb.stop();
            e
        };
        let probe = r#"set -e
case "$1" in /*) ;; *) cd ;; esac
mkdir -p -- "$1"
cd -- "$1"
pwd
printf '%s\n' "$HOME"
uname -sm"#;
        let out = sb.run(probe, &[&dir], None).map_err(|e| fail(&sb, e))?;
        if out.code != Some(0) {
            let e = format!("can't make the work directory {dir}: {}", clean(&out.stderr));
            return Err(fail(&sb, e));
        }
        let text = String::from_utf8_lossy(&out.stdout).into_owned();
        let mut lines = text.lines();
        sb.root = PathBuf::from(lines.next().unwrap_or_default());
        sb.home = PathBuf::from(lines.next().unwrap_or_default());
        sb.platform = lines.next().unwrap_or_default().to_string();
        sb.stamp = format!("{}.vcode-stamp", sb.root.display());
        if !sb.root.is_absolute() {
            return Err(fail(&sb, format!("unexpected answer from {}: {text}", sb.label)));
        }
        let report = sb.push(say).map_err(|e| fail(&sb, e))?;
        say(&report.describe("copied", "to", &sb.label));
        Ok(sb)
    }

    /// How the system prompt describes it
    pub fn describe(&self) -> String {
        if self.vm {
            format!(
                "a virtual machine of its own ({}), where you are root and may install what you need",
                self.label
            )
        } else {
            format!("the machine {}, over ssh", &self.label[4..])
        }
    }

    fn client(&self) -> Command {
        let mut c = Command::new("ssh");
        c.args(["-T", "-o", "ControlMaster=no", "-o", "BatchMode=yes"])
            .args(&self.ssh);
        c
    }

    /// `script` run there by `sh`, with `args` as its `$1`, `$2`…
    pub fn command(&self, script: &str, args: &[&str]) -> Command {
        let mut remote = format!("exec sh -c {} sh", sq(script));
        for a in args {
            remote.push(' ');
            remote.push_str(&sq(a));
        }
        let mut c = self.client();
        c.arg(remote);
        c
    }

    /// Run `script` there, `input` on its standard input
    pub fn run(&self, script: &str, args: &[&str], input: Option<&[u8]>) -> Result<Out, String> {
        let input = input.map(<[u8]>::to_vec);
        self.run_with(script, args, input.map(|data| {
            Box::new(move |w: &mut dyn Write| w.write_all(&data)) as Feed
        }))
    }

    fn run_with(&self, script: &str, args: &[&str], feed: Option<Feed>) -> Result<Out, String> {
        let mut c = self.command(script, args);
        c.stdin(if feed.is_some() {
            Stdio::piped()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
        let mut child = c.spawn().map_err(|e| format!("ssh: {e}"))?;
        // written from a thread of its own, while the output is read
        let writer = feed.map(|feed| {
            let mut stdin = child.stdin.take().expect("piped");
            std::thread::spawn(move || {
                let r = feed(&mut stdin);
                drop(stdin);
                r
            })
        });
        let out = child.wait_with_output().map_err(|e| format!("ssh: {e}"))?;
        if let Some(w) = writer
            && let Ok(Err(e)) = w.join()
            && e.kind() != io::ErrorKind::BrokenPipe
        {
            return Err(e.to_string());
        }
        let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
        if out.status.code() == Some(255) {
            return Err(format!(
                "ssh to {}: {}",
                self.label,
                match stderr.trim() {
                    "" => "the connection failed",
                    e => e,
                }
            ));
        }
        Ok(Out {
            code: out.status.code(),
            stdout: out.stdout,
            stderr,
        })
    }

    fn arg(path: &Path) -> String {
        path.to_string_lossy().into_owned()
    }

    /// The content of a file there
    pub fn read(&self, path: &Path) -> io::Result<Vec<u8>> {
        let script = r#"if [ -d "$1" ]; then exit 3; fi; exec cat -- "$1""#;
        let out = self.run(script, &[&Self::arg(path)], None).map_err(io::Error::other)?;
        match out.code {
            Some(0) => Ok(out.stdout),
            Some(3) => Err(io::Error::new(io::ErrorKind::IsADirectory, "is a directory")),
            _ => {
                let e = clean(&out.stderr);
                let kind = if e.contains("No such file") {
                    io::ErrorKind::NotFound
                } else {
                    io::ErrorKind::Other
                };
                Err(io::Error::new(kind, e))
            }
        }
    }

    /// Write a file there, making its directory
    pub fn write(&self, path: &Path, data: &[u8]) -> Result<(), String> {
        let script = r#"d=${1%/*}; if [ -n "$d" ]; then mkdir -p -- "$d" || exit 1; fi; cat > "$1""#;
        let out = self.run(script, &[&Self::arg(path)], Some(data))?;
        match out.code {
            Some(0) => Ok(()),
            _ => Err(clean(&out.stderr)),
        }
    }

    /// The entries of a directory there, directories ending with `/`
    pub fn list(&self, path: &Path) -> Result<Vec<String>, String> {
        let out = self.run(r#"cd -- "$1" || exit 1; exec ls -Ap"#, &[&Self::arg(path)], None)?;
        if out.code != Some(0) {
            return Err(clean(&out.stderr));
        }
        Ok(String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| !l.is_empty())
            .map(str::to_string)
            .collect())
    }

    /// The files under `root` there (or `root` itself, a file), without
    /// those git ignores and the `.git` directory: as the local walk finds them
    pub fn files(&self, root: &Path) -> Result<Vec<PathBuf>, String> {
        let script = r#"if [ -f "$1" ]; then echo F; exit 0; fi
cd -- "$1" || exit 1
if command -v git >/dev/null 2>&1 && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo G
  git ls-files -z -co --exclude-standard | tr '\0' '\n' | while IFS= read -r f; do
    [ -f "$f" ] && printf '%s\n' "$f"
  done
else
  echo N
  find . -name .git -prune -o -type f -print | sed 's|^\./||'
fi"#;
        let out = self.run(script, &[&Self::arg(root)], None)?;
        if out.code != Some(0) {
            return Err(clean(&out.stderr));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let mut lines = text.lines();
        let tag = lines.next().unwrap_or_default();
        if tag == "F" {
            return Ok(vec![root.to_path_buf()]);
        }
        // without git there, the rules of the project's .gitignore files
        let ignore = self.ignore.lock().unwrap_or_else(|e| e.into_inner());
        let under = root.strip_prefix(&self.root).ok().map(Path::to_path_buf);
        Ok(lines
            .filter(|l| !l.is_empty())
            .filter(|l| {
                tag != "N"
                    || under.as_ref().is_none_or(|u| {
                        !ignore.ignored(&u.join(l).to_string_lossy(), false)
                    })
            })
            .map(|l| root.join(l))
            .collect())
    }

    /// The files at `paths` there, given to `f` as they come until it
    /// returns false; those larger than `max` bytes, or missing, are left out
    pub fn fetch(
        &self,
        paths: &[PathBuf],
        max: u64,
        f: &mut dyn FnMut(PathBuf, tar::Entry) -> bool,
    ) -> Result<(), String> {
        if paths.is_empty() {
            return Ok(());
        }
        let script = format!(
            r#"cd / && tr '\n' '\0' | xargs -0 sh -c '[ $# -gt 0 ] && exec find "$@" -prune -type f -size -{} -print' sh 2>/dev/null | tar -cf - -T - 2>/dev/null"#,
            max / 512 + 1
        );
        let list: String = paths
            .iter()
            .map(|p| {
                let s = Self::arg(p);
                format!("{}\n", s.trim_start_matches('/'))
            })
            .collect();
        let mut child = self
            .command(&script, &[])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .map_err(|e| format!("ssh: {e}"))?;
        let mut stdin = child.stdin.take().expect("piped");
        let writer = std::thread::spawn(move || {
            let _ = stdin.write_all(list.as_bytes());
        });
        let stdout = child.stdout.take().expect("piped");
        let mut stopped = false;
        let r = tar::read(io::BufReader::new(stdout), &mut |e| {
            if !e.file {
                return true;
            }
            let path = PathBuf::from(format!("/{}", e.path.trim_start_matches("./")));
            stopped = !f(path, e);
            !stopped
        });
        if stopped {
            let _ = child.kill();
        }
        let _ = writer.join();
        let status = child.wait().map_err(|e| e.to_string())?;
        if !stopped && status.code() == Some(255) {
            return Err(format!("ssh to {}: the connection failed", self.label));
        }
        r.map_err(|e| e.to_string())
    }

    /// `paths` there, the most recently modified first
    pub fn newest_first(&self, paths: Vec<PathBuf>) -> Vec<PathBuf> {
        let script = r#"cd / && tr '\n' '\0' | xargs -0 sh -c '[ $# -gt 0 ] && exec ls -td "$@"' sh"#;
        let list: String = paths
            .iter()
            .map(|p| format!("{}\n", Self::arg(p).trim_start_matches('/')))
            .collect();
        let Ok(out) = self.run(script, &[], Some(list.as_bytes())) else {
            return paths;
        };
        let mut sorted: Vec<PathBuf> = String::from_utf8_lossy(&out.stdout)
            .lines()
            .map(|l| PathBuf::from(format!("/{l}")))
            .collect();
        let seen: HashSet<_> = sorted.iter().cloned().collect();
        sorted.extend(paths.into_iter().filter(|p| !seen.contains(p)));
        sorted
    }

    /// The command running `command` there with bash (sh without), in
    /// `cwd`. Its standard error starts with a mark giving its process group
    /// (to kill it with [`Sandbox::kill_group`]), and ends with one giving its
    /// exit status: ssh lasts as long as something holds the output, which
    /// processes left in the background do. [`Demux`] takes them out.
    pub fn shell(&self, cwd: &Path, command: &str) -> Command {
        let n = &self.nonce;
        let script = format!(
            r#"printf '\036{n} pid %s\036\n' "$$" >&2
export PAGER=cat GIT_PAGER=cat NO_COLOR=1 TERM=dumb
cd -- "$1" || {{ printf '\036{n} exit 1\036\n' >&2; exit 1; }}
if command -v bash >/dev/null 2>&1; then bash -c "$2" </dev/null; else sh -c "$2" </dev/null; fi
s=$?
printf '\036{n} exit %s\036\n' "$s" >&2
exit "$s""#
        );
        self.command(&script, &[&Self::arg(cwd), command])
    }

    pub fn demux(&self) -> Demux {
        Demux::new(&self.nonce)
    }

    /// Kill the process group `pgid` there
    pub fn kill_group(&self, pgid: u32) {
        let _ = self.run(
            r#"kill -KILL -"$1" 2>/dev/null || kill -s KILL -- -"$1""#,
            &[&pgid.to_string()],
            None,
        );
    }

    /// Stop it: power the VM off, close the connection
    pub fn stop(&self) {
        if self.stopped.swap(true, std::sync::atomic::Ordering::SeqCst) {
            return;
        }
        let machine = self.vm_proc.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(mut m) = machine {
            let _ = self.run("sync; poweroff", &[], None);
            let t = Instant::now();
            while t.elapsed() < Duration::from_secs(20) {
                if let Ok(Some(_)) = m.child.try_wait() {
                    break;
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            if let Ok(None) = m.child.try_wait() {
                let _ = m.child.kill();
                let _ = m.child.wait();
            }
            let _ = std::fs::remove_file(&m.pidfile);
        }
        let _ = Command::new("ssh")
            .args(["-O", "exit"])
            .args(&self.ssh)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        let _ = std::fs::remove_file(&self.control);
    }
}

type Feed = Box<dyn FnOnce(&mut dyn Write) -> io::Result<()> + Send>;

/// A file name made of `s`
fn safe(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || "._-@".contains(c) {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// The last part of an error message of a remote tool: `cat: can't open
/// 'x': No such file or directory` is `No such file or directory`
fn clean(stderr: &str) -> String {
    let line = stderr
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("failed");
    line.rsplit(": ").next().unwrap_or(line).to_string()
}

/// The ControlMaster's socket: short, as sockets' paths are limited
fn control_path(project: &str) -> PathBuf {
    let name = format!("vcode-{}-{:x}.ssh", std::process::id(), fnv(project) & 0xffff);
    let tmp = std::env::temp_dir().join(&name);
    if tmp.as_os_str().len() < 90 {
        tmp
    } else {
        PathBuf::from("/tmp").join(name)
    }
}

/// Open the ControlMaster connection that every command then goes through
fn connect(ssh: &[OsString], log: &Path, interactive: bool) -> Result<(), String> {
    let _ = std::fs::write(log, "");
    let mut c = Command::new("ssh");
    c.args(["-f", "-N", "-o", "ControlMaster=yes", "-o", "ControlPersist=yes"])
        .arg("-E")
        .arg(log);
    if !interactive {
        c.args(["-o", "BatchMode=yes", "-o", "ConnectTimeout=3"]);
    }
    c.args(ssh)
        .stdin(if interactive {
            Stdio::inherit()
        } else {
            Stdio::null()
        })
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    let status = c.status().map_err(|e| format!("ssh: {e}"))?;
    if status.success() {
        return Ok(());
    }
    let log = std::fs::read_to_string(log).unwrap_or_default();
    Err(match log.trim() {
        "" => "ssh failed".into(),
        l => l.to_string(),
    })
}

/// What `vm.conf` keeps of a VM, from when it was made
#[derive(Default)]
struct VmConf {
    image: Option<PathBuf>,
    kernel: Option<PathBuf>,
    memory: Option<String>,
    cpus: Option<u32>,
    key: Option<PathBuf>,
    append: Option<String>,
}

impl VmConf {
    fn load(path: &Path) -> Option<VmConf> {
        let text = std::fs::read_to_string(path).ok()?;
        let mut c = VmConf::default();
        for line in text.lines() {
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            let v = v.trim().to_string();
            match k.trim() {
                "image" => c.image = Some(v.into()),
                "kernel" => c.kernel = Some(v.into()),
                "memory" => c.memory = Some(v),
                "cpus" => c.cpus = v.parse().ok(),
                "key" => c.key = Some(v.into()),
                "append" => c.append = Some(v),
                _ => {}
            }
        }
        Some(c)
    }

    fn save(&self, path: &Path) -> io::Result<()> {
        let mut s = String::from("# a VM of vcode's sandboxes: --sandbox vemu:NAME\n");
        let mut put = |k: &str, v: Option<String>| {
            if let Some(v) = v {
                s.push_str(&format!("{k} = {v}\n"));
            }
        };
        put("image", self.image.as_ref().map(|p| p.display().to_string()));
        put("kernel", self.kernel.as_ref().map(|p| p.display().to_string()));
        put("memory", self.memory.clone());
        put("cpus", self.cpus.map(|n| n.to_string()));
        put("key", self.key.as_ref().map(|p| p.display().to_string()));
        put("append", self.append.clone());
        std::fs::write(path, s)
    }
}

/// Boot the VM of `v`, made on first use: its overlay on the golden image,
/// and a free port of this host forwarded to its ssh server
fn boot(
    v: &VmSpec,
    base: Vec<OsString>,
    say: &mut dyn FnMut(&str),
) -> Result<(Machine, Vec<OsString>, PathBuf), String> {
    let dir = data_dir().join("vms").join(&v.name);
    let conf_path = dir.join("vm.conf");
    let disk = dir.join("disk.vov");
    let vemu = v
        .vemu
        .clone()
        .or_else(|| std::env::var_os("VCODE_VEMU").map(PathBuf::from))
        .unwrap_or_else(|| "vemu".into());
    let mut conf = VmConf::load(&conf_path).unwrap_or_default();
    let new = !disk.exists();
    if new && (v.image.is_none() || v.kernel.is_none()) {
        return Err(format!(
            "there is no VM {name} yet: make it with --sandbox vemu:{name},image=GOLDEN.raw,kernel=Image \
             (an image from `vemu build`, and the kernel it was built with)",
            name = v.name
        ));
    }
    // what is given now replaces what was kept
    if v.image.is_some() && !new {
        say("note: the VM exists, image= is ignored (delete it to make it again)");
    } else if let Some(i) = &v.image {
        conf.image = Some(i.clone());
    }
    conf.kernel = v.kernel.clone().or(conf.kernel);
    conf.memory = v.memory.clone().or(conf.memory);
    conf.cpus = v.cpus.or(conf.cpus);
    conf.key = v.key.clone().or(conf.key);
    conf.append = v.append.clone().or(conf.append);
    let kernel = conf.kernel.clone().ok_or("vm.conf has no kernel")?;
    if !kernel.is_file() {
        return Err(format!("kernel {}: not found", kernel.display()));
    }
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    if new {
        let image = conf.image.clone().ok_or("no image")?;
        say(&format!("making the VM {} on {}…", v.name, image.display()));
        let out = Command::new(&vemu)
            .args(["overlay", "create"])
            .arg(&image)
            .arg(&disk)
            .output()
            .map_err(|e| format!("{}: {e}", vemu.display()))?;
        if !out.status.success() {
            return Err(format!(
                "vemu overlay create: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
    }
    conf.save(&conf_path)
        .map_err(|e| format!("{}: {e}", conf_path.display()))?;

    let pidfile = dir.join("vemu.pid");
    if let Some(pid) = std::fs::read_to_string(&pidfile)
        .ok()
        .and_then(|s| s.trim().parse::<u32>().ok())
        && running_vemu(pid)
    {
        return Err(format!(
            "the VM {} is already running (vemu, pid {pid}): another vcode uses it",
            v.name
        ));
    }
    let memory = conf.memory.clone().unwrap_or_else(|| "2G".into());
    let cpus = conf.cpus.unwrap_or_else(|| {
        std::thread::available_parallelism().map_or(2, |n| n.get().min(4) as u32)
    });
    let append = conf.append.clone().unwrap_or_else(|| "root=/dev/vda rw".into());
    let log_path = dir.join("console.log");
    let alias = format!("vcode-vm-{}", v.name);
    for attempt in 0..3 {
        let port = std::net::TcpListener::bind("127.0.0.1:0")
            .and_then(|l| l.local_addr())
            .map_err(|e| format!("no free port: {e}"))?
            .port();
        let log = std::fs::File::create(&log_path)
            .map_err(|e| format!("{}: {e}", log_path.display()))?;
        let err = log.try_clone().map_err(|e| e.to_string())?;
        say(&format!(
            "booting the VM {} ({cpus} cpus, {memory}; console: {})…",
            v.name,
            log_path.display()
        ));
        let mut child = Command::new(&vemu)
            .arg("--kernel")
            .arg(&kernel)
            .arg("--disk")
            .arg(format!("{},format=overlay", disk.display()))
            .args(["--append", &append, "--memory", &memory, "--cpus", &cpus.to_string()])
            .args(["-p", &format!("127.0.0.1:{port}:22"), "--serial", "stdio"])
            .stdin(Stdio::piped())
            .stdout(log)
            .stderr(err)
            .spawn()
            .map_err(|e| format!("{}: {e}", vemu.display()))?;
        let _ = std::fs::write(&pidfile, child.id().to_string());
        let stdin = child.stdin.take();
        let mut ssh = base.clone();
        ssh.extend(
            [
                "-p",
                &port.to_string(),
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
            ]
            .map(OsString::from),
        );
        ssh.push(format!("UserKnownHostsFile={}", dir.join("known_hosts").display()).into());
        ssh.extend(["-o".into(), format!("HostKeyAlias={alias}").into()]);
        if let Some(key) = &conf.key {
            ssh.extend(["-i".into(), key.into(), "-o".into(), "IdentitiesOnly=yes".into()]);
        }
        ssh.push("root@127.0.0.1".into());
        let t = Instant::now();
        let mut last = String::new();
        let exited = loop {
            if let Ok(Some(status)) = child.try_wait() {
                break Some(status);
            }
            if t.elapsed() > BOOT_TIMEOUT {
                let _ = child.kill();
                let _ = child.wait();
                let _ = std::fs::remove_file(&pidfile);
                return Err(format!(
                    "the VM's ssh server didn't answer in {}s ({last}); its console: {}\n{}",
                    BOOT_TIMEOUT.as_secs(),
                    log_path.display(),
                    tail(&log_path)
                ));
            }
            match connect(&ssh, &dir.join("ssh.log"), false) {
                Ok(()) => {
                    say(&format!("VM ready in {:.1}s", t.elapsed().as_secs_f64()));
                    let machine = Machine {
                        child,
                        _stdin: stdin,
                        pidfile,
                    };
                    let state = dir.join("projects");
                    std::fs::create_dir_all(&state).map_err(|e| e.to_string())?;
                    return Ok((machine, ssh, state));
                }
                Err(e) => last = e,
            }
            std::thread::sleep(Duration::from_millis(300));
        };
        let _ = std::fs::remove_file(&pidfile);
        let console = tail(&log_path);
        // the port taken in the meantime: another one
        if attempt < 2 && (console.contains("in use") && !console.contains("image is in use")) {
            continue;
        }
        return Err(format!(
            "vemu exited ({}):\n{console}",
            exited.map_or("?".into(), |s| s.to_string())
        ));
    }
    Err("vemu could not start".into())
}

/// Whether `pid` is a vemu still running
fn running_vemu(pid: u32) -> bool {
    Command::new("ps")
        .args(["-p", &pid.to_string(), "-o", "comm="])
        .output()
        .is_ok_and(|o| o.status.success() && String::from_utf8_lossy(&o.stdout).contains("vemu"))
}

/// The last lines of a file
fn tail(path: &Path) -> String {
    let text = std::fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(15)..].join("\n")
}

/// A mark of the shell wrapping a command there
#[derive(Debug, PartialEq, Eq)]
pub enum Mark {
    /// its process group
    Pid(u32),
    /// it exited
    Exit(i32),
}

/// Takes the marks out of a command's standard error
pub struct Demux {
    tag: Vec<u8>,
    pending: Vec<u8>,
}

impl Demux {
    fn new(nonce: &str) -> Self {
        Demux {
            tag: format!("\x1e{nonce} ").into_bytes(),
            pending: Vec::new(),
        }
    }

    /// The output in `data`, without the marks, which go to `marks`
    pub fn feed(&mut self, data: &[u8], marks: &mut Vec<Mark>) -> Vec<u8> {
        self.pending.extend_from_slice(data);
        let mut out = Vec::new();
        loop {
            let Some(i) = self.pending.iter().position(|&b| b == 0x1e) else {
                out.append(&mut self.pending);
                return out;
            };
            out.extend(self.pending.drain(..i));
            let n = self.tag.len().min(self.pending.len());
            if self.pending[..n] != self.tag[..n] {
                out.extend(self.pending.drain(..1));
                continue;
            }
            if self.pending.len() < self.tag.len() {
                return out;
            }
            let body = &self.pending[self.tag.len()..];
            match body.windows(2).position(|w| w == b"\x1e\n") {
                Some(end) => {
                    let text = String::from_utf8_lossy(&body[..end]).into_owned();
                    match text.split_once(' ') {
                        Some(("pid", p)) if let Ok(p) = p.parse() => marks.push(Mark::Pid(p)),
                        Some(("exit", c)) if let Ok(c) = c.parse() => marks.push(Mark::Exit(c)),
                        _ => {}
                    }
                    self.pending.drain(..self.tag.len() + end + 2);
                }
                // not a mark after all
                None if body.len() > 64 => out.extend(self.pending.drain(..1)),
                None => return out,
            }
        }
    }

    /// What is left at the end of the output
    pub fn finish(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.pending)
    }
}

/// The rules of the project's .gitignore files (and .git/info/exclude)
#[derive(Default)]
struct Ignore {
    local: PathBuf,
    rules: Vec<ignore::gitignore::Gitignore>,
}

impl Ignore {
    fn new(local: &Path, files: &[PathBuf]) -> Ignore {
        let mut rules = Vec::new();
        let mut root = ignore::gitignore::GitignoreBuilder::new(local);
        root.add(local.join(".git/info/exclude"));
        for f in files {
            let dir = f.parent().unwrap_or(local);
            if dir == local {
                root.add(f);
            } else {
                let mut b = ignore::gitignore::GitignoreBuilder::new(dir);
                b.add(f);
                if let Ok(g) = b.build() {
                    rules.push(g);
                }
            }
        }
        if let Ok(g) = root.build() {
            rules.insert(0, g);
        }
        Ignore {
            local: local.to_path_buf(),
            rules,
        }
    }

    /// Whether the path `rel`, relative to the project, is ignored
    fn ignored(&self, rel: &str, dir: bool) -> bool {
        let path = self.local.join(rel);
        self.rules.iter().any(|g| {
            path.starts_with(g.path()) && g.matched_path_or_any_parents(&path, dir).is_ignore()
        })
    }
}

/// What both sides agreed a file was, at the last copy
#[derive(Clone, Debug, PartialEq)]
struct Base {
    link: bool,
    crc: u32,
    /// its size and time of modification here, in nanoseconds
    size: u64,
    mtime: i128,
}

type Manifest = BTreeMap<String, Base>;

fn load_manifest(path: &Path) -> Option<Manifest> {
    let text = std::fs::read_to_string(path).ok()?;
    let mut m = Manifest::new();
    for line in text.lines() {
        let mut it = line.splitn(5, ' ');
        let (Some(kind), Some(crc), Some(size), Some(mtime), Some(path)) =
            (it.next(), it.next(), it.next(), it.next(), it.next())
        else {
            continue;
        };
        let (Ok(crc), Ok(size), Ok(mtime)) = (crc.parse(), size.parse(), mtime.parse()) else {
            continue;
        };
        m.insert(
            path.to_string(),
            Base {
                link: kind == "l",
                crc,
                size,
                mtime,
            },
        );
    }
    Some(m)
}

fn save_manifest(path: &Path, m: &Manifest) -> Result<(), String> {
    let mut s = String::new();
    for (p, b) in m {
        let kind = if b.link { 'l' } else { 'f' };
        s.push_str(&format!("{kind} {} {} {} {p}\n", b.crc, b.size, b.mtime));
    }
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, s)
        .and_then(|()| std::fs::rename(&tmp, path))
        .map_err(|e| format!("{}: {e}", path.display()))
}

/// A file of the project here
struct LocalFile {
    rel: String,
    abs: PathBuf,
    link: bool,
    size: u64,
    mtime: i128,
    mode: u32,
}

fn stat(abs: &Path) -> Option<(bool, u64, i128, u32)> {
    let m = std::fs::symlink_metadata(abs).ok()?;
    let mtime = m
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as i128);
    #[cfg(unix)]
    let mode = std::os::unix::fs::PermissionsExt::mode(&m.permissions());
    #[cfg(not(unix))]
    let mode = 0o644;
    let link = m.file_type().is_symlink();
    if !link && !m.is_file() {
        return None;
    }
    Some((link, m.len(), mtime, mode))
}

/// The files of the project to copy: those its .gitignore files don't
/// ignore (git repository or not), and the repository itself
fn scan(local: &Path) -> (Vec<LocalFile>, Ignore) {
    let mut files = Vec::new();
    let mut gitignores = Vec::new();
    let tree = ignore::WalkBuilder::new(local)
        .hidden(false)
        .require_git(false)
        .filter_entry(|e| e.file_name() != ".git")
        .build();
    let repo = ignore::WalkBuilder::new(local.join(".git"))
        .standard_filters(false)
        .build();
    for entry in tree.chain(repo).filter_map(Result::ok) {
        let abs = entry.path();
        let Ok(rel) = abs.strip_prefix(local) else {
            continue;
        };
        let Some(rel) = rel.to_str().filter(|r| !r.is_empty() && !r.contains('\n')) else {
            continue;
        };
        let Some((link, size, mtime, mode)) = stat(abs) else {
            continue;
        };
        if entry.file_name() == ".gitignore" && !rel.starts_with(".git/") {
            gitignores.push(abs.to_path_buf());
        }
        files.push(LocalFile {
            rel: rel.to_string(),
            abs: abs.to_path_buf(),
            link,
            size,
            mtime,
            mode,
        });
    }
    (files, Ignore::new(local, &gitignores))
}

/// The CRC of a file here, or of a link's target
fn local_crc(abs: &Path, link: bool) -> io::Result<u32> {
    if link {
        let target = std::fs::read_link(abs)?;
        return Ok(tar::cksum(target.to_string_lossy().as_bytes()));
    }
    Ok(tar::cksum(&std::fs::read(abs)?))
}

/// A file there that isn't what both sides agreed on
#[derive(Debug, Clone, Copy, PartialEq)]
enum Remote {
    /// its content now
    Changed(u32),
    Gone,
}

/// How a file changed there
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Added,
    Modified,
    Deleted,
}

impl Kind {
    pub fn verb(self) -> &'static str {
        match self {
            Kind::Added => "Added",
            Kind::Modified => "Changed",
            Kind::Deleted => "Deleted",
        }
    }
}

/// A file changed there since the last copy
#[derive(Debug, Clone)]
pub struct Change {
    /// relative to the project
    pub path: String,
    pub kind: Kind,
    /// changed here too: pulling leaves it as it is
    pub conflict: bool,
}

/// What a copy did
#[derive(Debug, Default)]
pub struct Report {
    pub files: usize,
    pub bytes: u64,
    pub deleted: usize,
    /// changed on both sides: left as they are
    pub conflicts: Vec<String>,
}

impl Report {
    /// e.g. `copied 12 files (1.2 MB) to vemu:dev`
    pub fn describe(&self, done: &str, to: &str, label: &str) -> String {
        let mut s = match self.files {
            0 => format!("nothing to copy {to} {label}"),
            1 => format!("{done} 1 file ({}) {to} {label}", size(self.bytes)),
            n => format!("{done} {n} files ({}) {to} {label}", size(self.bytes)),
        };
        if self.deleted > 0 {
            s.push_str(&format!(", {} deleted", self.deleted));
        }
        if !self.conflicts.is_empty() {
            let n = self.conflicts.len();
            let mut names: Vec<_> = self.conflicts.iter().take(8).map(String::as_str).collect();
            if n > 8 {
                names.push("…");
            }
            s.push_str(&format!(
                "; {n} changed on both sides, left as they are: {}",
                names.join(", ")
            ));
        }
        s
    }
}

pub fn size(bytes: u64) -> String {
    match bytes {
        b if b < 1 << 10 => format!("{b} B"),
        b if b < 1 << 20 => format!("{:.1} KB", b as f64 / 1024.0),
        b if b < 1 << 30 => format!("{:.1} MB", b as f64 / (1 << 20) as f64),
        b => format!("{:.1} GB", b as f64 / (1 << 30) as f64),
    }
}

/// A file changed there, compared to the file here
pub struct FileDiff {
    pub change: Change,
    /// from the file here to the file there; None for a binary file
    pub lines: Option<Vec<DiffLine>>,
}

impl Sandbox {
    fn lock_sync(&self) -> std::sync::MutexGuard<'_, ()> {
        self.sync.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copy there what changed here since the last copy (everything the
    /// first time): files changed there too are left as they are
    pub fn push(&self, say: &mut dyn FnMut(&str)) -> Result<Report, String> {
        let _sync = self.lock_sync();
        let root = Self::arg(&self.root);
        let check = self.run(r#"[ -e "$1" ] && [ -d "$2" ]"#, &[&self.stamp, &root], None)?;
        let old = match load_manifest(&self.manifest) {
            Some(m) if check.code == Some(0) => Some(m),
            _ => None,
        };
        let first = old.is_none();
        let old = old.unwrap_or_default();
        let (files, ignore) = scan(&self.local);
        *self.ignore.lock().unwrap_or_else(|e| e.into_inner()) = ignore;
        let remote = if first {
            HashMap::new()
        } else {
            self.remote_changes(&old)?
        };
        let mut report = Report::default();
        let mut new = Manifest::new();
        let mut send = Vec::new();
        for f in &files {
            let known = old.get(&f.rel);
            if let Some(b) = known
                && b.link == f.link
                && b.size == f.size
                && b.mtime == f.mtime
            {
                new.insert(f.rel.clone(), b.clone());
                continue;
            }
            let Ok(crc) = local_crc(&f.abs, f.link) else {
                continue;
            };
            let base = Base {
                link: f.link,
                crc,
                size: f.size,
                mtime: f.mtime,
            };
            match (known, remote.get(&f.rel)) {
                // touched, not changed
                (Some(b), _) if b.crc == crc => {
                    new.insert(f.rel.clone(), base);
                }
                // the same change on both sides
                (_, Some(Remote::Changed(c))) if *c == crc => {
                    new.insert(f.rel.clone(), base);
                }
                (_, Some(_)) => {
                    report.conflicts.push(f.rel.clone());
                    if let Some(b) = known {
                        new.insert(f.rel.clone(), b.clone());
                    }
                }
                _ => {
                    send.push(f);
                    new.insert(f.rel.clone(), base);
                }
            }
        }
        // deleted here since
        let here: HashSet<&str> = files.iter().map(|f| f.rel.as_str()).collect();
        let mut delete = Vec::new();
        for (rel, b) in &old {
            if here.contains(rel.as_str()) {
                continue;
            }
            if remote.contains_key(rel) {
                report.conflicts.push(rel.clone());
                new.insert(rel.clone(), b.clone());
            } else if !self.local.join(rel).exists() {
                delete.push(rel.clone());
            }
        }
        if first {
            // older than anything changed there from now on: busybox's
            // `find -newer` compares seconds
            let out = self.run(
                r#"mkdir -p -- "$2" && : > "$1" && sleep 1"#,
                &[&self.stamp, &root],
                None,
            )?;
            if out.code != Some(0) {
                return Err(format!("{}: {}", self.stamp, clean(&out.stderr)));
            }
        }
        if !send.is_empty() {
            let bytes: u64 = send.iter().map(|f| f.size).sum();
            say(&format!(
                "copying {} file{} ({}) to {}…",
                send.len(),
                if send.len() == 1 { "" } else { "s" },
                size(bytes),
                self.label
            ));
            let entries: Vec<_> = send
                .iter()
                .map(|f| (f.rel.clone(), f.abs.clone(), f.link, f.mode, f.mtime))
                .collect();
            let sent = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
            let counter = sent.clone();
            let feed: Feed = Box::new(move |w| {
                let mut tw = tar::Writer::new(io::BufWriter::with_capacity(1 << 20, w));
                for (rel, abs, link, mode, mtime) in entries {
                    let secs = (mtime / 1_000_000_000).max(0) as u64;
                    if link {
                        if let Ok(t) = std::fs::read_link(&abs) {
                            let _ = tw.symlink(&rel, &t.to_string_lossy(), secs);
                        }
                    } else if let Ok(data) = std::fs::read(&abs) {
                        tw.file(&rel, mode, secs, &data)?;
                    }
                }
                counter.store(tw.bytes, std::sync::atomic::Ordering::Relaxed);
                tw.finish()?;
                Ok(())
            });
            let out = self.run_with(r#"cd -- "$1" && exec tar -xf -"#, &[&root], Some(feed))?;
            if out.code != Some(0) {
                return Err(format!("tar there: {}", clean(&out.stderr)));
            }
            report.files = send.len();
            report.bytes = sent.load(std::sync::atomic::Ordering::Relaxed);
        }
        if !delete.is_empty() {
            let list: String = delete.iter().map(|d| format!("{d}\n")).collect();
            let script = r#"cd -- "$1" || exit 1; while IFS= read -r f; do rm -f -- "./$f"; done"#;
            self.run(script, &[&root], Some(list.as_bytes()))?;
            report.deleted = delete.len();
        }
        save_manifest(&self.manifest, &new)?;
        Ok(report)
    }

    /// The files there that aren't what both sides agreed on, `.git`'s included
    fn remote_changes(&self, old: &Manifest) -> Result<HashMap<String, Remote>, String> {
        // the large directories ignored, like target/, aren't looked into
        let top = self.list(&self.root).unwrap_or_default();
        let prune: String = {
            let ignore = self.ignore.lock().unwrap_or_else(|e| e.into_inner());
            top.iter()
                .filter_map(|e| e.strip_suffix('/'))
                .filter(|d| *d != ".git" && ignore.ignored(d, true))
                .map(|d| format!("-path {} -prune -o ", sq(&format!("./{d}"))))
                .collect()
        };
        let script = format!(
            r#"cd -- "$1" || exit 1
find . {prune}\( -type f -o -type l \) -print | sed 's/^/L /'
find . {prune}-type f -newer "$2" -print | tr '\n' '\0' | xargs -0 sh -c '[ $# -gt 0 ] && exec cksum "$@"' sh | sed 's/^/C /'"#
        );
        let out = self.run(&script, &[&Self::arg(&self.root), &self.stamp], None)?;
        if out.code != Some(0) {
            return Err(format!("{}: {}", self.label, clean(&out.stderr)));
        }
        let text = String::from_utf8_lossy(&out.stdout);
        let ignore = self.ignore.lock().unwrap_or_else(|e| e.into_inner());
        let mut there = HashSet::new();
        let mut changed = HashMap::new();
        for line in text.lines() {
            if let Some(p) = line.strip_prefix("L ./") {
                there.insert(p.to_string());
            } else if let Some(rest) = line.strip_prefix("C ") {
                let mut it = rest.splitn(3, ' ');
                let (Some(crc), Some(_), Some(p)) = (it.next(), it.next(), it.next()) else {
                    continue;
                };
                let (Ok(crc), Some(p)) = (crc.parse::<u32>(), p.strip_prefix("./")) else {
                    continue;
                };
                match old.get(p) {
                    Some(b) if b.crc == crc => {}
                    // a file the project's rules ignore, made there: a build's
                    None if !p.starts_with(".git/") && ignore.ignored(p, false) => {}
                    _ => {
                        changed.insert(p.to_string(), Remote::Changed(crc));
                    }
                }
            }
        }
        for rel in old.keys() {
            if !there.contains(rel) {
                changed.insert(rel.clone(), Remote::Gone);
            }
        }
        Ok(changed)
    }

    /// The files of the working tree changed there since the last copy
    pub fn changes(&self) -> Result<Vec<Change>, String> {
        let _sync = self.lock_sync();
        Ok(self.changes_unlocked()?.into_iter().map(|(c, _)| c).collect())
    }

    fn changes_unlocked(&self) -> Result<Vec<(Change, Remote)>, String> {
        let old = load_manifest(&self.manifest).ok_or("nothing was copied there yet")?;
        let remote = self.remote_changes(&old)?;
        let mut out = Vec::new();
        for (rel, r) in remote {
            if rel.starts_with(".git/") {
                continue;
            }
            let base = old.get(&rel);
            let kind = match (base, r) {
                (_, Remote::Gone) => Kind::Deleted,
                (None, _) => Kind::Added,
                (Some(_), _) => Kind::Modified,
            };
            let abs = self.local.join(&rel);
            let conflict = match (base, stat(&abs)) {
                (None, None) => false,
                // made here too: the same file, or not
                (None, Some((link, ..))) => match r {
                    Remote::Changed(c) => local_crc(&abs, link).ok() != Some(c),
                    Remote::Gone => true,
                },
                (Some(_), None) => kind != Kind::Deleted,
                (Some(b), Some((link, size, mtime, _))) => {
                    !(b.size == size && b.mtime == mtime)
                        && local_crc(&abs, link).ok() != Some(b.crc)
                }
            };
            // made on both sides alike: nothing to do
            if base.is_none() && !conflict && abs.exists() {
                continue;
            }
            out.push((
                Change {
                    path: rel,
                    kind,
                    conflict,
                },
                r,
            ));
        }
        out.sort_by(|a, b| a.0.path.cmp(&b.0.path));
        Ok(out)
    }

    /// The changes there, as diffs from the files here
    pub fn diff(&self) -> Result<Vec<FileDiff>, String> {
        let _sync = self.lock_sync();
        let changes = self.changes_unlocked()?;
        let mut there = self.fetch_changed(&changes)?;
        Ok(changes
            .into_iter()
            .map(|(c, _)| {
                let here = std::fs::read(self.local.join(&c.path)).unwrap_or_default();
                let new = there.remove(&c.path).map(|(_, d)| d).unwrap_or_default();
                let binary = |b: &[u8]| b.iter().take(8192).any(|&x| x == 0);
                let lines = (!binary(&here) && !binary(&new)).then(|| {
                    crate::tools::diff(&String::from_utf8_lossy(&here), &String::from_utf8_lossy(&new))
                });
                FileDiff { change: c, lines }
            })
            .collect())
    }

    /// The content there of the files added or modified among `changes`,
    /// with their mode
    fn fetch_changed(
        &self,
        changes: &[(Change, Remote)],
    ) -> Result<HashMap<String, (u32, Vec<u8>)>, String> {
        let paths: Vec<PathBuf> = changes
            .iter()
            .filter(|(c, _)| c.kind != Kind::Deleted)
            .map(|(c, _)| self.root.join(&c.path))
            .collect();
        let mut got = HashMap::new();
        self.fetch(&paths, MAX_FETCH, &mut |p, e| {
            if let Ok(rel) = p.strip_prefix(&self.root) {
                got.insert(rel.to_string_lossy().into_owned(), (e.mode, e.data));
            }
            true
        })?;
        Ok(got)
    }

    /// Bring here the changes made there, but those to files changed here too
    pub fn pull(&self) -> Result<Report, String> {
        let _sync = self.lock_sync();
        let changes = self.changes_unlocked()?;
        let mut there = self.fetch_changed(&changes)?;
        let mut manifest = load_manifest(&self.manifest).unwrap_or_default();
        let mut report = Report::default();
        for (c, _) in &changes {
            if c.conflict {
                report.conflicts.push(c.path.clone());
                continue;
            }
            let abs = self.local.join(&c.path);
            if c.kind == Kind::Deleted {
                match std::fs::remove_file(&abs) {
                    Ok(()) => report.deleted += 1,
                    Err(e) if e.kind() == io::ErrorKind::NotFound => {}
                    Err(e) => return Err(format!("{}: {e}", c.path)),
                }
                manifest.remove(&c.path);
                continue;
            }
            let Some((mode, data)) = there.remove(&c.path) else {
                // too large to copy, or gone since
                report.conflicts.push(format!("{} (not copied)", c.path));
                continue;
            };
            if let Some(dir) = abs.parent() {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", c.path))?;
            }
            std::fs::write(&abs, &data).map_err(|e| format!("{}: {e}", c.path))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                let _ = std::fs::set_permissions(&abs, std::fs::Permissions::from_mode(mode & 0o777));
            }
            #[cfg(not(unix))]
            let _ = mode;
            report.files += 1;
            report.bytes += data.len() as u64;
            if let Some((link, size, mtime, _)) = stat(&abs) {
                manifest.insert(
                    c.path.clone(),
                    Base {
                        link,
                        crc: tar::cksum(&data),
                        size,
                        mtime,
                    },
                );
            }
        }
        save_manifest(&self.manifest, &manifest)?;
        Ok(report)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs() {
        assert_eq!(
            Spec::parse("ssh:me@devbox:2222,dir=/srv/w").unwrap(),
            Spec::Ssh {
                dest: "me@devbox".into(),
                port: Some(2222),
                dir: Some("/srv/w".into())
            }
        );
        assert_eq!(
            Spec::parse("ssh:devbox").unwrap(),
            Spec::Ssh {
                dest: "devbox".into(),
                port: None,
                dir: None
            }
        );
        let Spec::Vemu(v) =
            Spec::parse("vemu:dev,image=/i/a.raw,kernel=/k/Image,cpus=4,memory=4G,append=root=/dev/vda rw,quiet")
                .unwrap()
        else {
            panic!()
        };
        assert_eq!(v.name, "dev");
        assert_eq!(v.image, Some("/i/a.raw".into()));
        assert_eq!(v.kernel, Some("/k/Image".into()));
        assert_eq!(v.cpus, Some(4));
        assert_eq!(v.memory.as_deref(), Some("4G"));
        assert_eq!(v.append.as_deref(), Some("root=/dev/vda rw,quiet"));
        assert!(Spec::parse("vemu:../x").is_err());
        assert!(Spec::parse("vemu:dev,bogus=1").is_err());
        assert!(Spec::parse("docker:x").is_err());
        assert!(Spec::parse("ssh:").is_err());
    }

    #[test]
    fn quoting() {
        assert_eq!(sq("it's"), r"'it'\''s'");
        let out = Command::new("sh")
            .arg("-c")
            .arg(format!("printf %s {}", sq("a 'b' $c \\d")))
            .output()
            .unwrap();
        assert_eq!(String::from_utf8_lossy(&out.stdout), "a 'b' $c \\d");
    }

    #[test]
    fn marks_taken_out() {
        let mut d = Demux::new("vcodeabc");
        let mut marks = Vec::new();
        let mut out = d.feed(b"\x1evcodeabc pid 42\x1e\nwarning: x\n\x1eother", &mut marks);
        // a mark split between reads
        out.extend(d.feed(b"\nmore\n\x1evcodeab", &mut marks));
        out.extend(d.feed(b"c exit 3\x1e\nlate\n", &mut marks));
        out.extend(d.finish());
        assert_eq!(marks, vec![Mark::Pid(42), Mark::Exit(3)]);
        assert_eq!(out, b"warning: x\n\x1eother\nmore\nlate\n");
    }

    #[test]
    fn manifest_round_trip() {
        let dir = std::env::temp_dir().join(format!("vcode-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("m");
        let mut m = Manifest::new();
        m.insert(
            "src/a file.rs".into(),
            Base {
                link: false,
                crc: 123,
                size: 4,
                mtime: 1_700_000_000_123_456_789,
            },
        );
        m.insert(
            "link".into(),
            Base {
                link: true,
                crc: 9,
                size: 0,
                mtime: 0,
            },
        );
        save_manifest(&path, &m).unwrap();
        assert_eq!(load_manifest(&path).unwrap(), m);
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn scan_skips_ignored() {
        let dir = std::env::temp_dir().join(format!("vcode-scan-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("target/debug")).unwrap();
        std::fs::create_dir_all(dir.join("src/gen")).unwrap();
        std::fs::create_dir_all(dir.join(".git/refs")).unwrap();
        std::fs::write(dir.join(".gitignore"), "target/\n*.log\n").unwrap();
        std::fs::write(dir.join("src/.gitignore"), "gen/\n").unwrap();
        std::fs::write(dir.join("src/a.rs"), "x").unwrap();
        std::fs::write(dir.join("src/gen/out.rs"), "x").unwrap();
        std::fs::write(dir.join("build.log"), "x").unwrap();
        std::fs::write(dir.join("target/debug/bin"), "x").unwrap();
        std::fs::write(dir.join(".git/HEAD"), "ref").unwrap();
        let (files, ignore) = scan(&dir);
        let mut rels: Vec<_> = files.iter().map(|f| f.rel.as_str()).collect();
        rels.sort();
        assert_eq!(rels, [".git/HEAD", ".gitignore", "src/.gitignore", "src/a.rs"]);
        assert!(ignore.ignored("target", true));
        assert!(ignore.ignored("target/release/x", false));
        assert!(ignore.ignored("x.log", false));
        assert!(ignore.ignored("src/gen/y.rs", false));
        assert!(!ignore.ignored("src/b.rs", false));
        let _ = std::fs::remove_dir_all(dir);
    }
}

//! Desktop notifications: through the terminal when it shows them (its
//! notification escape, which also works over ssh), else with the system's
//! command (osascript, notify-send).

use std::io::Write;
use std::process::{Command, Stdio};

/// How a notification is shown
#[derive(Debug, Clone, Copy, PartialEq)]
enum Via {
    /// OSC 9: iTerm2, WezTerm, Ghostty
    Osc9,
    /// OSC 99: kitty
    Osc99,
    /// OSC 777: foot, rxvt, also WezTerm and Ghostty
    Osc777,
    /// osascript on macOS, notify-send elsewhere
    Command,
}

/// Show `body` under `title`; nothing is said of a failure, it's a nicety
pub fn send(title: &str, body: &str) {
    let env = |k: &str| std::env::var(k).ok().filter(|v| !v.is_empty());
    let (title, body) = (clean(title), clean(body));
    let tmux = env("TMUX").is_some();
    let via = via(&env);
    if via == Via::Command && command(&title, &body).is_ok() {
        return;
    }
    let mut out = std::io::stdout();
    let _ = out
        .write_all(escape(via, &title, &body, tmux).as_bytes())
        .and_then(|_| out.flush());
}

/// The terminal's way if it has one, else the system's when the session is
/// local; OSC 9, the most common, for a remote terminal not known
fn via(env: &dyn Fn(&str) -> Option<String>) -> Via {
    let is = |k: &str, v: &str| env(k).is_some_and(|x| x == v);
    let remote = env("SSH_TTY").is_some() || env("SSH_CONNECTION").is_some();
    // in tmux locally, the variables are those of the terminal the server
    // started in, which may not be this one
    if env("TMUX").is_some() && !remote {
        return Via::Command;
    }
    let term = env("TERM").unwrap_or_default();
    if term == "xterm-kitty" || env("KITTY_WINDOW_ID").is_some() {
        Via::Osc99
    } else if term.starts_with("foot") || term.starts_with("rxvt") {
        Via::Osc777
    } else if term == "xterm-ghostty"
        || is("TERM_PROGRAM", "iTerm.app")
        || is("TERM_PROGRAM", "WezTerm")
        || is("TERM_PROGRAM", "ghostty")
        || is("LC_TERMINAL", "iTerm2")
        || env("WEZTERM_PANE").is_some()
        || remote
    {
        Via::Osc9
    } else {
        Via::Command
    }
}

/// The escape showing it; inside tmux, passed through to the terminal
/// (which needs `set -g allow-passthrough on`)
fn escape(via: Via, title: &str, body: &str, tmux: bool) -> String {
    let seq = match via {
        Via::Osc9 | Via::Command => format!("\x1b]9;{title}: {body}\x07"),
        Via::Osc99 => format!(
            "\x1b]99;i=vcode:d=0;{title}\x1b\\\x1b]99;i=vcode:d=1:p=body;{body}\x1b\\"
        ),
        Via::Osc777 => format!("\x1b]777;notify;{};{body}\x07", title.replace(';', ",")),
    };
    if tmux {
        format!("\x1bPtmux;{}\x1b\\", seq.replace('\x1b', "\x1b\x1b"))
    } else {
        seq
    }
}

/// The system's notification; the text goes as arguments, never as code
fn command(title: &str, body: &str) -> std::io::Result<()> {
    let mut cmd = if cfg!(target_os = "macos") {
        let mut c = Command::new("osascript");
        c.args([
            "-e",
            "on run argv",
            "-e",
            "display notification (item 2 of argv) with title (item 1 of argv)",
            "-e",
            "end run",
            title,
            body,
        ]);
        c
    } else {
        let mut c = Command::new("notify-send");
        c.args(["--app-name=vcode", "--", title, body]);
        c
    };
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    // not waited for here: osascript takes a while
    std::thread::spawn(move || child.wait());
    Ok(())
}

/// One line of text, without the control characters that would end an
/// escape early or start another
fn clean(s: &str) -> String {
    let s: String = s
        .chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect();
    let s = s.split_whitespace().collect::<Vec<_>>().join(" ");
    match s.char_indices().nth(160) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chosen_by_terminal() {
        let with = |vars: &'static [(&'static str, &'static str)]| {
            via(&move |k: &str| {
                vars.iter()
                    .find(|(n, _)| *n == k)
                    .map(|(_, v)| v.to_string())
            })
        };
        assert_eq!(with(&[("TERM_PROGRAM", "iTerm.app")]), Via::Osc9);
        assert_eq!(with(&[("TERM", "xterm-kitty")]), Via::Osc99);
        assert_eq!(with(&[("TERM", "foot")]), Via::Osc777);
        assert_eq!(with(&[("TERM_PROGRAM", "Apple_Terminal")]), Via::Command);
        assert_eq!(with(&[("SSH_TTY", "/dev/ttys001")]), Via::Osc9);
        // in tmux: the system's locally, the terminal's over ssh
        let local = &[("TMUX", "/tmp/tmux"), ("TERM_PROGRAM", "iTerm.app")];
        assert_eq!(with(local), Via::Command);
        let remote = &[("TMUX", "/tmp/tmux"), ("SSH_TTY", "x"), ("TERM", "xterm-kitty")];
        assert_eq!(with(remote), Via::Osc99);
    }

    #[test]
    fn escapes() {
        assert_eq!(escape(Via::Osc9, "vcode", "done", false), "\x1b]9;vcode: done\x07");
        assert_eq!(
            escape(Via::Osc777, "a;b", "done", false),
            "\x1b]777;notify;a,b;done\x07"
        );
        assert_eq!(
            escape(Via::Osc9, "vcode", "done", true),
            "\x1bPtmux;\x1b\x1b]9;vcode: done\x07\x1b\\"
        );
        // the model's text can't end the escape, nor start one
        assert_eq!(clean("ok\x07\x1b]0;pwned\x1b\\\n  next"), "ok ]0;pwned \\ next");
        assert_eq!(clean(&"x".repeat(200)).chars().count(), 161);
    }
}

//! Copying to the system clipboard: with the platform's command when the
//! session is local (pbcopy, wl-copy, xclip, xsel), else with the terminal's
//! OSC 52 escape, which also works over ssh in the terminals that support it.

use std::io::Write;
use std::process::{Command, Stdio};

/// Copy `text`: how it was copied
pub fn copy(text: &str) -> Result<&'static str, String> {
    let remote = std::env::var_os("SSH_TTY").is_some() || std::env::var_os("SSH_CONNECTION").is_some();
    if !remote {
        for (cmd, args) in COMMANDS {
            if run(cmd, args, text).is_ok() {
                return Ok(cmd);
            }
        }
    }
    osc52(text).map_err(|e| e.to_string())?;
    Ok("the terminal (OSC 52)")
}

const COMMANDS: &[(&str, &[&str])] = &[
    #[cfg(target_os = "macos")]
    ("pbcopy", &[]),
    #[cfg(not(target_os = "macos"))]
    ("wl-copy", &[]),
    #[cfg(not(target_os = "macos"))]
    ("xclip", &["-selection", "clipboard"]),
    #[cfg(not(target_os = "macos"))]
    ("xsel", &["--clipboard", "--input"]),
];

fn run(cmd: &str, args: &[&str], text: &str) -> std::io::Result<()> {
    let mut child = Command::new(cmd)
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    child
        .stdin
        .take()
        .expect("stdin is piped")
        .write_all(text.as_bytes())?;
    let status = child.wait()?;
    if status.success() {
        Ok(())
    } else {
        Err(std::io::Error::other(format!("{cmd}: {status}")))
    }
}

fn osc52(text: &str) -> std::io::Result<()> {
    let mut out = std::io::stdout();
    write!(out, "\x1b]52;c;{}\x07", base64(text.as_bytes()))?;
    out.flush()
}

fn base64(bytes: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        for i in 0..4 {
            if i <= c.len() {
                s.push(ABC[(n >> (18 - 6 * i) & 63) as usize] as char);
            } else {
                s.push('=');
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    #[test]
    fn base64() {
        for (i, o) in [
            ("", ""),
            ("f", "Zg=="),
            ("fo", "Zm8="),
            ("foo", "Zm9v"),
            ("foob", "Zm9vYg=="),
            ("hello, wörld", "aGVsbG8sIHfDtnJsZA=="),
        ] {
            assert_eq!(super::base64(i.as_bytes()), o, "{i}");
        }
    }
}

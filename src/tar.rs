//! Just enough tar to move files to and from a sandbox: archives written
//! here are extracted there by its `tar`, and those its `tar` writes are
//! read here (ustar, GNU long names, pax paths).
//!
//! And POSIX `cksum`'s CRC, to tell which files changed on the other side
//! without copying them.

use std::io::{self, Read, Write};

const BLOCK: usize = 512;

/// Writes an archive to `out`
pub struct Writer<W: Write> {
    out: W,
    /// bytes of file content written
    pub bytes: u64,
}

impl<W: Write> Writer<W> {
    pub fn new(out: W) -> Self {
        Writer { out, bytes: 0 }
    }

    /// A regular file, `path` relative
    pub fn file(&mut self, path: &str, mode: u32, mtime: u64, data: &[u8]) -> io::Result<()> {
        self.entry(path, b'0', mode, mtime, data.len() as u64, "")?;
        self.out.write_all(data)?;
        self.pad(data.len())?;
        self.bytes += data.len() as u64;
        Ok(())
    }

    /// A symbolic link to `target`
    pub fn symlink(&mut self, path: &str, target: &str, mtime: u64) -> io::Result<()> {
        if target.len() > 100 {
            return Err(io::Error::other(format!("{path}: link target too long")));
        }
        self.entry(path, b'2', 0o777, mtime, 0, target)
    }

    /// The end of the archive, and its writer back
    pub fn finish(mut self) -> io::Result<W> {
        self.out.write_all(&[0; BLOCK * 2])?;
        self.out.flush()?;
        Ok(self.out)
    }

    fn entry(
        &mut self,
        path: &str,
        kind: u8,
        mode: u32,
        mtime: u64,
        size: u64,
        link: &str,
    ) -> io::Result<()> {
        let (prefix, name) = match split(path) {
            Some(s) => s,
            None => {
                // a GNU long name: an entry of its own holding the path
                let data = [path.as_bytes(), &[0]].concat();
                let mut h = header("././@LongLink", "", b'L', 0, 0, data.len() as u64, "");
                h[257..265].copy_from_slice(b"ustar  \0");
                checksum(&mut h);
                self.out.write_all(&h)?;
                self.out.write_all(&data)?;
                self.pad(data.len())?;
                ("", truncate(path, 100))
            }
        };
        let h = header(name, prefix, kind, mode, mtime, size, link);
        self.out.write_all(&h)
    }

    fn pad(&mut self, n: usize) -> io::Result<()> {
        let rest = (BLOCK - n % BLOCK) % BLOCK;
        self.out.write_all(&[0; BLOCK][..rest])
    }
}

/// `path` as ustar's prefix and name, if it fits
fn split(path: &str) -> Option<(&str, &str)> {
    if path.len() <= 100 {
        return Some(("", path));
    }
    // the prefix takes 155 bytes at most, up to a slash
    path.match_indices('/')
        .map(|(i, _)| i)
        .find(|&i| i <= 155 && path.len() - i - 1 <= 100 && i > 0)
        .map(|i| (&path[..i], &path[i + 1..]))
}

fn truncate(s: &str, n: usize) -> &str {
    let mut end = s.len().min(n);
    while !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn header(
    name: &str,
    prefix: &str,
    kind: u8,
    mode: u32,
    mtime: u64,
    size: u64,
    link: &str,
) -> [u8; BLOCK] {
    let mut h = [0u8; BLOCK];
    let put = |h: &mut [u8; BLOCK], at: usize, s: &[u8]| h[at..at + s.len()].copy_from_slice(s);
    put(&mut h, 0, truncate(name, 100).as_bytes());
    put(&mut h, 100, format!("{:07o}\0", mode & 0o7777).as_bytes());
    put(&mut h, 108, b"0000000\0");
    put(&mut h, 116, b"0000000\0");
    put(&mut h, 124, format!("{size:011o}\0").as_bytes());
    put(&mut h, 136, format!("{:011o}\0", mtime.min(0o77777777777)).as_bytes());
    h[156] = kind;
    put(&mut h, 157, truncate(link, 100).as_bytes());
    put(&mut h, 257, b"ustar\0");
    put(&mut h, 263, b"00");
    put(&mut h, 345, truncate(prefix, 155).as_bytes());
    checksum(&mut h);
    h
}

fn checksum(h: &mut [u8; BLOCK]) {
    h[148..156].copy_from_slice(b"        ");
    let sum: u32 = h.iter().map(|&b| b as u32).sum();
    h[148..156].copy_from_slice(format!("{sum:06o}\0 ").as_bytes());
}

/// An entry of an archive read
pub struct Entry {
    pub path: String,
    pub mode: u32,
    /// regular file
    pub file: bool,
    pub data: Vec<u8>,
}

/// Each entry of the archive `r` given to `f`, until it returns false. A
/// truncated archive ends quietly: what came is given.
pub fn read(mut r: impl Read, f: &mut dyn FnMut(Entry) -> bool) -> io::Result<()> {
    let mut long: Option<String> = None;
    loop {
        let mut h = [0u8; BLOCK];
        if !fill(&mut r, &mut h)? || h.iter().all(|&b| b == 0) {
            return Ok(());
        }
        let size = number(&h[124..136]) as usize;
        let mut data = vec![0u8; size];
        if !fill(&mut r, &mut data)? {
            return Ok(());
        }
        let mut pad = vec![0u8; (BLOCK - size % BLOCK) % BLOCK];
        fill(&mut r, &mut pad)?;
        let kind = h[156];
        match kind {
            b'L' => {
                long = Some(cstr(&data));
                continue;
            }
            b'x' => {
                long = pax_path(&data).or(long);
                continue;
            }
            b'g' => continue,
            _ => {}
        }
        let path = long.take().unwrap_or_else(|| {
            let name = cstr(&h[0..100]);
            let prefix = cstr(&h[345..500]);
            // GNU archives use the prefix's place for other things
            if &h[257..263] == b"ustar\0" && !prefix.is_empty() {
                format!("{prefix}/{name}")
            } else {
                name
            }
        });
        let entry = Entry {
            path,
            mode: number(&h[100..108]) as u32,
            file: kind == b'0' || kind == 0 || kind == b'7',
            data,
        };
        if !f(entry) {
            return Ok(());
        }
    }
}

/// Reads all of `buf`: false at the end of the stream
fn fill(r: &mut impl Read, buf: &mut [u8]) -> io::Result<bool> {
    let mut n = 0;
    while n < buf.len() {
        match r.read(&mut buf[n..]) {
            Ok(0) => return Ok(false),
            Ok(k) => n += k,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(true)
}

fn cstr(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

/// An octal field, or GNU's base-256 for large ones
fn number(b: &[u8]) -> u64 {
    if b.first().is_some_and(|&c| c & 0x80 != 0) {
        return b[1..].iter().fold(0, |n, &c| (n << 8) | c as u64);
    }
    let s = cstr(b);
    u64::from_str_radix(s.trim_matches([' ', '\0']), 8).unwrap_or(0)
}

/// The `path` record of a pax header
fn pax_path(data: &[u8]) -> Option<String> {
    let mut rest = data;
    while !rest.is_empty() {
        let space = rest.iter().position(|&c| c == b' ')?;
        let len: usize = std::str::from_utf8(&rest[..space]).ok()?.parse().ok()?;
        if len <= space || len > rest.len() {
            return None;
        }
        let record = &rest[space + 1..len - 1];
        if let Some(path) = record.strip_prefix(b"path=") {
            return Some(String::from_utf8_lossy(path).into_owned());
        }
        rest = &rest[len..];
    }
    None
}

/// The CRC POSIX `cksum` gives `data`
pub fn cksum(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let mut c = (i as u32) << 24;
            for _ in 0..8 {
                c = if c & 0x8000_0000 != 0 {
                    (c << 1) ^ 0x04C1_1DB7
                } else {
                    c << 1
                };
            }
            *e = c;
        }
        t
    });
    let mut crc = 0u32;
    let mut feed = |b: u8| crc = (crc << 8) ^ table[((crc >> 24) as u8 ^ b) as usize];
    for &b in data {
        feed(b);
    }
    // then the length, least significant byte first
    let mut n = data.len();
    while n > 0 {
        feed(n as u8);
        n >>= 8;
    }
    !crc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let long = format!("{}/{}", "d".repeat(120), "f".repeat(60));
        let longer = "x".repeat(300);
        let mut w = Writer::new(Vec::new());
        w.file("src/a.rs", 0o644, 1_700_000_000, b"fn main() {}\n").unwrap();
        w.file(&long, 0o755, 0, b"long").unwrap();
        w.file(&longer, 0o600, 0, &[7; 1000]).unwrap();
        w.symlink("link", "src/a.rs", 0).unwrap();
        let tar = w.finish().unwrap();
        assert_eq!(tar.len() % BLOCK, 0);
        let mut got = Vec::new();
        read(&tar[..], &mut |e| {
            got.push((e.path, e.mode, e.file, e.data));
            true
        })
        .unwrap();
        assert_eq!(got.len(), 4);
        assert_eq!(got[0], ("src/a.rs".into(), 0o644, true, b"fn main() {}\n".to_vec()));
        assert_eq!(got[1].0, long);
        assert_eq!(got[1].1, 0o755);
        assert_eq!(got[2].0, longer);
        assert_eq!(got[2].3, vec![7; 1000]);
        assert!(!got[3].2);
    }

    #[test]
    fn system_tar_reads_ours() {
        let dir = std::env::temp_dir().join(format!("vcode-tar-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let long = format!("{}/{}", "d".repeat(120), "x".repeat(110));
        let mut w = Writer::new(Vec::new());
        w.file("a/b.txt", 0o755, 1_600_000_000, b"hello\n").unwrap();
        w.file(&long, 0o644, 0, b"deep").unwrap();
        let tar = w.finish().unwrap();
        let mut child = std::process::Command::new("tar")
            .args(["-xf", "-", "-C"])
            .arg(&dir)
            .stdin(std::process::Stdio::piped())
            .spawn()
            .unwrap();
        child.stdin.take().unwrap().write_all(&tar).unwrap();
        assert!(child.wait().unwrap().success());
        assert_eq!(std::fs::read(dir.join("a/b.txt")).unwrap(), b"hello\n");
        assert_eq!(std::fs::read(dir.join(&long)).unwrap(), b"deep");
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let m = std::fs::metadata(dir.join("a/b.txt")).unwrap();
            assert_eq!(m.permissions().mode() & 0o777, 0o755);
        }
        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn crc_is_posix_cksum() {
        assert_eq!(cksum(b""), 4294967295);
        for data in [&b"hello\n"[..], &[0u8; 300], b"The quick brown fox"] {
            let mut child = std::process::Command::new("cksum")
                .stdin(std::process::Stdio::piped())
                .stdout(std::process::Stdio::piped())
                .spawn()
                .unwrap();
            child.stdin.take().unwrap().write_all(data).unwrap();
            let out = child.wait_with_output().unwrap();
            let text = String::from_utf8(out.stdout).unwrap();
            let crc: u32 = text.split_whitespace().next().unwrap().parse().unwrap();
            assert_eq!(cksum(data), crc, "{data:?}");
        }
    }
}

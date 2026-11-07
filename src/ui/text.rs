//! Text as the transcript shows it: word wrapping of styled text, and a
//! little Markdown (code blocks, headings, lists, `code`, **bold**).

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;

/// Pieces of a line of text, each with its style
pub type Segments = Vec<(String, Style)>;

pub fn width(s: &str) -> usize {
    UnicodeWidthStr::width(s)
}

/// Lines of `segs` wrapped at `width` columns, the first starting with
/// `first`, the others with `rest`
pub fn wrap(
    segs: &[(String, Style)],
    width: usize,
    first: &[Span<'static>],
    rest: &[Span<'static>],
) -> Vec<Line<'static>> {
    let prefix_w = |p: &[Span<'static>]| p.iter().map(|s| self::width(&s.content)).sum::<usize>();
    let mut cells: Vec<(&str, Style, usize)> = Vec::new();
    for (text, style) in segs {
        for g in text.graphemes(true) {
            let g = if g == "\t" { "    " } else { g };
            cells.push((g, *style, self::width(g)));
        }
    }
    let mut lines: Vec<Vec<(&str, Style, usize)>> = Vec::new();
    let mut cur: Vec<(&str, Style, usize)> = Vec::new();
    let mut cur_w = 0;
    let mut last_space: Option<usize> = None;
    for cell in cells {
        let avail = width
            .saturating_sub(prefix_w(if lines.is_empty() { first } else { rest }))
            .max(8);
        if cur_w + cell.2 > avail && !cur.is_empty() {
            if cell.0 == " " {
                lines.push(std::mem::take(&mut cur));
                cur_w = 0;
                last_space = None;
                continue;
            }
            match last_space {
                Some(i) => {
                    let tail = cur.split_off(i + 1);
                    while cur.last().is_some_and(|c| c.0 == " ") {
                        cur.pop();
                    }
                    lines.push(std::mem::take(&mut cur));
                    cur = tail;
                }
                _ => lines.push(std::mem::take(&mut cur)),
            }
            cur_w = cur.iter().map(|c| c.2).sum();
            last_space = None;
        }
        cur_w += cell.2;
        cur.push(cell);
        if cell.0 == " " {
            last_space = Some(cur.len() - 1);
        }
    }
    lines.push(cur);
    lines
        .into_iter()
        .enumerate()
        .map(|(i, cells)| {
            let mut spans: Vec<Span<'static>> = if i == 0 {
                first.to_vec()
            } else {
                rest.to_vec()
            };
            let mut text = String::new();
            let mut style = None;
            for (g, s, _) in cells {
                if style != Some(s) {
                    if let Some(st) = style {
                        spans.push(Span::styled(std::mem::take(&mut text), st));
                    }
                    style = Some(s);
                }
                text.push_str(g);
            }
            if let Some(st) = style {
                spans.push(Span::styled(text, st));
            }
            Line::from(spans)
        })
        .collect()
}

/// Plain text wrapped, every line in `style`
pub fn wrap_plain(
    text: &str,
    style: Style,
    width: usize,
    first: &[Span<'static>],
    rest: &[Span<'static>],
) -> Vec<Line<'static>> {
    let mut out = Vec::new();
    for (i, line) in text.split('\n').enumerate() {
        let p = if i == 0 { first } else { rest };
        out.extend(wrap(&[(line.to_string(), style)], width, p, rest));
    }
    out
}

pub const CODE: Style = Style::new().fg(Color::Cyan);

/// `**bold**` and `` `code` `` within a line
fn inline(line: &str, base: Style) -> Segments {
    let mut segs: Segments = Vec::new();
    let mut text = String::new();
    let mut bold = false;
    let mut rest = line;
    let flush = |segs: &mut Segments, text: &mut String, style: Style| {
        if !text.is_empty() {
            segs.push((std::mem::take(text), style));
        }
    };
    while !rest.is_empty() {
        let style = if bold {
            base.add_modifier(Modifier::BOLD)
        } else {
            base
        };
        if let Some(after) = rest.strip_prefix('`')
            && let Some(end) = after.find('`')
        {
            flush(&mut segs, &mut text, style);
            segs.push((after[..end].to_string(), CODE));
            rest = &after[end + 1..];
        } else if let Some(after) = rest.strip_prefix("**")
            && (bold || after.contains("**"))
        {
            flush(&mut segs, &mut text, style);
            bold = !bold;
            rest = after;
        } else {
            let c = rest.chars().next().unwrap_or(' ');
            text.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    let style = if bold {
        base.add_modifier(Modifier::BOLD)
    } else {
        base
    };
    flush(&mut segs, &mut text, style);
    segs
}

/// Markdown text, wrapped at `width`, every line after `indent`
pub fn markdown(
    text: &str,
    base: Style,
    width: usize,
    first: &[Span<'static>],
    indent: &[Span<'static>],
) -> Vec<Line<'static>> {
    let dim = Style::new().fg(Color::DarkGray);
    let mut out = Vec::new();
    let mut in_code = false;
    for line in text.split('\n') {
        let first_line = out.is_empty();
        let lead: &[Span<'static>] = if first_line { first } else { indent };
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") {
            in_code = !in_code;
            let label = trimmed.trim_start_matches('`').trim();
            let mut spans = lead.to_vec();
            if in_code {
                spans.push(Span::styled(
                    if label.is_empty() {
                        "┌".to_string()
                    } else {
                        format!("┌ {label}")
                    },
                    dim,
                ));
            } else {
                spans.push(Span::styled("└".to_string(), dim));
            }
            out.push(Line::from(spans));
            continue;
        }
        if in_code {
            let mut p = lead.to_vec();
            p.push(Span::styled("│ ", dim));
            let mut r = indent.to_vec();
            r.push(Span::styled("│ ", dim));
            out.extend(wrap(
                &[(line.to_string(), Style::new().fg(Color::Gray))],
                width,
                &p,
                &r,
            ));
            continue;
        }
        if let Some(h) = heading(trimmed) {
            let style = base.add_modifier(Modifier::BOLD).fg(Color::LightBlue);
            out.extend(wrap(&inline(h, style), width, lead, indent));
            continue;
        }
        if trimmed.len() >= 3 && trimmed.chars().all(|c| c == '-' || c == '*' || c == '_') {
            let mut spans = lead.to_vec();
            spans.push(Span::styled(
                "─".repeat(width.saturating_sub(4).min(40)),
                dim,
            ));
            out.push(Line::from(spans));
            continue;
        }
        let depth = line.len() - trimmed.len();
        let bullet = ["- ", "* ", "+ "]
            .iter()
            .find_map(|b| trimmed.strip_prefix(b));
        if let Some(item) = bullet {
            let pad = " ".repeat(depth);
            let mut p = lead.to_vec();
            p.push(Span::raw(format!("{pad}• ")));
            let mut r = indent.to_vec();
            r.push(Span::raw(format!("{pad}  ")));
            out.extend(wrap(&inline(item, base), width, &p, &r));
            continue;
        }
        if let Some(q) = trimmed
            .strip_prefix("> ")
            .or(trimmed.strip_prefix('>').filter(|q| q.is_empty()))
        {
            let mut p = lead.to_vec();
            p.push(Span::styled("▎ ", dim));
            let mut r = indent.to_vec();
            r.push(Span::styled("▎ ", dim));
            out.extend(wrap(
                &inline(q, base.add_modifier(Modifier::ITALIC)),
                width,
                &p,
                &r,
            ));
            continue;
        }
        // numbered items keep their number; continuation lines align after it
        let number = trimmed.split_once(". ").filter(|(n, _)| {
            !n.is_empty() && n.len() <= 3 && n.chars().all(|c| c.is_ascii_digit())
        });
        if let Some((n, item)) = number {
            let pad = " ".repeat(depth);
            let mut p = lead.to_vec();
            p.push(Span::raw(format!("{pad}{n}. ")));
            let mut r = indent.to_vec();
            r.push(Span::raw(" ".repeat(depth + n.len() + 2)));
            out.extend(wrap(&inline(item, base), width, &p, &r));
            continue;
        }
        out.extend(wrap(&inline(line, base), width, lead, indent));
    }
    out
}

fn heading(line: &str) -> Option<&str> {
    let hashes = line.chars().take_while(|&c| c == '#').count();
    (1..=6)
        .contains(&hashes)
        .then(|| line[hashes..].strip_prefix(' '))
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(lines: &[Line<'_>]) -> Vec<String> {
        lines
            .iter()
            .map(|l| l.spans.iter().map(|s| s.content.as_ref()).collect())
            .collect()
    }

    #[test]
    fn wraps_at_words() {
        let segs = vec![("the quick brown fox jumps".to_string(), Style::new())];
        let p = [Span::raw("> ")];
        let r = [Span::raw("  ")];
        assert_eq!(
            text(&wrap(&segs, 14, &p, &r)),
            ["> the quick", "  brown fox", "  jumps"]
        );
    }

    #[test]
    fn breaks_long_words() {
        let segs = vec![("abcdefghijklmnopqrstuvwxyz".to_string(), Style::new())];
        assert_eq!(
            text(&wrap(&segs, 10, &[], &[])),
            ["abcdefghij", "klmnopqrst", "uvwxyz"]
        );
    }

    #[test]
    fn markdown_bits() {
        let md = "# Title\nsome `code` and **bold**\n- item\n```rust\nfn x() {}\n```";
        let out = text(&markdown(md, Style::new(), 80, &[], &[]));
        assert_eq!(
            out,
            [
                "Title",
                "some code and bold",
                "• item",
                "┌ rust",
                "│ fn x() {}",
                "└"
            ]
        );
    }
}

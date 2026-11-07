//! The message being typed: a multi-line editor with history.

use unicode_segmentation::UnicodeSegmentation;

use super::text::width;

#[derive(Default)]
pub struct Input {
    pub text: String,
    /// byte offset of the cursor in `text`
    cursor: usize,
    history: Vec<String>,
    /// position in `history` while browsing it, and the draft put aside
    browsing: Option<(usize, String)>,
}

impl Input {
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    pub fn set(&mut self, text: &str) {
        self.text = text.to_string();
        self.cursor = self.text.len();
    }

    /// Take the text, adding it to the history
    pub fn take(&mut self) -> String {
        let text = std::mem::take(&mut self.text);
        self.cursor = 0;
        self.browsing = None;
        if !text.trim().is_empty() && self.history.last() != Some(&text) {
            self.history.push(text.clone());
        }
        text
    }

    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.browsing = None;
    }

    pub fn insert(&mut self, s: &str) {
        let s = s.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &s);
        self.cursor += s.len();
    }

    fn prev_boundary(&self, i: usize) -> usize {
        self.text[..i]
            .grapheme_indices(true)
            .next_back()
            .map(|(j, _)| j)
            .unwrap_or(0)
    }

    fn next_boundary(&self, i: usize) -> usize {
        self.text[i..]
            .graphemes(true)
            .next()
            .map(|g| i + g.len())
            .unwrap_or(i)
    }

    pub fn backspace(&mut self) {
        if self.cursor > 0 {
            let p = self.prev_boundary(self.cursor);
            self.text.replace_range(p..self.cursor, "");
            self.cursor = p;
        }
    }

    pub fn delete(&mut self) {
        let n = self.next_boundary(self.cursor);
        self.text.replace_range(self.cursor..n, "");
    }

    pub fn left(&mut self) {
        self.cursor = self.prev_boundary(self.cursor);
    }

    pub fn right(&mut self) {
        self.cursor = self.next_boundary(self.cursor);
    }

    fn line_start(&self) -> usize {
        self.text[..self.cursor]
            .rfind('\n')
            .map(|i| i + 1)
            .unwrap_or(0)
    }

    fn line_end(&self) -> usize {
        self.text[self.cursor..]
            .find('\n')
            .map(|i| self.cursor + i)
            .unwrap_or(self.text.len())
    }

    pub fn home(&mut self) {
        self.cursor = self.line_start();
    }

    pub fn end(&mut self) {
        self.cursor = self.line_end();
    }

    fn word_start(&self) -> usize {
        let before = &self.text[..self.cursor];
        let trimmed = before.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_');
        trimmed
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map(|i| i + trimmed[i..].chars().next().map_or(1, char::len_utf8))
            .unwrap_or(0)
    }

    fn word_end(&self) -> usize {
        let after = &self.text[self.cursor..];
        let skip = after.len()
            - after
                .trim_start_matches(|c: char| !c.is_alphanumeric() && c != '_')
                .len();
        let rest = &after[skip..];
        self.cursor
            + skip
            + rest
                .find(|c: char| !c.is_alphanumeric() && c != '_')
                .unwrap_or(rest.len())
    }

    pub fn word_left(&mut self) {
        self.cursor = self.word_start();
    }

    pub fn word_right(&mut self) {
        self.cursor = self.word_end();
    }

    /// Ctrl-W
    pub fn delete_word(&mut self) {
        let s = self.word_start();
        self.text.replace_range(s..self.cursor, "");
        self.cursor = s;
    }

    /// Ctrl-U
    pub fn kill_line_before(&mut self) {
        let s = self.line_start();
        self.text.replace_range(s..self.cursor, "");
        self.cursor = s;
    }

    /// Ctrl-K
    pub fn kill_line_after(&mut self) {
        let e = self.line_end();
        if e == self.cursor && e < self.text.len() {
            self.text.remove(e);
        } else {
            self.text.replace_range(self.cursor..e, "");
        }
    }

    /// Move up a line; false on the first line
    pub fn up(&mut self) -> bool {
        let start = self.line_start();
        if start == 0 {
            return false;
        }
        let col = width(&self.text[start..self.cursor]);
        let prev_start = self.text[..start - 1]
            .rfind('\n')
            .map(|i| i + 1)
            .unwrap_or(0);
        self.cursor = column(&self.text, prev_start, start - 1, col);
        true
    }

    /// Move down a line; false on the last line
    pub fn down(&mut self) -> bool {
        let end = self.line_end();
        if end == self.text.len() {
            return false;
        }
        let col = width(&self.text[self.line_start()..self.cursor]);
        let next_end = self.text[end + 1..]
            .find('\n')
            .map(|i| end + 1 + i)
            .unwrap_or(self.text.len());
        self.cursor = column(&self.text, end + 1, next_end, col);
        true
    }

    pub fn history_prev(&mut self) {
        let i = match &self.browsing {
            Some((0, _)) => return,
            Some((i, _)) => i - 1,
            None if self.history.is_empty() => return,
            None => {
                self.browsing = Some((self.history.len(), self.text.clone()));
                self.history.len() - 1
            }
        };
        if let Some((pos, _)) = &mut self.browsing {
            *pos = i;
        }
        self.text = self.history[i].clone();
        self.cursor = self.text.len();
    }

    pub fn history_next(&mut self) {
        let Some((i, draft)) = &self.browsing else {
            return;
        };
        if i + 1 >= self.history.len() {
            self.text = draft.clone();
            self.browsing = None;
        } else {
            let i = i + 1;
            self.text = self.history[i].clone();
            if let Some((pos, _)) = &mut self.browsing {
                *pos = i;
            }
        }
        self.cursor = self.text.len();
    }

    /// The text cut in rows of `cols` columns, and the cursor's row and column
    pub fn layout(&self, cols: usize) -> (Vec<String>, (usize, usize)) {
        let cols = cols.max(4);
        let mut rows = vec![String::new()];
        let mut w = 0;
        let mut cursor = (0, 0);
        let mut offset = 0;
        for g in self.text.graphemes(true) {
            if offset == self.cursor {
                cursor = (rows.len() - 1, w);
            }
            offset += g.len();
            if g == "\n" {
                rows.push(String::new());
                w = 0;
                continue;
            }
            let gw = width(g).max(if g == "\t" { 1 } else { 0 });
            if w + gw > cols {
                rows.push(String::new());
                w = 0;
            }
            rows.last_mut()
                .unwrap()
                .push_str(if g == "\t" { " " } else { g });
            w += gw;
        }
        if offset == self.cursor {
            if w >= cols {
                rows.push(String::new());
                w = 0;
            }
            cursor = (rows.len() - 1, w);
        }
        (rows, cursor)
    }
}

/// The byte offset in `text[start..end]` at display column `col`, or its end
fn column(text: &str, start: usize, end: usize, col: usize) -> usize {
    let mut w = 0;
    for (i, g) in text[start..end].grapheme_indices(true) {
        if w >= col {
            return start + i;
        }
        w += width(g);
    }
    end
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editing() {
        let mut i = Input::default();
        i.insert("hello world");
        i.delete_word();
        assert_eq!(i.text, "hello ");
        i.insert("there\nsecond");
        assert!(i.up());
        assert_eq!(&i.text[..i.cursor], "hello ");
        assert!(!i.up());
        assert!(i.down());
        assert_eq!(i.cursor, i.text.len());
        i.home();
        i.kill_line_after();
        assert_eq!(i.text, "hello there\n");
        i.backspace();
        assert_eq!(i.text, "hello there");
    }

    #[test]
    fn history() {
        let mut i = Input::default();
        i.insert("one");
        i.take();
        i.insert("two");
        i.take();
        i.insert("draft");
        i.history_prev();
        assert_eq!(i.text, "two");
        i.history_prev();
        assert_eq!(i.text, "one");
        i.history_next();
        i.history_next();
        assert_eq!(i.text, "draft");
    }

    #[test]
    fn layout_wraps() {
        let mut i = Input::default();
        i.insert("abcdef\ngh");
        let (rows, cur) = i.layout(4);
        assert_eq!(rows, ["abcd", "ef", "gh"]);
        assert_eq!(cur, (2, 2));
    }
}

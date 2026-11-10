//! Built-in color themes.

use ratatui::style::{Color, Modifier, Style};

/// A color theme.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Theme {
    /// The original palette (magenta accent).
    Default,
    /// Nord-inspired: icy blues and purples.
    Nord,
    /// Dracula: vibrant purples and pinks.
    Dracula,
    /// Catppuccin Mocha: soft greens and lavenders.
    Catppuccin,
    /// Monokai Pro: warm oranges and pinks.
    Monokai,
    /// One Dark: balanced teal accent.
    OneDark,
    /// Gruvbox: warm earth tones.
    Gruvbox,
}

impl Default for Theme {
    fn default() -> Self {
        Theme::Default
    }
}

impl Theme {
    /// Cycle to the next theme.
    pub fn next(self) -> Self {
        match self {
            Theme::Default => Theme::Nord,
            Theme::Nord => Theme::Dracula,
            Theme::Dracula => Theme::Catppuccin,
            Theme::Catppuccin => Theme::Monokai,
            Theme::Monokai => Theme::OneDark,
            Theme::OneDark => Theme::Gruvbox,
            Theme::Gruvbox => Theme::Default,
        }
    }

    /// Resolve a theme name to a theme.
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "default" => Some(Theme::Default),
            "nord" => Some(Theme::Nord),
            "dracula" => Some(Theme::Dracula),
            "catppuccin" => Some(Theme::Catppuccin),
            "monokai" => Some(Theme::Monokai),
            "onedark" | "one-dark" => Some(Theme::OneDark),
            "gruvbox" => Some(Theme::Gruvbox),
            _ => None,
        }
    }

    /// The theme's name.
    pub fn name(self) -> &'static str {
        match self {
            Theme::Default => "default",
            Theme::Nord => "nord",
            Theme::Dracula => "dracula",
            Theme::Catppuccin => "catppuccin",
            Theme::Monokai => "monokai",
            Theme::OneDark => "one-dark",
            Theme::Gruvbox => "gruvbox",
        }
    }

    /// Colors used throughout the UI, keyed by purpose.
    pub fn colors(self) -> Colors {
        match self {
            Theme::Default => Colors {
                // magenta accent
                accent: Color::Magenta,
                // bright magenta for status-line pilot
                accent_mild: Color::Magenta,
                cyan: Color::Cyan,
                magenta: Color::Magenta,
                yellow: Color::Yellow,
                red: Color::Red,
                green: Color::Green,
                white: Color::White,
                black_on_bg: Color::Yellow,
            },
            Theme::Nord => Colors {
                // lavender
                accent: Color::from_u32(0x88C0D0), // nord4
                accent_mild: Color::from_u32(0x88C0D0),
                cyan: Color::from_u32(0x8FBCBB),   // nord2
                magenta: Color::from_u32(0x8FBCBB),
                yellow: Color::from_u32(0xEBCB8B), // nord6
                red: Color::from_u32(0xBF616A),    // nord1
                green: Color::from_u32(0xA3BE8A),  // nord3
                white: Color::from_u32(0xECEFF4),  // nord8
                black_on_bg: Color::from_u32(0xEBCB8B),
            },
            Theme::Dracula => Colors {
                // purple
                accent: Color::from_u32(0xBD93F9),
                accent_mild: Color::from_u32(0xBD93F9),
                cyan: Color::from_u32(0x8BE9FD),
                magenta: Color::from_u32(0xFF79C6),
                yellow: Color::from_u32(0xF1FA58),
                red: Color::from_u32(0xFF5555),
                green: Color::from_u32(0x50FA7B),
                white: Color::from_u32(0xF8F8F2),
                black_on_bg: Color::from_u32(0xF1FA58),
            },
            Theme::Catppuccin => Colors {
                // teal/green
                accent: Color::from_u32(0x89B4FA),
                accent_mild: Color::from_u32(0x89B4FA),
                cyan: Color::from_u32(0x89B4FA),
                magenta: Color::from_u32(0xF5C2E7),
                yellow: Color::from_u32(0xF9E2AF),
                red: Color::from_u32(0xF38BA8),
                green: Color::from_u32(0xA6E3A1),
                white: Color::from_u32(0xF5E0DC),
                black_on_bg: Color::from_u32(0xF9E2AF),
            },
            Theme::Monokai => Colors {
                // orange accent
                accent: Color::from_u32(0xFF9600),
                accent_mild: Color::from_u32(0xFF9600),
                cyan: Color::from_u32(0x8E9C79),
                magenta: Color::from_u32(0xFF79C6),
                yellow: Color::from_u32(0xE6DB74),
                red: Color::from_u32(0xFF6565),
                green: Color::from_u32(0xA6E3A1),
                white: Color::from_u32(0xF8F8F2),
                black_on_bg: Color::from_u32(0xE6DB74),
            },
            Theme::OneDark => Colors {
                // teal accent
                accent: Color::from_u32(0x61AFEF),
                accent_mild: Color::from_u32(0x61AFEF),
                cyan: Color::from_u32(0x56B6C2),
                magenta: Color::from_u32(0xE06CFF),
                yellow: Color::from_u32(0xE5C054),
                red: Color::from_u32(0xE06C75),
                green: Color::from_u32(0x98C379),
                white: Color::from_u32(0xABB2BF),
                black_on_bg: Color::from_u32(0xE5C054),
            },
            Theme::Gruvbox => Colors {
                // warm green accent
                accent: Color::from_u32(0xB8BB26),
                accent_mild: Color::from_u32(0xD5C4A1),
                cyan: Color::from_u32(0x83A598),
                magenta: Color::from_u32(0xD3A6A6),
                yellow: Color::from_u32(0xF9E2AF),
                red: Color::from_u32(0xCC5555),
                green: Color::from_u32(0xA9B665),
                white: Color::from_u32(0xEBDBB2),
                black_on_bg: Color::from_u32(0xF9E2AF),
            },
        }
    }

    /// Get a style for a purpose.
    pub fn style(self, purpose: ColorPurpose) -> Style {
        let c = self.colors();
        match purpose {
            ColorPurpose::Accent => Style::new().fg(c.accent).add_modifier(Modifier::BOLD),
            ColorPurpose::AccentMild => Style::new().fg(c.accent_mild),
            ColorPurpose::Cyan => Style::new().fg(c.cyan),
            ColorPurpose::CyanBold => Style::new().fg(c.cyan).add_modifier(Modifier::BOLD),
            ColorPurpose::Magenta => Style::new().fg(c.magenta),
            ColorPurpose::Yellow => Style::new().fg(c.yellow),
            ColorPurpose::Red => Style::new().fg(c.red),
            ColorPurpose::Green => Style::new().fg(c.green),
            ColorPurpose::White => Style::new().fg(c.white),
            ColorPurpose::BlackOnBg => Style::new().fg(Color::Black).bg(c.black_on_bg),
        }
    }
}

/// A palette of 9 colors, keyed by what they decorate.
pub struct Colors {
    /// Bold accent for titles and banners.
    pub accent: Color,
    /// Mild accent for status-line pilot text.
    pub accent_mild: Color,
    /// User prompt prefix and help commands.
    pub cyan: Color,
    /// Agent phases, spinner, shell prefix.
    pub magenta: Color,
    /// Permissions, warnings, AcceptEdits mode.
    pub yellow: Color,
    /// Errors, deletions, AcceptAll mode, high ctx.
    pub red: Color,
    /// Additions, success.
    pub green: Color,
    /// Answer bullets.
    pub white: Color,
    /// Background of the scroll-down indicator.
    pub black_on_bg: Color,
}

/// What a color is used for.
pub enum ColorPurpose {
    Accent,        // bold title
    AccentMild,    // non-bold accent
    Cyan,          // user prefix
    CyanBold,      // bold user prefix
    Magenta,       // spinner, phases
    Yellow,        // warnings, permissions
    Red,           // errors, deletions
    Green,         // additions, success
    White,         // bullets
    BlackOnBg,     // scroll indicator bg
}

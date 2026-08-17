//! Terminal presentation: colour when a human is watching, plain otherwise.

use std::io::IsTerminal;
use std::sync::OnceLock;

use indicatif::{ProgressBar, ProgressStyle};

fn colour_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        // NO_COLOR is the convention; a pipe or a redirect means nobody is
        // reading escape codes.
        std::env::var_os("NO_COLOR").is_none() && std::io::stdout().is_terminal()
    })
}

macro_rules! paint {
    ($name:ident, $code:literal) => {
        pub fn $name(s: &str) -> String {
            if colour_enabled() {
                format!("\x1b[{}m{s}\x1b[0m", $code)
            } else {
                s.to_string()
            }
        }
    };
}

paint!(dim, "2");
paint!(bold, "1");
paint!(green, "32");
paint!(yellow, "33");
paint!(blue, "36");
paint!(red, "31");

/// Render a tag list the way it will appear in the filename.
pub fn tags(list: &[String]) -> String {
    list.iter()
        .map(|t| blue(&format!("#{t}")))
        .collect::<Vec<_>>()
        .join(" ")
}

/// A progress bar that stays out of the way when output is not a terminal.
pub fn progress(len: u64, what: &str) -> ProgressBar {
    if !std::io::stderr().is_terminal() {
        return ProgressBar::hidden();
    }
    let bar = ProgressBar::new(len);
    bar.set_style(
        ProgressStyle::with_template(
            "  {msg} [{bar:30.cyan/blue}] {pos}/{len} {percent:>3}%  eta {eta}",
        )
        .unwrap()
        .progress_chars("=> "),
    );
    bar.set_message(what.to_string());
    bar
}

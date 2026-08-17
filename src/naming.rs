//! Turning a tag list into a filename.
//!
//! Tags live in the filename on purpose: launchers like lumen feed bare
//! filenames to rofi, so whatever is in the name is what rofi can search.

/// Longest name we will produce, in bytes. Most filesystems cap a single path
/// component at 255; the margin leaves room for the `-2` collision suffix.
const MAX_NAME_BYTES: usize = 250;

/// Does this filename already carry tags?
pub fn is_tagged(name: &str) -> bool {
    name.contains('#')
}

/// Strip every existing `#tag`, along with the separator in front of it, then
/// any separator left dangling at the end.
pub fn strip_tags(base: &str) -> String {
    let chars: Vec<char> = base.chars().collect();
    let mut out: Vec<char> = Vec::with_capacity(chars.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '#' {
            let mut j = i + 1;
            while j < chars.len() && chars[j].is_ascii_alphanumeric() {
                j += 1;
            }
            // A bare '#' with no word after it is part of the name, not a tag.
            if j > i + 1 {
                while matches!(out.last(), Some('-' | '_' | ' ')) {
                    out.pop();
                }
                i = j;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    let s: String = out.into_iter().collect();
    s.trim_end_matches(['-', '_', ' ', '.']).to_string()
}

/// Build `base-#tag-#tag.ext`, dropping tags that would overflow the name.
///
/// Returns `None` when not a single tag fits, which leaves the caller free to
/// report the file as untouched rather than renaming it to something useless.
pub fn compose(base: &str, ext: &str, tags: &[String], max_tags: usize) -> Option<String> {
    let base = if base.is_empty() { "wallpaper" } else { base };
    let mut suffix = String::new();
    let mut kept = 0;

    for tag in tags {
        if max_tags > 0 && kept >= max_tags {
            break;
        }
        let candidate = format!("{suffix}-#{tag}");
        if base.len() + candidate.len() + ext.len() + 1 > MAX_NAME_BYTES {
            break;
        }
        suffix = candidate;
        kept += 1;
    }

    if kept == 0 {
        return None;
    }
    Some(format!("{base}{suffix}.{ext}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_tags_and_separator() {
        assert_eq!(strip_tags("sunset-#orange-#blue"), "sunset");
        assert_eq!(strip_tags("my wallpaper #dark #cool"), "my wallpaper");
        assert_eq!(strip_tags("a_#red"), "a");
    }

    #[test]
    fn keeps_names_that_only_look_tagged() {
        assert_eq!(strip_tags("track #"), "track #");
        assert_eq!(strip_tags("no-tags-here"), "no-tags-here");
        // Non-ASCII survives untouched.
        assert_eq!(strip_tags("forêt-#green"), "forêt");
    }

    #[test]
    fn a_name_of_nothing_but_tags_strips_to_nothing() {
        // --clean leans on this: an empty result means there is no original
        // name to go back to, so the file is left for its owner to name.
        assert_eq!(strip_tags("#red-#blue"), "");
        assert_eq!(strip_tags("-#red"), "");
    }

    #[test]
    fn round_trips() {
        let tags = vec!["orange".to_string(), "blue".to_string()];
        let name = compose("sunset", "png", &tags, 0).unwrap();
        assert_eq!(name, "sunset-#orange-#blue.png");
        assert_eq!(strip_tags(name.rsplit_once('.').unwrap().0), "sunset");
    }

    #[test]
    fn honours_max_tags_and_length() {
        let tags = vec!["a".into(), "b".into(), "c".into()];
        assert_eq!(compose("x", "png", &tags, 2).unwrap(), "x-#a-#b.png");
        let long = "w".repeat(245);
        assert!(compose(&long, "png", &tags, 0).is_none());
    }

    #[test]
    fn empty_base_gets_a_name() {
        assert_eq!(compose("", "jpg", &["red".into()], 0).unwrap(), "wallpaper-#red.jpg");
    }
}

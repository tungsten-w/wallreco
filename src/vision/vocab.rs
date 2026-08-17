//! The label vocabulary: what CLIP is asked to look for, and how the answers
//! are turned back into tags.
//!
//! Scoring happens per category rather than over the whole vocabulary. Asking
//! "forest, city, beach or desert?" produces a sharp answer; asking "which of
//! these ninety things is it?" spreads the probability so thin that nothing
//! ever clears a threshold worth having.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

use anyhow::{Result, bail};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mode {
    /// The prompts are alternatives; at most one tag comes out.
    One,
    /// Several may be true at once.
    Many,
}

#[derive(Debug, Clone)]
pub struct Entry {
    /// `None` marks a decoy: scored like the rest, never emitted.
    pub tag: Option<String>,
    pub prompts: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Category {
    pub name: String,
    pub mode: Mode,
    pub threshold: f32,
    pub max: usize,
    /// Only score this category if the named earlier category produced a tag.
    ///
    /// Time of day and weather only mean something once the picture is agreed
    /// to be a place: without this, a samurai on a flat red background comes
    /// out `#night #storm`, because "a scene at night" really is the closest
    /// of the options offered and softmax has to put its mass somewhere.
    pub requires: Option<String>,
    pub entries: Vec<Entry>,
}

#[derive(Debug, Clone)]
pub struct Vocab {
    pub categories: Vec<Category>,
    digest: u64,
}

impl Vocab {
    pub fn parse(text: &str) -> Result<Vocab> {
        let mut categories: Vec<Category> = Vec::new();

        for (lineno, raw) in text.lines().enumerate() {
            let line = raw.trim();
            let line = match line.find('#') {
                // '#' only starts a comment at the beginning of a line, so a
                // prompt is free to mention one.
                Some(0) => "",
                _ => line,
            };
            if line.is_empty() {
                continue;
            }

            if line.starts_with('[') {
                categories.push(parse_header(line, lineno + 1)?);
                continue;
            }

            let Some((lhs, rhs)) = line.split_once('=') else {
                bail!("line {}: expected `tag = prompt`, got {raw:?}", lineno + 1);
            };
            let Some(cat) = categories.last_mut() else {
                bail!("line {}: entry before any [category]", lineno + 1);
            };

            let key = lhs.trim();
            let prompts: Vec<String> = rhs
                .split('|')
                .map(|p| p.trim().to_string())
                .filter(|p| !p.is_empty())
                .collect();
            if prompts.is_empty() {
                bail!("line {}: no prompt after '='", lineno + 1);
            }
            if key != "~" && !key.chars().all(|c| c.is_ascii_alphanumeric() || c == '-') {
                bail!("line {}: tag {key:?} must be alphanumeric or dashes — it becomes part of a filename", lineno + 1);
            }

            cat.entries.push(Entry {
                tag: (key != "~").then(|| key.to_string()),
                prompts,
            });
        }

        if categories.is_empty() {
            bail!("vocabulary is empty");
        }
        for (i, cat) in categories.iter().enumerate() {
            if let Some(required) = &cat.requires {
                // Earlier only: scoring is a single pass, so a category cannot
                // depend on one that has not been decided yet.
                if !categories[..i].iter().any(|c| &c.name == required) {
                    bail!(
                        "category [{}] requires [{required}], which is not defined above it",
                        cat.name
                    );
                }
            }
            if !cat.entries.iter().any(|e| e.tag.is_some()) {
                bail!("category [{}] has no taggable entry", cat.name);
            }
            if !cat.entries.iter().any(|e| e.tag.is_none()) {
                bail!(
                    "category [{}] has no `~` decoy — without one every image is forced into a tag",
                    cat.name
                );
            }
        }

        let mut hasher = DefaultHasher::new();
        text.hash(&mut hasher);
        Ok(Vocab { categories, digest: hasher.finish() })
    }

    /// Identifies this exact vocabulary, so cached embeddings are dropped the
    /// moment the user edits a prompt.
    pub fn digest(&self) -> u64 {
        self.digest
    }

    pub fn entries(&self) -> impl Iterator<Item = &Entry> {
        self.categories.iter().flat_map(|c| c.entries.iter())
    }

    pub fn entry_count(&self) -> usize {
        self.categories.iter().map(|c| c.entries.len()).sum()
    }

    pub fn prompt_count(&self) -> usize {
        self.entries().map(|e| e.prompts.len()).sum()
    }

    pub fn tag_count(&self) -> usize {
        self.entries().filter(|e| e.tag.is_some()).count()
    }
}

fn parse_header(line: &str, lineno: usize) -> Result<Category> {
    let Some(close) = line.find(']') else {
        bail!("line {lineno}: unterminated [category] header");
    };
    let name = line[1..close].trim().to_string();
    if name.is_empty() {
        bail!("line {lineno}: category has no name");
    }

    let mut cat = Category {
        name,
        mode: Mode::One,
        threshold: 0.35,
        max: 1,
        requires: None,
        entries: Vec::new(),
    };

    for opt in line[close + 1..].split_whitespace() {
        let Some((k, v)) = opt.split_once('=') else {
            bail!("line {lineno}: expected key=value, got {opt:?}");
        };
        match k {
            "mode" => {
                cat.mode = match v {
                    "one" => Mode::One,
                    "many" => Mode::Many,
                    _ => bail!("line {lineno}: mode must be `one` or `many`, got {v:?}"),
                };
                if cat.mode == Mode::Many && cat.max == 1 {
                    cat.max = usize::MAX;
                }
            }
            "threshold" => {
                cat.threshold = v
                    .parse()
                    .map_err(|_| anyhow::anyhow!("line {lineno}: threshold {v:?} is not a number"))?;
                if !(0.0..=1.0).contains(&cat.threshold) {
                    bail!("line {lineno}: threshold must be between 0 and 1");
                }
            }
            "max" => {
                cat.max = v
                    .parse()
                    .map_err(|_| anyhow::anyhow!("line {lineno}: max {v:?} is not a number"))?;
            }
            "requires" => cat.requires = Some(v.to_string()),
            _ => bail!("line {lineno}: unknown option {k:?}"),
        }
    }
    if cat.mode == Mode::One {
        cat.max = 1;
    }
    Ok(cat)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "
# a comment
[style] mode=one threshold=0.5
anime = anime artwork | anime style
~ = a photo

[subject] mode=many threshold=0.2 max=2
cat = a cat
dog = a dog
~ = nothing
";

    #[test]
    fn parses_categories_and_options() {
        let v = Vocab::parse(SAMPLE).unwrap();
        assert_eq!(v.categories.len(), 2);
        assert_eq!(v.categories[0].mode, Mode::One);
        assert_eq!(v.categories[0].threshold, 0.5);
        assert_eq!(v.categories[0].entries[0].prompts.len(), 2);
        assert_eq!(v.categories[1].max, 2);
        assert_eq!(v.tag_count(), 3);
        assert_eq!(v.prompt_count(), 6);
    }

    #[test]
    fn decoys_are_not_tags() {
        let v = Vocab::parse(SAMPLE).unwrap();
        assert!(v.categories[0].entries[1].tag.is_none());
    }

    #[test]
    fn the_shipped_vocabulary_is_valid() {
        let v = Vocab::parse(include_str!("../../labels.txt")).unwrap();
        assert!(v.tag_count() > 50, "got {}", v.tag_count());
        for c in &v.categories {
            assert!(c.entries.iter().any(|e| e.tag.is_none()), "[{}] needs decoys", c.name);
        }
    }

    #[test]
    fn rejects_a_category_with_no_decoy() {
        let err = Vocab::parse("[x] mode=one\na = a thing\n").unwrap_err();
        assert!(err.to_string().contains("decoy"), "{err}");
    }

    #[test]
    fn rejects_tags_that_would_break_a_filename() {
        assert!(Vocab::parse("[x]\nmy tag = a thing\n~ = other\n").is_err());
    }

    #[test]
    fn digest_tracks_content() {
        let a = Vocab::parse(SAMPLE).unwrap();
        let b = Vocab::parse(&SAMPLE.replace("a cat", "a kitten")).unwrap();
        assert_ne!(a.digest(), b.digest());
    }
}

//! Naming what is actually in the picture, offline, with CLIP.
//!
//! Colour statistics can tell you a wallpaper is dark and blue. They cannot
//! tell you it is a cat. This stage embeds every image with CLIP ViT-B/32 and
//! compares it against the prompt vocabulary in `labels.txt`.
//!
//! Two properties keep it fast enough to run over a whole wallpaper folder:
//! the text side is embedded once and cached to disk, and the image side runs
//! as batches through a single session while rayon does the decoding and
//! pixel preparation in parallel around it.

pub mod download;
pub mod vocab;

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use image::RgbImage;
use image::imageops::FilterType;
use ndarray::Array4;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::TensorRef;

pub use vocab::{Mode, Vocab};

/// CLIP ViT-B/32 preprocessing, from the model's `preprocessor_config.json`.
const INPUT: u32 = 224;
const MEAN: [f32; 3] = [0.48145466, 0.4578275, 0.40821073];
const STD: [f32; 3] = [0.26862954, 0.26130258, 0.27577711];
/// CLIP's trained temperature: `exp(4.6052)`. Turns a cosine similarity in
/// [-1, 1] into a logit whose softmax is calibrated.
const LOGIT_SCALE: f32 = 100.0;

/// How many images go through the vision tower in one call. Big enough to keep
/// every core busy, small enough that the activations stay in cache.
pub const BATCH: usize = 16;

const CACHE_MAGIC: &[u8; 8] = b"WRCLIPE1";

/// One image's worth of prepared pixels: 3x224x224, normalised, CHW.
pub struct Pixels(Vec<f32>);

/// A tag CLIP chose, with the confidence it had.
#[derive(Debug, Clone)]
pub struct Scored {
    pub tag: String,
    pub category: String,
    pub probability: f32,
}

pub struct Clip {
    vision: Session,
    vocab: Vocab,
    /// One L2-normalised embedding per vocabulary entry, in `vocab.entries()`
    /// order, laid out flat as `entries * dim`.
    text: Vec<f32>,
    dim: usize,
}

impl Clip {
    /// Load the vision tower and the vocabulary embeddings.
    ///
    /// The text tower is only touched when the embedding cache misses, and is
    /// dropped again immediately — it is a quarter of a gigabyte that has no
    /// reason to stay resident while images are being tagged.
    pub fn load(dir: &Path, vocab: Vocab, threads: usize) -> Result<Clip> {
        if !download::is_installed(dir) {
            bail!(
                "the object recogniser is not installed — run `wallreco --vision-setup` \
                 (about {} MB, once)",
                download::total_bytes() / 1_000_000
            );
        }

        let text = load_or_build_text_embeddings(dir, &vocab, threads)?;
        let dim = text.len() / vocab.entry_count();

        let vision = session(&dir.join("vision_model.onnx"), threads)?;
        Ok(Clip { vision, vocab, text, dim })
    }

    /// Resize, centre crop and normalise one image the way CLIP expects.
    ///
    /// Cheap and thread safe, so callers run it under rayon while the session
    /// is busy with the previous batch.
    pub fn prepare(img: &RgbImage) -> Pixels {
        // Shortest edge to 224, then centre crop — matching CLIPFeatureExtractor.
        let (w, h) = (img.width().max(1), img.height().max(1));
        let scale = INPUT as f32 / w.min(h) as f32;
        let rw = ((w as f32 * scale).round() as u32).max(INPUT);
        let rh = ((h as f32 * scale).round() as u32).max(INPUT);
        let resized = image::imageops::resize(img, rw, rh, FilterType::CatmullRom);
        let cropped = image::imageops::crop_imm(
            &resized,
            (rw - INPUT) / 2,
            (rh - INPUT) / 2,
            INPUT,
            INPUT,
        )
        .to_image();

        let n = (INPUT * INPUT) as usize;
        let mut out = vec![0f32; 3 * n];
        for (i, p) in cropped.pixels().enumerate() {
            for c in 0..3 {
                out[c * n + i] = (p[c] as f32 / 255.0 - MEAN[c]) / STD[c];
            }
        }
        Pixels(out)
    }

    /// Embed a batch of prepared images. Returns one L2-normalised vector each.
    pub fn embed(&mut self, batch: &[Pixels]) -> Result<Vec<Vec<f32>>> {
        if batch.is_empty() {
            return Ok(Vec::new());
        }
        let n = (INPUT * INPUT) as usize;
        let mut flat = Vec::with_capacity(batch.len() * 3 * n);
        for p in batch {
            flat.extend_from_slice(&p.0);
        }
        let array = Array4::from_shape_vec(
            (batch.len(), 3, INPUT as usize, INPUT as usize),
            flat,
        )
        .context("build pixel tensor")?;

        let outputs = self
            .vision
            .run(ort::inputs!["pixel_values" => TensorRef::from_array_view(&array)?])
            .context("CLIP vision forward pass")?;
        let (shape, data) = outputs
            .get("image_embeds")
            .ok_or_else(|| anyhow::anyhow!("vision model has no `image_embeds` output"))?
            .try_extract_tensor::<f32>()
            .context("read image embeddings")?;

        let dim = *shape.last().context("empty embedding shape")? as usize;
        Ok(data.chunks_exact(dim).map(normalised).collect())
    }

    /// Score one image embedding against the vocabulary.
    pub fn score(&self, embedding: &[f32]) -> Vec<Scored> {
        let mut out: Vec<Scored> = Vec::new();
        let mut offset = 0usize;

        for cat in &self.vocab.categories {
            let count = cat.entries.len();

            // A category that depends on another only gets asked once that one
            // has answered. Its prompts still occupy their slots in `text`, so
            // the offset has to advance either way.
            if let Some(required) = &cat.requires
                && !out.iter().any(|s| &s.category == required)
            {
                offset += count;
                continue;
            }
            let logits: Vec<f32> = (0..count)
                .map(|i| {
                    let e = &self.text[(offset + i) * self.dim..(offset + i + 1) * self.dim];
                    LOGIT_SCALE * dot(embedding, e)
                })
                .collect();
            let probs = softmax(&logits);
            offset += count;

            // Best first, decoys included — a decoy winning is exactly how a
            // category declines to answer.
            let mut ranked: Vec<usize> = (0..count).collect();
            ranked.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));

            let mut kept = 0;
            for i in ranked {
                if kept >= cat.max || probs[i] < cat.threshold {
                    break;
                }
                let Some(tag) = &cat.entries[i].tag else {
                    // In `one` mode a winning decoy ends the category. In
                    // `many` mode it only means this slot is empty.
                    if cat.mode == Mode::One {
                        break;
                    }
                    continue;
                };
                out.push(Scored {
                    tag: tag.clone(),
                    category: cat.name.clone(),
                    probability: probs[i],
                });
                kept += 1;
            }
        }
        out
    }
}

/// The session builder hands back its own error type, which carries the
/// builder itself and so cannot be absorbed by anyhow directly. Flatten it to
/// a message.
fn flatten<T>(what: &'static str) -> impl FnOnce(ort::Error<T>) -> anyhow::Error {
    move |e| anyhow::anyhow!("{what}: {e}")
}

fn session(path: &Path, threads: usize) -> Result<Session> {
    Session::builder()
        .map_err(flatten("create ONNX session builder"))?
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(flatten("set optimisation level"))?
        .with_intra_threads(threads.max(1))
        .map_err(flatten("set thread count"))?
        .commit_from_file(path)
        .with_context(|| format!("load {}", path.display()))
}

// ---------------------------------------------------------------------------
// text side
// ---------------------------------------------------------------------------

fn cache_path(dir: &Path, vocab: &Vocab) -> PathBuf {
    dir.join(format!("prompts-{:016x}.bin", vocab.digest()))
}

fn load_or_build_text_embeddings(dir: &Path, vocab: &Vocab, threads: usize) -> Result<Vec<f32>> {
    let path = cache_path(dir, vocab);
    if let Some(cached) = read_cache(&path, vocab.entry_count()) {
        return Ok(cached);
    }

    let embeddings = embed_prompts(dir, vocab, threads)?;
    let dim = embeddings.len() / vocab.entry_count();
    // A failed cache write costs a slow start next time, nothing more.
    let _ = write_cache(&path, &embeddings, vocab.entry_count(), dim);
    // Old vocabularies leave their own files behind; sweep them.
    if let Ok(entries) = fs::read_dir(dir) {
        for e in entries.flatten() {
            let name = e.file_name();
            let name = name.to_string_lossy();
            if name.starts_with("prompts-") && e.path() != path {
                let _ = fs::remove_file(e.path());
            }
        }
    }
    Ok(embeddings)
}

fn read_cache(path: &Path, entries: usize) -> Option<Vec<f32>> {
    let bytes = fs::read(path).ok()?;
    if bytes.len() < 16 || &bytes[..8] != CACHE_MAGIC {
        return None;
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().ok()?) as usize;
    let dim = u32::from_le_bytes(bytes[12..16].try_into().ok()?) as usize;
    if count != entries || dim == 0 || bytes.len() != 16 + count * dim * 4 {
        return None;
    }
    Some(
        bytes[16..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect(),
    )
}

fn write_cache(path: &Path, data: &[f32], count: usize, dim: usize) -> Result<()> {
    let mut out = Vec::with_capacity(16 + data.len() * 4);
    out.extend_from_slice(CACHE_MAGIC);
    // The header carries the shape so a file left over from a different
    // vocabulary is rejected rather than reinterpreted.
    out.extend_from_slice(&(count as u32).to_le_bytes());
    out.extend_from_slice(&(dim as u32).to_le_bytes());
    for v in data {
        out.extend_from_slice(&v.to_le_bytes());
    }
    fs::write(path, out)?;
    Ok(())
}

fn embed_prompts(dir: &Path, vocab: &Vocab, threads: usize) -> Result<Vec<f32>> {
    use tokenizers::Tokenizer;

    let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
        .map_err(|e| anyhow::anyhow!("load CLIP tokenizer: {e}"))?;
    let mut text = session(&dir.join("text_model.onnx"), threads)?;

    let mut out: Vec<f32> = Vec::new();
    let mut dim = 0usize;

    for entry in vocab.entries() {
        // Prompt ensembling: average the prompts that mean the same thing and
        // renormalise. A tag described three ways is more robust than one.
        let mut acc: Vec<f32> = Vec::new();
        for prompt in &entry.prompts {
            let encoded = tokenizer
                .encode(prompt.as_str(), true)
                .map_err(|e| anyhow::anyhow!("tokenize {prompt:?}: {e}"))?;
            let ids: Vec<i64> = encoded.get_ids().iter().map(|&i| i as i64).collect();
            let shape = [1i64, ids.len() as i64];

            // One prompt per call, so nothing is padded and CLIP's end-of-text
            // pooling lands on the real final token every time. This runs once
            // per vocabulary, then the result is cached.
            let outputs = text
                .run(ort::inputs!["input_ids" => TensorRef::from_array_view((shape, ids.as_slice()))?])
                .with_context(|| format!("CLIP text forward pass for {prompt:?}"))?;
            let (shape, data) = outputs
                .get("text_embeds")
                .ok_or_else(|| anyhow::anyhow!("text model has no `text_embeds` output"))?
                .try_extract_tensor::<f32>()
                .context("read text embedding")?;

            dim = *shape.last().context("empty embedding shape")? as usize;
            let v = normalised(&data[..dim]);
            if acc.is_empty() {
                acc = v;
            } else {
                for (a, b) in acc.iter_mut().zip(v) {
                    *a += b;
                }
            }
        }
        out.extend(normalised(&acc));
    }

    if dim == 0 || out.len() != vocab.entry_count() * dim {
        bail!("text embedding produced {} values for {} entries", out.len(), vocab.entry_count());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// small numerics
// ---------------------------------------------------------------------------

fn normalised(v: &[f32]) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    if sum > 0.0 {
        exp.into_iter().map(|e| e / sum).collect()
    } else {
        vec![0.0; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_sums_to_one() {
        let p = softmax(&[1.0, 2.0, 3.0]);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        assert!(p[2] > p[1] && p[1] > p[0]);
    }

    #[test]
    fn normalised_is_unit_length() {
        let v = normalised(&[3.0, 4.0]);
        assert!((dot(&v, &v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalised_tolerates_zero() {
        assert_eq!(normalised(&[0.0, 0.0]), vec![0.0, 0.0]);
    }
}

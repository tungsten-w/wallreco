//! Decoding images down to the small pixel grid the analyser works on.
//!
//! The bash version shelled out to ImageMagick once per file. Doing it in
//! process removes both the fork and the external dependency, and lets us keep
//! the full size RGB around when the vision stage wants its own 224x224 crop.

use std::path::Path;

use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};

/// A square grid of RGB samples, row major, `size` by `size`.
pub struct Grid {
    pub size: u32,
    pub px: Vec<[f32; 3]>,
}

impl Grid {
    #[inline]
    pub fn at(&self, x: u32, y: u32) -> [f32; 3] {
        self.px[(y * self.size + x) as usize]
    }
}

/// Decode `path`, honour its EXIF orientation and flatten transparency onto
/// black — the same normalisation the ImageMagick pipeline did, so the tuned
/// thresholds in `analyse` still mean what they meant.
pub fn decode(path: &Path) -> Result<RgbImage> {
    let reader = ImageReader::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .with_guessed_format()
        .with_context(|| format!("detect format of {}", path.display()))?;

    // Orientation has to come off the decoder before the pixels are consumed.
    let mut decoder = reader
        .into_decoder()
        .with_context(|| format!("decode {}", path.display()))?;
    let orientation = decoder.orientation().unwrap_or(image::metadata::Orientation::NoTransforms);
    let mut img = DynamicImage::from_decoder(decoder)
        .with_context(|| format!("decode {}", path.display()))?;
    img.apply_orientation(orientation);

    Ok(flatten_on_black(img))
}

/// Composite any alpha channel onto black. Wallpapers with transparent corners
/// otherwise report the colour of undefined pixels.
fn flatten_on_black(img: DynamicImage) -> RgbImage {
    if !img.color().has_alpha() {
        return img.to_rgb8();
    }
    let rgba = img.to_rgba8();
    let mut out = RgbImage::new(rgba.width(), rgba.height());
    for (dst, src) in out.pixels_mut().zip(rgba.pixels()) {
        let a = src[3] as u32;
        dst[0] = ((src[0] as u32 * a) / 255) as u8;
        dst[1] = ((src[1] as u32 * a) / 255) as u8;
        dst[2] = ((src[2] as u32 * a) / 255) as u8;
    }
    out
}

/// Box-average `img` down to `size` x `size`.
///
/// Two stages on purpose: a cheap nearest-neighbour prepass caps how much data
/// the good filter has to touch, which matters on 8K wallpapers, and Triangle
/// over an already small image is a true area average — every source pixel gets
/// a say in the colour mass, which is what the histogram measures.
pub fn to_grid(img: &RgbImage, size: u32) -> Grid {
    let prepass = size * 8;
    let small = if img.width() > prepass || img.height() > prepass {
        let scaled = image::imageops::resize(
            img,
            img.width().min(prepass).max(size),
            img.height().min(prepass).max(size),
            FilterType::Nearest,
        );
        image::imageops::resize(&scaled, size, size, FilterType::Triangle)
    } else {
        image::imageops::resize(img, size, size, FilterType::Triangle)
    };

    let px = small
        .pixels()
        .map(|p| {
            [
                p[0] as f32 / 255.0,
                p[1] as f32 / 255.0,
                p[2] as f32 / 255.0,
            ]
        })
        .collect();

    Grid { size, px }
}

//! Colour, mood and scene tags from pixel statistics.
//!
//! This is a port of the awk analyser the shell version used, thresholds
//! included. They were tuned against hand labelled wallpapers, so they are
//! treated as data here rather than something to re-derive: the port is
//! deliberately literal, and `Metrics` exposes every intermediate value so
//! `--explain` can show why a tag did or did not fire.

use crate::sample::Grid;

const NB: usize = 72; // hue bins
const BW: f32 = 360.0 / NB as f32; // 5 degrees each

/// Every intermediate statistic, kept so a human can audit a tagging decision.
#[derive(Debug, Clone, Default)]
pub struct Metrics {
    pub mean_l: f32,
    pub sd_l: f32,
    pub mean_c: f32,
    pub chroma_frac: f32,
    pub achrom_frac: f32,
    pub bright_frac: f32,
    pub vivid_frac: f32,
    pub distinct: usize,
    pub detail: f32,
    pub edge_frac: f32,
    pub flat_frac: f32,
    pub zone_detail: [f32; 3],
    pub grad_res: f32,
    pub ramp: f32,
    pub n_clusters: usize,
    pub zone_l: [f32; 3],
    pub zone_h: [f32; 3],
    pub green_frac: f32,
    pub warm_ratio: f32,
    pub cool_ratio: f32,
    /// Dominant hue clusters, strongest first: (hue, mass fraction, name).
    pub clusters: Vec<(f32, f32, Option<&'static str>)>,
}

/// How many hue clusters may become colour tags.
///
/// The shell version stopped at two. A third is allowed here but has to earn
/// it (see `MIN3_*` below) — a wallpaper genuinely built from three colours
/// gets all three, a two colour one does not pick up a stray.
const MAX_COLORS: usize = 3;
const MIN3_ABS: f32 = 0.045; // third cluster: share of all pixels
const MIN3_REL: f32 = 0.35; // ... and share of the leading cluster

#[inline]
fn hue_of(r: f32, g: f32, b: f32) -> f32 {
    let mx = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let c = mx - mn;
    if c <= 0.0 {
        return -1.0;
    }
    let mut h = if mx == r {
        60.0 * ((g - b) / c)
    } else if mx == g {
        60.0 * (2.0 + (b - r) / c)
    } else {
        60.0 * (4.0 + (r - g) / c)
    };
    if h < 0.0 {
        h += 360.0;
    }
    h
}

/// Map a hue onto the short, searchable colour vocabulary.
///
/// Deliberately small: these are the words someone actually types into rofi.
/// Brown and pink are not hues of their own, they are dark orange/red and pale
/// red, hence the value and saturation tests.
fn color_name(h: f32, s: f32, v: f32) -> Option<&'static str> {
    if h < 0.0 {
        return None;
    }
    Some(if h >= 345.0 || h < 14.0 {
        if v <= 0.45 && s >= 0.25 {
            "brown"
        } else if v >= 0.70 && s <= 0.65 {
            "pink"
        } else {
            "red"
        }
    } else if h < 45.0 {
        if v <= 0.50 { "brown" } else { "orange" }
    } else if h < 68.0 {
        if v <= 0.45 { "brown" } else { "yellow" }
    } else if h < 160.0 {
        "green"
    } else if h < 188.0 {
        "teal"
    } else if h < 205.0 {
        "cyan"
    } else if h < 252.0 {
        "blue"
    } else if h < 292.0 {
        "purple"
    } else if h < 330.0 {
        if v >= 0.62 { "pink" } else { "purple" }
    } else {
        "pink"
    })
}

/// Accumulators filled by the single pass over the grid.
struct Pass {
    n: f32,
    sum_l: f32,
    sum_l2: f32,
    sum_c: f32,
    chroma_n: f32,
    achrom_n: f32,
    bright_n: f32,
    vivid_n: f32,
    luma: Vec<f32>,
    key: Vec<u32>,
    quant: Vec<bool>,
    hb: [f32; NB],
    hb_v: [f32; NB],
    hb_s: [f32; NB],
    hb_v2: [f32; NB],
    hb_s2: [f32; NB],
    hb_w2: [f32; NB],
    hue_mass: f32,
    warm_w: f32,
    cool_w: f32,
    zone_hb: [[f32; NB]; 3],
    zone_mass: [f32; 3],
    zone_n: [f32; 3],
    zone_l: [f32; 3],
    row_l: Vec<f32>,
}

fn scan(grid: &Grid) -> Pass {
    let g = grid.size;
    let mut p = Pass {
        n: 0.0,
        sum_l: 0.0,
        sum_l2: 0.0,
        sum_c: 0.0,
        chroma_n: 0.0,
        achrom_n: 0.0,
        bright_n: 0.0,
        vivid_n: 0.0,
        luma: vec![0.0; (g * g) as usize],
        key: vec![0; (g * g) as usize],
        quant: vec![false; 16 * 16 * 16],
        hb: [0.0; NB],
        hb_v: [0.0; NB],
        hb_s: [0.0; NB],
        hb_v2: [0.0; NB],
        hb_s2: [0.0; NB],
        hb_w2: [0.0; NB],
        hue_mass: 0.0,
        warm_w: 0.0,
        cool_w: 0.0,
        zone_hb: [[0.0; NB]; 3],
        zone_mass: [0.0; 3],
        zone_n: [0.0; 3],
        zone_l: [0.0; 3],
        row_l: vec![0.0; g as usize],
    };

    for y in 0..g {
        let z = ((y * 3 / g) as usize).min(2);
        for x in 0..g {
            let [r, gg, b] = grid.at(x, y);
            let mx = r.max(gg).max(b);
            let mn = r.min(gg).min(b);
            let ch = mx - mn; // chroma: how much colour, absolutely
            let v = mx;
            let s = if mx > 0.0 { ch / mx } else { 0.0 };
            let l = 0.2126 * r + 0.7152 * gg + 0.0722 * b;
            let h = hue_of(r, gg, b);

            p.n += 1.0;
            p.sum_l += l;
            p.sum_l2 += l * l;
            p.sum_c += ch;

            let i = (y * g + x) as usize;
            p.luma[i] = l;
            p.key[i] = ((r * 255.0).round() as u32) << 16
                | ((gg * 255.0).round() as u32) << 8
                | (b * 255.0).round() as u32;

            // Chroma, not HSV saturation, decides whether a pixel reads as
            // coloured: a near black pixel like (0,17,30) has saturation 1.0
            // but no visible colour, which is what used to make every dark
            // wallpaper come out "blue".
            if ch >= 0.10 && v >= 0.10 {
                p.chroma_n += 1.0;
            }
            if ch < 0.07 {
                p.achrom_n += 1.0;
            }
            if l > 0.75 {
                p.bright_n += 1.0;
            }
            if ch >= 0.42 && v >= 0.60 {
                p.vivid_n += 1.0;
            }

            let q = (r * 15.999) as usize * 256 + (gg * 15.999) as usize * 16 + (b * 15.999) as usize;
            p.quant[q] = true;

            if h >= 0.0 && ch >= 0.06 && v >= 0.08 {
                let w = ch * (0.35 + 0.65 * v);
                let bi = ((h / BW) as usize).min(NB - 1);
                p.hb[bi] += w;
                p.hb_v[bi] += w * v;
                p.hb_s[bi] += w * s;
                if ch >= 0.25 {
                    p.hb_v2[bi] += w * v;
                    p.hb_w2[bi] += w;
                    p.hb_s2[bi] += w * s;
                }
                p.hue_mass += w;
                if h < 90.0 || h >= 300.0 {
                    p.warm_w += w;
                } else {
                    p.cool_w += w;
                }
                p.zone_hb[z][bi] += w;
                p.zone_mass[z] += w;
            }

            p.zone_n[z] += 1.0;
            p.zone_l[z] += l;
            p.row_l[y as usize] += l;
        }
    }
    p
}

/// Sum one hue cluster (peak +/- 5 bins).
///
/// Returns `(mass, saturation, value)`. The representative saturation and value
/// come from the strongly chromatic pixels when there are enough of them, so a
/// dark street lit by vivid signs is not called brown.
fn cluster(p: &Pass, peak: Option<usize>) -> (f32, f32, f32) {
    let Some(peak) = peak else {
        return (0.0, 0.0, 0.0);
    };
    let (mut mass, mut sv, mut vv, mut ws) = (0.0, 0.0, 0.0, 0.0);
    let (mut sv2, mut vv2, mut ws2) = (0.0, 0.0, 0.0);
    for k in -5i32..=5 {
        let j = (peak as i32 + k).rem_euclid(NB as i32) as usize;
        mass += p.hb[j];
        sv += p.hb_s[j];
        vv += p.hb_v[j];
        ws += p.hb[j];
        sv2 += p.hb_s2[j];
        vv2 += p.hb_v2[j];
        ws2 += p.hb_w2[j];
    }
    if ws2 > 0.0 && ws > 0.0 && ws2 >= 0.15 * ws {
        (mass, sv2 / ws2, vv2 / ws2)
    } else if ws > 0.0 {
        (mass, sv / ws, vv / ws)
    } else {
        (mass, 0.0, 0.0)
    }
}

/// Circular distance between two hue bins.
#[inline]
fn bin_dist(a: usize, b: usize) -> usize {
    let d = a.abs_diff(b);
    d.min(NB - d)
}

pub struct Analysis {
    pub tags: Vec<String>,
    pub metrics: Metrics,
}

pub fn analyse(grid: &Grid) -> Analysis {
    let g = grid.size;
    let p = scan(grid);

    let mut m = Metrics::default();
    if p.n < 16.0 {
        return Analysis { tags: Vec::new(), metrics: m };
    }

    m.mean_l = p.sum_l / p.n;
    m.mean_c = p.sum_c / p.n;
    let var_l = p.sum_l2 / p.n - m.mean_l * m.mean_l;
    m.sd_l = if var_l > 0.0 { var_l.sqrt() } else { 0.0 };
    m.chroma_frac = p.chroma_n / p.n;
    m.achrom_frac = p.achrom_n / p.n;
    m.bright_frac = p.bright_n / p.n;
    m.vivid_frac = p.vivid_n / p.n;
    m.distinct = p.quant.iter().filter(|q| **q).count();

    // ---- detail, per third, plus flat area fraction -----------------------
    // Neighbouring pixels that are byte for byte identical mean a poster-like
    // flat fill; photographs and painted art almost never have them.
    let (mut e_sum, mut e_n, mut e_strong, mut flat_n) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let mut zone_d = [0.0f32; 3];
    let mut zone_dn = [0.0f32; 3];
    for y in 0..g.saturating_sub(1) {
        let z = ((y * 3 / g) as usize).min(2);
        for x in 0..g.saturating_sub(1) {
            let i = (y * g + x) as usize;
            let a = p.luma[i];
            let d = (a - p.luma[i + 1]).abs() + (a - p.luma[i + g as usize]).abs();
            e_sum += d;
            e_n += 1.0;
            if d > 0.15 {
                e_strong += 1.0;
            }
            if p.key[i] == p.key[i + 1] {
                flat_n += 1.0;
            }
            zone_d[z] += d;
            zone_dn[z] += 1.0;
        }
    }
    m.detail = if e_n > 0.0 { e_sum / e_n } else { 0.0 };
    m.edge_frac = if e_n > 0.0 { e_strong / e_n } else { 0.0 };
    m.flat_frac = if e_n > 0.0 { flat_n / e_n } else { 0.0 };
    for z in 0..3 {
        m.zone_detail[z] = if zone_dn[z] > 0.0 { zone_d[z] / zone_dn[z] } else { 0.0 };
    }

    // ---- vertical gradient: how well row luma fits a straight ramp --------
    let gf = g as f32;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
    for y in 0..g {
        let rm = p.row_l[y as usize] / gf;
        let yf = y as f32;
        sx += yf;
        sy += rm;
        sxx += yf * yf;
        sxy += yf * rm;
    }
    let den = gf * sxx - sx * sx;
    let slope = if den != 0.0 { (gf * sxy - sx * sy) / den } else { 0.0 };
    let icept = (sy - slope * sx) / gf;
    let mut res = 0.0;
    for y in 0..g {
        let rm = p.row_l[y as usize] / gf;
        let d = rm - (slope * y as f32 + icept);
        res += d * d;
    }
    m.grad_res = (res / gf).sqrt();
    m.ramp = slope.abs() * gf;

    for z in 0..3 {
        m.zone_l[z] = if p.zone_n[z] > 0.0 { p.zone_l[z] / p.zone_n[z] } else { 0.0 };
    }

    // ---- dominant hue clusters -------------------------------------------
    let mut sm = [0.0f32; NB];
    for i in 0..NB {
        for k in -3i32..=3 {
            sm[i] += p.hb[(i as i32 + k).rem_euclid(NB as i32) as usize];
        }
    }

    // Peaks strongest first, each at least 7 bins (35 degrees) from the ones
    // already taken so two tags never name the same colour.
    let mut peaks: Vec<usize> = Vec::new();
    for _ in 0..MAX_COLORS {
        let mut best: Option<usize> = None;
        let mut best_w = 0.0;
        for i in 0..NB {
            if sm[i] > best_w && peaks.iter().all(|&q| bin_dist(i, q) >= 7) {
                best_w = sm[i];
                best = Some(i);
            }
        }
        match best {
            Some(i) => peaks.push(i),
            None => break,
        }
    }

    let mut colors: Vec<&'static str> = Vec::new();
    let mut lead_mass = 0.0f32;
    for (rank, &peak) in peaks.iter().enumerate() {
        let (mass, s, v) = cluster(&p, Some(peak));
        let hue = peak as f32 * BW + BW / 2.0;
        let frac = mass / p.n;
        let keep = match rank {
            // A faint colour still gets named when it is clearly *the* colour
            // of the image, which is how washed out greens or foggy blues keep
            // their tag.
            0 => frac >= 0.030 || (frac >= 0.012 && p.hue_mass > 0.0 && mass >= 0.45 * p.hue_mass),
            1 => frac >= 0.015 && lead_mass > 0.0 && mass >= 0.25 * lead_mass,
            _ => frac >= MIN3_ABS && lead_mass > 0.0 && mass >= MIN3_REL * lead_mass,
        };
        if rank == 0 {
            lead_mass = mass;
        }
        let name = if keep { color_name(hue, s, v) } else { None };
        m.clusters.push((hue, frac, name));
        if let Some(name) = name
            && !colors.contains(&name)
        {
            colors.push(name);
        }
    }

    m.n_clusters = (0..NB)
        .filter(|&i| {
            let lead = (-4i32..=4)
                .all(|k| sm[(i as i32 + k).rem_euclid(NB as i32) as usize] <= sm[i]);
            lead && sm[i] > 0.0 && p.hue_mass > 0.0 && sm[i] / p.hue_mass >= 0.10
        })
        .count();

    for z in 0..3 {
        let mut best = None;
        let mut best_w = 0.0;
        for i in 0..NB {
            if p.zone_hb[z][i] > best_w {
                best_w = p.zone_hb[z][i];
                best = Some(i);
            }
        }
        m.zone_h[z] = best.map_or(-1.0, |i| i as f32 * BW + BW / 2.0);
    }
    let zone_rel = |z: usize| {
        if p.zone_mass[z] > 0.0 && p.zone_n[z] > 0.0 { p.zone_mass[z] / p.zone_n[z] } else { 0.0 }
    };
    let top_blue = m.zone_h[0] >= 195.0 && m.zone_h[0] <= 255.0 && zone_rel(0) >= 0.04;
    let is_warm_hue = |h: f32| (h >= 0.0 && h < 50.0) || h >= 325.0;
    let top_warm = is_warm_hue(m.zone_h[0]) && zone_rel(0) >= 0.04;
    let mid_warm = is_warm_hue(m.zone_h[1]) && zone_rel(1) >= 0.04;

    let green_mass: f32 = p.hb[(68.0 / BW) as usize..=(159.0 / BW) as usize].iter().sum();
    m.green_frac = green_mass / p.n;
    if p.hue_mass > 0.0 {
        m.warm_ratio = p.warm_w / p.hue_mass;
        m.cool_ratio = p.cool_w / p.hue_mass;
    }

    // =======================================================================
    // colours first — they are what people search most
    // =======================================================================
    let mut tags: Vec<String> = Vec::new();
    let emit = |t: &str, tags: &mut Vec<String>| {
        if !t.is_empty() && !tags.iter().any(|e| e == t) {
            tags.push(t.to_string());
        }
    };

    for c in &colors {
        emit(c, &mut tags);
    }
    if m.achrom_frac >= 0.55 || colors.is_empty() {
        emit(
            if m.mean_l < 0.16 {
                "black"
            } else if m.mean_l > 0.82 {
                "white"
            } else {
                "gray"
            },
            &mut tags,
        );
    }

    // ---- mood -------------------------------------------------------------
    let mono = m.achrom_frac >= 0.85;
    let pastel = m.chroma_frac >= 0.25 && m.mean_c < 0.25 && m.mean_l > 0.62;

    if m.mean_l < 0.26 {
        emit("dark", &mut tags);
    }
    if m.mean_l > 0.70 {
        emit("light", &mut tags);
    }
    if mono {
        emit("monochrome", &mut tags);
    }
    if !mono && m.mean_c >= 0.32 && m.chroma_frac >= 0.55 {
        emit("vibrant", &mut tags);
    }
    if pastel {
        emit("pastel", &mut tags);
    }
    if !mono && !pastel && m.mean_c < 0.12 {
        emit("muted", &mut tags);
    }
    if m.n_clusters >= 3 && m.chroma_frac >= 0.30 && m.mean_c >= 0.18 && m.distinct >= 200 {
        emit("colorful", &mut tags);
    }
    if !mono && p.hue_mass > 0.0 {
        if m.warm_ratio >= 0.72 {
            emit("warm", &mut tags);
        } else if m.cool_ratio >= 0.72 {
            emit("cool", &mut tags);
        }
    }

    // =======================================================================
    // content — only the rules that survived checking against hand labelled
    // wallpapers. Scenes that colour statistics cannot separate were dropped
    // rather than guessed at: a wrong tag is worse than a missing one, because
    // it puts the wrong wallpaper in front of you in rofi. City, water,
    // mountain, desert, snow and people all collapsed into their backgrounds —
    // those are the vision stage's job.
    // =======================================================================
    let lead_color = colors.first().copied().unwrap_or("");
    let night = m.mean_l < 0.24 && m.zone_l[0] < 0.30;
    let space = m.mean_l < 0.18
        && m.bright_frac >= 0.002
        && m.bright_frac <= 0.05
        && m.detail < 0.04
        && m.green_frac < 0.02;
    // a real sunset sky is smooth; a warm lit street is not
    let sunset = (top_warm || mid_warm)
        && m.ramp >= 0.12
        && m.zone_detail[0] < 0.09
        && m.mean_c >= 0.15
        && m.zone_l[0] > m.zone_l[2];
    let sky = top_blue && m.zone_l[0] >= 0.35 && m.zone_detail[0] < 0.06 && m.zone_l[0] > m.zone_l[2];
    // green has to be the main colour, or cover real ground: neon green signs
    // on a dark street were otherwise coming out as forest
    let nature = (lead_color == "green" || m.green_frac >= 0.10) && m.detail >= 0.03;
    let forest = nature && m.mean_l < 0.45;
    let gradient = m.grad_res < 0.02 && m.detail < 0.02 && m.ramp >= 0.10;
    let minimal = m.flat_frac >= 0.35 && m.detail < 0.06 && !gradient;
    let pattern = m.detail >= 0.10
        && m.flat_frac < 0.20
        && (m.zone_l[0] - m.zone_l[2]).abs() < 0.08
        && (m.zone_l[0] - m.zone_l[1]).abs() < 0.08;

    if night {
        emit("night", &mut tags);
    }
    if space {
        emit("space", &mut tags);
    }
    if sunset {
        emit("sunset", &mut tags);
    }
    if sky {
        emit("sky", &mut tags);
    }
    if forest {
        emit("forest", &mut tags);
    } else if nature {
        emit("nature", &mut tags);
    }
    if gradient {
        emit("gradient", &mut tags);
    }
    if minimal {
        emit("minimal", &mut tags);
    }
    if pattern {
        emit("pattern", &mut tags);
    }

    Analysis { tags, metrics: m }
}

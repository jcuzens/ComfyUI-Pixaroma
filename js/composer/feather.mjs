// Per-layer border fade: edge blur (RGB) + feather (alpha).
// Single source of truth for the math is docs/composer-feather-math.md.
// Mirrored 1:1 by nodes/node_composition.py (_feather_radius_px /
// _edge_blur_radius_px / _border_profile / _border_fade_arr). Keep them in
// lockstep and run tests/test_feather_math.mjs + tests/test_feather.py.
//
// The treatment runs in DISPLAY space (post scale/flip, pre rotation/paste):
// a rotation preserves distance, so a rotated layer keeps a uniform band.
// Feather fades the alpha to 0 over the last F px of the layer's rectangular
// border; edge blur blends a precomputed gaussian-blurred copy of the RGB
// (sigma = R, edge-clamped like PIL) in over the last R px so a hard border
// colour does not survive as a seam inside the fade.

export const FEATHER_MAX_PX = 200;
export const EDGE_BLUR_MAX_PX = 50;

function slider01(v) {
  const n = Number(v);
  if (!Number.isFinite(n)) return 0;
  return Math.min(100, Math.max(0, n)) / 100;
}

// Slider 0-100 -> radius in px, quadratic (same pattern as the layer blur:
// (v/100)^2 * max). Unreadable values read as 0 (no fade).
export function featherRadiusPx(slider) {
  const v = slider01(slider);
  return v * v * FEATHER_MAX_PX;
}
export function edgeBlurRadiusPx(slider) {
  const v = slider01(slider);
  return v * v * EDGE_BLUR_MAX_PX;
}

// smoothstep falloff of the distance d (px) to the border over radius r:
// 0 for d <= 0, 1 for d >= r; r <= 0 -> 1 (no fade).
export function borderProfile(d, r) {
  if (!(r > 0)) return 1;
  const t = Math.min(1, Math.max(0, d / r));
  return t * t * (3 - 2 * t);
}

// Apply edge blur + feather to an RGBA buffer (w*h*4). `blurred` is a
// precomputed (w*h*3) RGB gaussian (sigma = edgeBlurPx, edge-clamped like
// PIL); pass null when edgeBlurPx <= 0. Returns a new Uint8ClampedArray —
// the clamped store rounds half-to-even, identical to np.rint (parity).
export function applyBorderFade(src, w, h, featherPx, edgeBlurPx, blurred) {
  const out = new Uint8ClampedArray(w * h * 4);
  if (!(featherPx > 0) && !(edgeBlurPx > 0)) {
    out.set(src);
    return out;
  }
  const doBlur = edgeBlurPx > 0;
  const doFeather = featherPx > 0;
  for (let y = 0; y < h; y++) {
    const dy = Math.min(y, h - 1 - y);
    const rowBase = y * w;
    for (let x = 0; x < w; x++) {
      const d = Math.min(dy, Math.min(x, w - 1 - x));
      const o = (rowBase + x) * 4;
      let r = src[o], g = src[o + 1], b = src[o + 2];
      if (doBlur) {
        const m = borderProfile(d, edgeBlurPx);
        const bi = (rowBase + x) * 3;
        const s = 1 - m;
        r = m * src[o] + s * blurred[bi];
        g = m * src[o + 1] + s * blurred[bi + 1];
        b = m * src[o + 2] + s * blurred[bi + 2];
      }
      out[o] = r;
      out[o + 1] = g;
      out[o + 2] = b;
      out[o + 3] = doFeather ? src[o + 3] * borderProfile(d, featherPx) : src[o + 3];
    }
  }
  return out;
}

// Replicated-edge gaussian of `d`'s RGB (sigma = R): the border is
// replicated 2R+2 px (1px stretched) so the filter sees clamped edges like
// PIL's GaussianBlur, then the centre is cropped back. Returns a w*h*3 RGB
// buffer. Kernel reach beyond 2R affects at most the outermost pixel, where
// the feather alpha is ~0 — see docs/composer-feather-math.md.
export function edgeBlurRGB(dctx, R) {
  const w = dctx.canvas.width, h = dctx.canvas.height;
  const d = dctx.canvas;
  const pad = Math.ceil(R * 2) + 2;
  const pw = w + pad * 2, ph = h + pad * 2;
  const p = document.createElement("canvas");
  p.width = pw; p.height = ph;
  const pc = p.getContext("2d");
  pc.drawImage(d, pad, pad, w, h);
  pc.drawImage(d, 0, 0, w, 1, pad, 0, w, pad);
  pc.drawImage(d, 0, h - 1, w, 1, pad, ph - pad, w, pad);
  pc.drawImage(d, 0, 0, 1, h, 0, pad, pad, h);
  pc.drawImage(d, w - 1, 0, 1, h, pw - pad, pad, pad, h);
  pc.drawImage(d, 0, 0, 1, 1, 0, 0, pad, pad);
  pc.drawImage(d, w - 1, 0, 1, 1, pw - pad, 0, pad, pad);
  pc.drawImage(d, 0, h - 1, 1, 1, 0, ph - pad, pad, pad);
  pc.drawImage(d, w - 1, h - 1, 1, 1, pw - pad, ph - pad, pad, pad);
  const b = document.createElement("canvas");
  b.width = pw; b.height = ph;
  const bc = b.getContext("2d");
  bc.filter = "blur(" + R + "px)";
  bc.drawImage(p, 0, 0);
  bc.filter = "none";
  const rgba = bc.getImageData(pad, pad, w, h).data;
  const rgb = new Uint8ClampedArray(w * h * 3);
  for (let i = 0, j = 0; i < w * h; i++, j += 4) {
    rgb[j] = rgba[i * 4];
    rgb[j + 1] = rgba[i * 4 + 1];
    rgb[j + 2] = rgba[i * 4 + 2];
  }
  return rgb;
}

// Display-space border-fade canvas: draws `src` (any drawable) into a
// dw x dh canvas, applies edge blur + feather (see applyBorderFade), and
// returns the treated canvas. null when both radii are 0.
export function borderFadeCanvas(src, dw, dh, featherPx, edgeBlurPx) {
  const F = featherPx > 0 ? featherPx : 0;
  const R = edgeBlurPx > 0 ? edgeBlurPx : 0;
  if (F === 0 && R === 0) return null;
  const w = Math.max(1, Math.round(dw));
  const h = Math.max(1, Math.round(dh));
  const d = document.createElement("canvas");
  d.width = w; d.height = h;
  const dctx = d.getContext("2d");
  dctx.imageSmoothingEnabled = true;
  dctx.imageSmoothingQuality = "high";
  dctx.drawImage(src, 0, 0, w, h);
  let blurred = null;
  if (R > 0) blurred = edgeBlurRGB(dctx, R);
  const data = applyBorderFade(
    dctx.getImageData(0, 0, w, h).data, w, h, F, R, blurred,
  );
  const out = document.createElement("canvas");
  out.width = w; out.height = h;
  out.getContext("2d").putImageData(new ImageData(data, w, h), 0, 0);
  return out;
}

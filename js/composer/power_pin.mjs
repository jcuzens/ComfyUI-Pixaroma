// CC Power Pin: math + warp core.
// Pins are defined in post-crop image pixels; the layer's affine placement
// (cx/cy, scale, rotation, flips) still applies on top of the warped output.
// Reference algorithm: cc_power_pin (effectcraft distort3). DOM-free except
// warpPowerPinCanvas (needs document).
function solve8(A, b) {
  const M = A.map((row, i) => [...row, b[i]]);
  for (let col = 0; col < 8; col++) {
    let piv = col;
    for (let r = col + 1; r < 8; r++)
      if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
    if (Math.abs(M[piv][col]) < 1e-12) return null;
    const tmp = M[col];
    M[col] = M[piv];
    M[piv] = tmp;
    const d = M[col][col];
    for (let j = col; j < 9; j++) M[col][j] /= d;
    for (let r = 0; r < 8; r++) {
      if (r === col) continue;
      const f = M[r][col];
      if (f === 0) continue;
      for (let j = col; j < 9; j++) M[r][j] -= f * M[col][j];
    }
  }
  return M.map((row) => row[8]);
}

// Homography unit square -> quad (q0=TL, q1=TR, q2=BR, q3=BL), 8 values
// [h00 h01 h02 h10 h11 h12 h20 h21] with h22 = 1. null when degenerate.
export function buildHomography(q0, q1, q2, q3) {
  const A = Array.from({ length: 8 }, () => new Array(8).fill(0));
  const b = new Array(8).fill(0);
  const pts = [
    [q0[0], q0[1], 0, 0],
    [q1[0], q1[1], 1, 0],
    [q2[0], q2[1], 1, 1],
    [q3[0], q3[1], 0, 1],
  ];
  for (let i = 0; i < 4; i++) {
    const [dx, dy, sx, sy] = pts[i];
    A[2 * i][0] = sx;
    A[2 * i][1] = sy;
    A[2 * i][2] = 1;
    A[2 * i][6] = -dx * sx;
    A[2 * i][7] = -dx * sy;
    A[2 * i + 1][3] = sx;
    A[2 * i + 1][4] = sy;
    A[2 * i + 1][5] = 1;
    A[2 * i + 1][6] = -dy * sx;
    A[2 * i + 1][7] = -dy * sy;
    b[2 * i] = dx;
    b[2 * i + 1] = dy;
  }
  const h = solve8(A, b);
  if (!h) return null;
  const a = h[0], b1 = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  const det = a * (e - f * hh) - b1 * (d - f * g) + c * (d * hh - e * g);
  if (!isFinite(det) || Math.abs(det) < 1e-9) return null;
  return h;
}

// Apply a homography (8 values, h22 = 1). null when the point sits on the
// line at infinity (|w| ~ 0).
export function hApply(h, x, y) {
  const w = h[6] * x + h[7] * y + 1;
  if (!isFinite(w) || Math.abs(w) < 1e-12) return null;
  return [(h[0] * x + h[1] * y + h[2]) / w, (h[3] * x + h[4] * y + h[5]) / w];
}

// Analytic inverse of a homography (8 values, h22 = 1). adj(H) with the
// result re-normalized to h22 = 1. null when singular.
export function hInverse(h) {
  const a = h[0], b = h[1], c = h[2], d = h[3], e = h[4], f = h[5], g = h[6], hh = h[7];
  const det = a * (e - f * hh) - b * (d - f * g) + c * (d * hh - e * g);
  if (!isFinite(det) || Math.abs(det) < 1e-9) return null;
  const n22 = a * e - b * d;
  if (Math.abs(n22) < 1e-12) return null;
  const inv = 1 / n22;
  return [
    (e - f * hh) * inv, (c * hh - b) * inv, (b * f - c * e) * inv,
    (f * g - d) * inv, (a - c * g) * inv, (c * d - a * f) * inv,
    (d * hh - e * g) * inv, (b * g - a * hh) * inv,
  ];
}

// Coons (bilinear) quad mapping: unit square -> quad, straight edges.
export function coonsQuad(q, u, v) {
  const tx = q[0][0] + (q[1][0] - q[0][0]) * u;
  const bx = q[3][0] + (q[2][0] - q[3][0]) * u;
  const ty = q[0][1] + (q[1][1] - q[0][1]) * u;
  const by = q[3][1] + (q[2][1] - q[3][1]) * u;
  return [tx + (bx - tx) * v, ty + (by - ty) * v];
}

// Newton iteration for the bilinear-quad inverse (used for the perspective <
// 100% blend). Starts from the projective preimage; 8 iterations.
export function newtonBilinearInverse(q, px, py, u0, v0) {
  let bu = u0, bv = v0;
  for (let i = 0; i < 8; i++) {
    const f = coonsQuad(q, bu, bv);
    const rx = f[0] - px, ry = f[1] - py;
    if (Math.abs(rx) + Math.abs(ry) < 1e-6) break;
    const e = 1e-4;
    const fa = coonsQuad(q, bu + e, bv);
    const fc = coonsQuad(q, bu, bv + e);
    const j00 = (fa[0] - f[0]) / e;
    const j10 = (fa[1] - f[1]) / e;
    const j01 = (fc[0] - f[0]) / e;
    const j11 = (fc[1] - f[1]) / e;
    const det = j00 * j11 - j01 * j10;
    if (Math.abs(det) < 1e-12) break;
    bu -= (j11 * rx - j01 * ry) / det;
    bv -= (-j10 * rx + j00 * ry) / det;
  }
  return [bu, bv];
}

// Resolve the full warp setup for a powerPin param object. null when absent
// or degenerate (collinear pins, zero-size expanded rect, NaN pins).
export function powerPinSetup(pp, W, H) {
  if (!pp || !Array.isArray(pp.pins) || pp.pins.length !== 4) return null;
  const q = pp.pins.map((p) => [Number(p[0]), Number(p[1])]);
  for (const p of q)
    if (!isFinite(p[0]) || !isFinite(p[1])) return null;
  const p = Math.min(1, Math.max(0, Number(pp.perspective ?? 100) / 100));
  const eT = Number(pp.expandTop ?? 0) / 100;
  const eL = Number(pp.expandLeft ?? 0) / 100;
  const eR = Number(pp.expandRight ?? 0) / 100;
  const eB = Number(pp.expandBottom ?? 0) / 100;
  const unstretch = !!pp.unstretch;
  const x0 = -eL * W, y0 = -eT * H;
  const x1 = W * (1 + eR), y1 = H * (1 + eB);
  if (x1 - x0 <= 0 || y1 - y0 <= 0) return null;
  const h = buildHomography(q[0], q[1], q[2], q[3]);
  if (!h) return null;
  let ox = 0, oy = 0, ow = W, oh = H, inv = null;
  if (!unstretch) {
    const xs = q.map((pt) => pt[0]);
    const ys = q.map((pt) => pt[1]);
    ox = Math.floor(Math.min(...xs));
    oy = Math.floor(Math.min(...ys));
    ow = Math.max(1, Math.ceil(Math.max(...xs)) - ox);
    oh = Math.max(1, Math.ceil(Math.max(...ys)) - oy);
    inv = hInverse(h);
    if (!inv) return null;
  }
  return { q, p, x0, y0, x1, y1, unstretch, ox, oy, ow, oh, h, inv, W, H };
}

// Warp a pixel buffer (RGBA, w*h*4) into the power-pin output. outScale < 1
// renders a smaller preview (each output pixel samples the source at the
// center of a (1/outScale)-pixel block). Returns {data, w, h}; pixels outside
// the quad (or outside the source image) stay transparent. The clamped-array
// store quantizes round-half-to-even — identical to np.rint (parity).
export function warpPowerPinToBuffer(src, sw, sh, setup, outScale = 1) {
  const { q, p, x0, y0, x1, y1, unstretch, ox, oy, ow, oh, h, inv } = setup;
  const owOut = Math.max(1, Math.round(ow * outScale));
  const ohOut = Math.max(1, Math.round(oh * outScale));
  const out = new Uint8ClampedArray(owOut * ohOut * 4);
  const step = 1 / outScale;
  for (let j = 0; j < ohOut; j++) {
    const py = oy + (j + 0.5) * step;
    for (let i = 0; i < owOut; i++) {
      const px = ox + (i + 0.5) * step;
      let u, v;
      if (!unstretch) {
        const r = hApply(inv, px, py);
        if (!r) continue;
        u = r[0];
        v = r[1];
        if (p < 1) {
          const bv = newtonBilinearInverse(q, px, py, u, v);
          u = bv[0] + (u - bv[0]) * p;
          v = bv[1] + (v - bv[1]) * p;
        }
        if (u < -1e-9 || u > 1 + 1e-9 || v < -1e-9 || v > 1 + 1e-9) continue;
        u = Math.min(1, Math.max(0, u));
        v = Math.min(1, Math.max(0, v));
      } else {
        u = (px - x0) / (x1 - x0);
        v = (py - y0) / (y1 - y0);
        if (u < -1e-9 || u > 1 + 1e-9 || v < -1e-9 || v > 1 + 1e-9) continue;
      }
      let lx, ly;
      if (!unstretch) {
        lx = x0 + u * (x1 - x0);
        ly = y0 + v * (y1 - y0);
      } else {
        const b = coonsQuad(q, u, v);
        const m = hApply(h, u, v);
        if (!m) continue;
        lx = b[0] + (m[0] - b[0]) * p;
        ly = b[1] + (m[1] - b[1]) * p;
      }
      const i0 = Math.floor(lx - 0.5);
      const j0 = Math.floor(ly - 0.5);
      if (i0 < -1 || j0 < -1 || i0 >= sw || j0 >= sh) continue;
      const fx = Math.min(1, Math.max(0, lx - (i0 + 0.5)));
      const fy = Math.min(1, Math.max(0, ly - (j0 + 0.5)));
      const o = (i + j * owOut) * 4;
      for (let c = 0; c < 4; c++) {
        // Transparent padding: a neighbor outside the source contributes 0,
        // so exact pixel centers (identity) sample pixel-exact and the source
        // edge fades over the half pixel (clean border under Expand).
        const c00 = i0 >= 0 && j0 >= 0 ? src[(j0 * sw + i0) * 4 + c] : 0;
        const c10 = i0 + 1 < sw && j0 >= 0 ? src[(j0 * sw + i0 + 1) * 4 + c] : 0;
        const c01 = i0 >= 0 && j0 + 1 < sh ? src[((j0 + 1) * sw + i0) * 4 + c] : 0;
        const c11 = i0 + 1 < sw && j0 + 1 < sh ? src[((j0 + 1) * sw + i0 + 1) * 4 + c] : 0;
        const top = c00 * (1 - fx) + c10 * fx;
        const bot = c01 * (1 - fx) + c11 * fx;
        out[o + c] = top * (1 - fy) + bot * fy;
      }
    }
  }
  return { data: out, w: owOut, h: ohOut };
}

// Supersampled warp (the per-layer antialias toggle): warps at ss pixels per
// output pixel, then box-averages each ss*ss block back to (ow, oh). The
// integer sum is exact (< 2^24) and the clamped store rounds half-to-even —
// pixel-identical to the Python reshape-mean + np.rint (parity).
export function warpPowerPinToBufferAA(src, sw, sh, setup, ss = 4) {
  const r = warpPowerPinToBuffer(src, sw, sh, setup, ss);
  const w = r.w / ss;
  const h = r.h / ss;
  const out = new Uint8ClampedArray(w * h * 4);
  for (let j = 0; j < h; j++)
    for (let i = 0; i < w; i++) {
      const o = (i + j * w) * 4;
      for (let c = 0; c < 4; c++) {
        let s = 0;
        for (let dy = 0; dy < ss; dy++)
          for (let dx = 0; dx < ss; dx++)
            s += r.data[((j * ss + dy) * r.w + (i * ss + dx)) * 4 + c];
        out[o + c] = s / (ss * ss);
      }
    }
  return { data: out, w, h };
}

// Read a DOM source (HTMLImageElement | HTMLCanvasElement) into raw RGBA.
function sourceToRGBA(source) {
  const sw = source.naturalWidth || source.width;
  const sh = source.naturalHeight || source.height;
  const scratch = document.createElement("canvas");
  scratch.width = sw;
  scratch.height = sh;
  const sctx = scratch.getContext("2d", { willReadFrequently: true });
  sctx.drawImage(source, 0, 0);
  return { data: sctx.getImageData(0, 0, sw, sh).data, sw, sh };
}

// DOM wrapper: draw the source into a full-res scratch canvas, warp, wrap the
// result in a canvas. source = HTMLImageElement | HTMLCanvasElement.
export function warpPowerPinCanvas(source, setup, outScale = 1) {
  const { data, sw, sh } = sourceToRGBA(source);
  const r = warpPowerPinToBuffer(data, sw, sh, setup, outScale);
  return canvasFromBuffer(r);
}

// DOM wrapper for the supersampled warp (full-quality antialias pass).
export function warpPowerPinCanvasAA(source, setup, ss = 4) {
  const { data, sw, sh } = sourceToRGBA(source);
  return canvasFromBuffer(warpPowerPinToBufferAA(data, sw, sh, setup, ss));
}

function canvasFromBuffer(r) {
  const out = document.createElement("canvas");
  out.width = r.w;
  out.height = r.h;
  out.getContext("2d").putImageData(new ImageData(r.data, r.w, r.h), 0, 0);
  return out;
}

// Full affine image-frame -> doc. MUST stay in lockstep with core.mjs
// getCoordinatesInLayerImage (the inverse) and the render.mjs draw chain
// (translate -> rotate -> scale(flips) -> draw).
export function affineToDoc(layer, px, py, W, H) {
  let x = (px - W / 2) * (layer.scaleX || 1);
  let y = (py - H / 2) * (layer.scaleY || 1);
  if (layer.flippedX) x = -x;
  if (layer.flippedY) y = -y;
  const rad = ((layer.rotation || 0) * Math.PI) / 180;
  return {
    x: x * Math.cos(rad) - y * Math.sin(rad) + (layer.cx || 0),
    y: x * Math.sin(rad) + y * Math.cos(rad) + (layer.cy || 0),
  };
}

// doc -> image frame (inverse of affineToDoc).
export function affineFromDoc(layer, dx, dy, W, H) {
  const x0 = dx - (layer.cx || 0);
  const y0 = dy - (layer.cy || 0);
  const rad = (-(layer.rotation || 0) * Math.PI) / 180;
  const rx = x0 * Math.cos(rad) - y0 * Math.sin(rad);
  const ry = x0 * Math.sin(rad) + y0 * Math.cos(rad);
  const fx = rx * (layer.flippedX ? -1 : 1);
  const fy = ry * (layer.flippedY ? -1 : 1);
  return {
    x: fx / (layer.scaleX || 1) + W / 2,
    y: fy / (layer.scaleY || 1) + H / 2,
  };
}

// Doc-space positions of the 4 pins (selection handles, hit tests).
export function pinnedPointsToDoc(layer, W, H) {
  return layer.powerPin.pins.map((pt) => affineToDoc(layer, pt[0], pt[1], W, H));
}

// Selection-overlay local frame: the overlay canvas applies
// translate(cx+pad, cy+pad) then rotate, but NOT flip/scale (see render.mjs
// overlay block). A point drawn at (lx, ly) lands at doc (cx,cy) + R·(lx,ly);
// a pin lands at (cx,cy) + R·F·S·(p−c) — so the overlay coord is F·S·(p−c).
export function pinnedPointsToLocal(layer, W, H) {
  return layer.powerPin.pins.map((pt) => ({
    x: (pt[0] - W / 2) * (layer.scaleX || 1) * (layer.flippedX ? -1 : 1),
    y: (pt[1] - H / 2) * (layer.scaleY || 1) * (layer.flippedY ? -1 : 1),
  }));
}

// Is a doc point inside the (possibly warped) quad? Transforms to the image
// frame, then tests the unit square via the map's inverse. Degenerate -> false.
export function docPointInPinQuad(layer, dx, dy, W, H) {
  const setup = powerPinSetup(layer.powerPin, W, H);
  if (!setup) return false;
  const q = affineFromDoc(layer, dx, dy, W, H);
  if (!setup.unstretch) {
    const r = hApply(setup.inv, q.x, q.y);
    return !!r && r[0] >= -1e-9 && r[0] <= 1 + 1e-9 && r[1] >= -1e-9 && r[1] <= 1 + 1e-9;
  }
  const u = (q.x - setup.x0) / (setup.x1 - setup.x0);
  const v = (q.y - setup.y0) / (setup.y1 - setup.y0);
  return u >= -1e-9 && u <= 1 + 1e-9 && v >= -1e-9 && v <= 1 + 1e-9;
}

// layer.img was replaced with a different-size image: scale the pins so they
// keep pointing at the same image corners/content. No-op on same size / no pin.
export function rescalePowerPin(layer, oldW, oldH) {
  const pp = layer.powerPin;
  if (!pp || !pp.pins) return;
  const nx = layer.img.width || layer.img.naturalWidth;
  const ny = layer.img.height || layer.img.naturalHeight;
  if (nx === oldW && ny === oldH) return;
  const sx = nx / oldW, sy = ny / oldH;
  pp.pins = pp.pins.map((pt) => [pt[0] * sx, pt[1] * sy]);
}

// A crop is (re)applied in-session. cropRect is in SOURCE pixels and the
// pre-crop layer.img has its origin at (prevX, prevY) in source coords, so a
// pin in current layer.img coords moves by (prev − cr). Call with cr = (0,0)
// when the crop is reset to the full source.
export function remapPowerPinForCrop(layer, crX, crY, prevX, prevY) {
  const pp = layer.powerPin;
  if (!pp || !pp.pins) return;
  pp.pins = pp.pins.map((pt) => [pt[0] + prevX - crX, pt[1] + prevY - crY]);
}

// C' - C for a pinned layer (spec 3.8): the quad-AABB center minus the image
// center, through the layer's scale-flip-rotate. The placement translate uses
// C + this so the warped frame never moves on enable (identity pin -> (0,0)).
export function powerPinAnchorShift(layer, setup, W, H) {
  const d = affineToDoc(
    layer,
    setup.ox + setup.ow / 2,
    setup.oy + setup.oh / 2,
    W,
    H,
  );
  return { x: d.x - (layer.cx || 0), y: d.y - (layer.cy || 0) };
}

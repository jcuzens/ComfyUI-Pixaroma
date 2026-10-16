import torch
import numpy as np
from PIL import Image, ImageChops, ImageFilter
import os
import json
import time
import math
import folder_paths
from server import PromptServer
from .node_ref import any_type, FlexibleOptionalInputType
from ._path_guard import safe_join
from ._resize_helpers import normalize_bit_depth
from ._fx_adjust_engine import apply_fx, is_neutral, _fx_seed
from ._bg_removal_helpers import (
    is_birefnet_model_id,
    run_birefnet_on_pil,
    get_birefnet_inventory,
)


# Temp subfolder for Image Composer node previews (cleared by ComfyUI
# between runs). We save the final executed image here so the node's
# mini preview can show the EXACT result — matching what downstream
# Preview Image nodes see, including any auto-rembg applied.
_PIXAROMA_PREVIEW_SUBFOLDER = "pixaroma_composer_preview"


def _save_preview_png(pil_img, doc_w, doc_h):
    """Save the final composed image into ComfyUI's temp dir so the
    ImageComposer node's JS side can fetch it and update the mini
    preview. Returns the UI image descriptor dict expected by
    ComfyUI's onExecuted message."""
    try:
        temp_dir = os.path.join(folder_paths.get_temp_directory(), _PIXAROMA_PREVIEW_SUBFOLDER)
        os.makedirs(temp_dir, exist_ok=True)
        filename = f"composer_{int(time.time() * 1000)}.png"
        path = os.path.join(temp_dir, filename)
        # RGB export — the ImageComposer's canvas is always a flat RGB
        # image from the browser's point of view. Alpha channel from
        # rembg output gets baked onto whatever bg color was set.
        out = pil_img.convert("RGB") if pil_img.mode != "RGB" else pil_img
        out.save(path, format="PNG", optimize=False)
        return {
            "filename": filename,
            "subfolder": _PIXAROMA_PREVIEW_SUBFOLDER,
            "type": "temp",
        }
    except Exception as e:
        print(f"[Pixaroma] preview save failed: {e}")
        return None


# ── Helpers for placeholder compositing ──────────────────────────────────────

def _hex_to_rgba(hex_str):
    try:
        h = (hex_str or "").lstrip("#")
        return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255)
    except (ValueError, IndexError, TypeError):
        return (30, 30, 30, 255)  # neutral fallback on a malformed colour


def _tensor_to_pil(tensor):
    """Convert ComfyUI IMAGE tensor [B,H,W,C] float32 0-1 → PIL RGBA."""
    arr = tensor[0].cpu().numpy()
    arr = (arr * 255).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr, "RGB" if arr.shape[2] == 3 else "RGBA")
    return img.convert("RGBA")


def _pil_to_tensor(img):
    """Convert PIL RGB image → ComfyUI IMAGE tensor [1,H,W,3] float32 0-1."""
    arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
    return torch.from_numpy(arr)[None,]


def _load_server_image(src, input_dir):
    """Load a layer image from the ComfyUI input directory (path-safe)."""
    real_input = os.path.realpath(input_dir)
    full_path = os.path.realpath(os.path.join(input_dir, src))
    if not full_path.startswith(real_input + os.sep):
        return None
    if not os.path.exists(full_path):
        return None
    # A LAYER is a raw file the user put in ComfyUI's input folder, so it can be
    # a 16-bit scan or depth map, and a bare .convert() would CLAMP it to
    # near-white with no error raised. See load-image.md #21; no-op for 8-bit.
    return normalize_bit_depth(Image.open(full_path)).convert("RGBA")


def _remove_background(img, quality="auto"):
    """Remove background from a PIL RGBA image using rembg (returns RGBA).

    Accepts either a modern model id ('u2net' / 'isnet-general-use' /
    'birefnet-general') or a legacy quality tier ('normal' / 'high' /
    'auto'). Tries the requested model first, then walks the auto
    fallback chain (best → lightest) if it's not installed.
    """
    # Legacy tier → modern model. Keeps old saved scenes working.
    _legacy = {"normal": "isnet-general-use", "high": "birefnet-general"}
    requested = _legacy.get(quality, quality) or "auto"

    # Pixaroma BiRefNet variants (birefnet / -hr / -matting) live in our
    # own loader, not rembg. Route explicit picks here so Auto Remove on
    # Execute uses the same model the manual Remove Background button
    # uses (which goes through /pixaroma/remove_bg with the same dispatch).
    if is_birefnet_model_id(requested):
        try:
            return run_birefnet_on_pil(img.convert("RGB"), requested).convert("RGBA")
        except Exception as e:
            print(f"[Pixaroma] Auto Remove BG: BiRefNet '{requested}' failed: {e}")
            return img

    # 'auto' picks whatever is best. Mirror the frontend's pickDefaultModel
    # priority: any installed BiRefNet variant (standard, HR, matting)
    # beats rembg. Falls through to rembg only when no BiRefNet variant
    # is installed (or the inventory lookup itself fails).
    if requested == "auto":
        try:
            for variant in get_birefnet_inventory().get("variants", []):
                if variant.get("installed"):
                    pick = variant["id"]
                    try:
                        return run_birefnet_on_pil(img.convert("RGB"), pick).convert("RGBA")
                    except Exception as e:
                        print(f"[Pixaroma] Auto Remove BG: BiRefNet '{pick}' failed: {e}")
                        return img
        except Exception as e:
            print(f"[Pixaroma] Auto Remove BG: BiRefNet inventory lookup failed: {e}")

    try:
        from rembg import remove, new_session
        import io

        # Try requested first, then fall through the auto chain. This
        # matches the server-side /pixaroma/remove_bg behaviour so the
        # workflow output matches what the Image Composer preview uses.
        auto_chain = ("birefnet-general", "isnet-general-use", "u2net")
        order = (list(auto_chain) if requested == "auto"
                 else [requested] + [n for n in auto_chain if n != requested])
        session = None
        for name in order:
            try:
                session = new_session(name)
                print(f"[Pixaroma] Auto Remove BG: using model '{name}'")
                break
            except Exception as e:
                print(f"[Pixaroma] model '{name}' not available: {e}")
        if session is None:
            print("[Pixaroma] No rembg model could be loaded, skipping remove BG")
            return img

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result_bytes = remove(buf.getvalue(), session=session)
        return Image.open(io.BytesIO(result_bytes)).convert("RGBA")
    except ImportError:
        print("[Pixaroma] rembg not installed — skipping auto remove BG")
        return img
    except Exception as e:
        print(f"[Pixaroma] Auto remove BG failed: {e}")
        return img


def _fit_to_placeholder(img, ph_w, ph_h, mode="cover"):
    """Resize and crop/pad img to fit placeholder dimensions using the given mode."""
    src_w, src_h = img.size
    if mode == "fill":
        return img.resize((ph_w, ph_h), Image.LANCZOS)
    elif mode == "contain":
        scale = min(ph_w / src_w, ph_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        result = Image.new("RGBA", (ph_w, ph_h), (0, 0, 0, 0))
        result.paste(resized, ((ph_w - new_w) // 2, (ph_h - new_h) // 2))
        return result
    else:  # cover
        scale = max(ph_w / src_w, ph_h / src_h)
        new_w = max(1, int(src_w * scale))
        new_h = max(1, int(src_h * scale))
        resized = img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - ph_w) // 2
        top = (new_h - ph_h) // 2
        return resized.crop((left, top, left + ph_w, top + ph_h))


def _apply_eraser_mask(img, mask_path, input_dir):
    """Apply eraser mask to image alpha. Mask white pixels = erased."""
    mask_img = _load_server_image(mask_path, input_dir)
    if mask_img is None:
        return img
    mask_img = mask_img.resize(img.size, Image.LANCZOS)
    # Mask uses destination-out: opaque pixels in mask = erased
    # So we subtract mask alpha from image alpha
    r, g, b, a = img.split()
    mask_a = mask_img.split()[3]  # alpha channel of mask (opaque = erased)
    # destination-out: result_alpha = image_alpha * (1 - mask_alpha). Use
    # multiply(a, invert(mask_a)) NOT subtract, so soft-brush partial alpha and
    # already-transparent pixels match the editor's canvas compositing exactly.
    a = ImageChops.multiply(a, ImageChops.invert(mask_a))
    return Image.merge("RGBA", (r, g, b, a))


def _rgb_to_hsl(rgb):
    """rgb: (...,3) float 0-1 → (...,3) float HSL (H 0-1, S 0-1, L 0-1)."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    l = (mx + mn) / 2
    d = mx - mn
    s = np.where(d == 0, 0, d / np.where(l < 0.5, mx + mn, 2 - mx - mn + 1e-10))
    h = np.zeros_like(l)
    mask = d != 0
    rc = (mx - r) / np.where(d == 0, 1, d)
    gc = (mx - g) / np.where(d == 0, 1, d)
    bc = (mx - b) / np.where(d == 0, 1, d)
    h = np.where((r == mx) & mask, bc - gc, h)
    h = np.where((g == mx) & mask, 2 + rc - bc, h)
    h = np.where((b == mx) & mask, 4 + gc - rc, h)
    h = (h / 6) % 1
    return np.stack([h, s, l], axis=-1)


def _hsl_to_rgb(hsl):
    h, s, l = hsl[..., 0], hsl[..., 1], hsl[..., 2]
    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    def hue2rgb(t):
        t = t % 1
        return np.where(t < 1/6, p + (q - p) * 6 * t,
               np.where(t < 1/2, q,
               np.where(t < 2/3, p + (q - p) * (2/3 - t) * 6, p)))

    r = np.where(s == 0, l, hue2rgb(h + 1/3))
    g = np.where(s == 0, l, hue2rgb(h))
    b = np.where(s == 0, l, hue2rgb(h - 1/3))
    return np.stack([r, g, b], axis=-1)


def _blend_over(base_rgba, top_rgba, mode):
    """Porter-Duff source-over with a blend function per W3C Compositing L1.
    Returns a new PIL RGBA image. Both inputs must be the same size RGBA."""
    if not mode or mode == "Normal":
        return Image.alpha_composite(base_rgba, top_rgba)

    b = np.asarray(base_rgba, dtype=np.float32) / 255.0
    t = np.asarray(top_rgba, dtype=np.float32) / 255.0
    base_rgb = b[..., :3]
    top_rgb = t[..., :3]
    base_a = b[..., 3:4]
    top_a = t[..., 3:4]

    if mode == "Multiply":
        blended = base_rgb * top_rgb
    elif mode == "Screen":
        blended = 1 - (1 - base_rgb) * (1 - top_rgb)
    elif mode == "Overlay":
        blended = np.where(base_rgb < 0.5, 2 * base_rgb * top_rgb, 1 - 2 * (1 - base_rgb) * (1 - top_rgb))
    elif mode == "Darken":
        blended = np.minimum(base_rgb, top_rgb)
    elif mode == "Lighten":
        blended = np.maximum(base_rgb, top_rgb)
    elif mode == "Color Dodge":
        blended = np.where(top_rgb >= 1, 1.0, np.minimum(1.0, base_rgb / np.maximum(1 - top_rgb, 1e-6)))
    elif mode == "Color Burn":
        blended = np.where(top_rgb <= 0, 0.0, np.maximum(0.0, 1 - (1 - base_rgb) / np.maximum(top_rgb, 1e-6)))
    elif mode == "Hard Light":
        blended = np.where(top_rgb < 0.5, 2 * base_rgb * top_rgb, 1 - 2 * (1 - base_rgb) * (1 - top_rgb))
    elif mode == "Soft Light":
        gd = np.where(base_rgb < 0.25, ((16 * base_rgb - 12) * base_rgb + 4) * base_rgb, np.sqrt(np.maximum(base_rgb, 0)))
        blended = np.where(top_rgb < 0.5,
                           base_rgb - (1 - 2 * top_rgb) * base_rgb * (1 - base_rgb),
                           base_rgb + (2 * top_rgb - 1) * (gd - base_rgb))
    elif mode == "Difference":
        blended = np.abs(base_rgb - top_rgb)
    elif mode == "Exclusion":
        blended = base_rgb + top_rgb - 2 * base_rgb * top_rgb
    elif mode in ("Hue", "Saturation", "Color", "Luminosity"):
        base_hsl = _rgb_to_hsl(base_rgb)
        top_hsl = _rgb_to_hsl(top_rgb)
        if mode == "Hue":
            out_hsl = np.stack([top_hsl[..., 0], base_hsl[..., 1], base_hsl[..., 2]], axis=-1)
        elif mode == "Saturation":
            out_hsl = np.stack([base_hsl[..., 0], top_hsl[..., 1], base_hsl[..., 2]], axis=-1)
        elif mode == "Color":
            out_hsl = np.stack([top_hsl[..., 0], top_hsl[..., 1], base_hsl[..., 2]], axis=-1)
        else:  # Luminosity
            out_hsl = np.stack([base_hsl[..., 0], base_hsl[..., 1], top_hsl[..., 2]], axis=-1)
        blended = _hsl_to_rgb(out_hsl)
    else:
        blended = top_rgb

    # W3C compositing: source_color = (1 - αb) * Cs + αb * B(Cb, Cs)
    # Then standard source-over with αs.
    adjusted_top = (1 - base_a) * top_rgb + base_a * np.clip(blended, 0, 1)
    out_rgb = top_a * adjusted_top + (1 - top_a) * base_a * base_rgb
    out_a = top_a + (1 - top_a) * base_a
    safe_a = np.where(out_a > 0, out_a, 1)
    out_rgb = np.where(out_a > 0, out_rgb / safe_a, 0)
    out = np.concatenate([out_rgb, out_a], axis=-1)
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(out, "RGBA")


def _power_pin_homography(q0, q1, q2, q3):
    """Homography unit square -> quad (q0=TL, q1=TR, q2=BR, q3=BL): 8 values
    [h00 h01 h02 h10 h11 h12 h20 h21] with h22 = 1. None when degenerate.
    Mirrors power_pin.mjs buildHomography (cross-language parity)."""
    A = np.zeros((8, 8), dtype=np.float64)
    b = np.zeros(8, dtype=np.float64)
    pts = ((q0, 0.0, 0.0), (q1, 1.0, 0.0), (q2, 1.0, 1.0), (q3, 0.0, 1.0))
    for i, (q, sx, sy) in enumerate(pts):
        A[2 * i, 0] = sx
        A[2 * i, 1] = sy
        A[2 * i, 2] = 1.0
        A[2 * i, 6] = -q[0] * sx
        A[2 * i, 7] = -q[0] * sy
        A[2 * i + 1, 3] = sx
        A[2 * i + 1, 4] = sy
        A[2 * i + 1, 5] = 1.0
        A[2 * i + 1, 6] = -q[1] * sx
        A[2 * i + 1, 7] = -q[1] * sy
        b[2 * i] = q[0]
        b[2 * i + 1] = q[1]
    try:
        h = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return None
    if not np.all(np.isfinite(h)):
        return None
    a, bb, c, d, e, f, g, hh = h
    det = a * (e - f * hh) - bb * (d - f * g) + c * (d * hh - e * g)
    if not np.isfinite(det) or abs(det) < 1e-9:
        return None
    return h


def _power_pin_homography_inverse(h):
    """Analytic inverse of an 8-value homography (h22 = 1), re-normalized to
    h22 = 1. adj(H) with the 3x3 (h22=1) convention. None when singular."""
    a, bb, c, d, e, f, g, hh = h
    det = a * (e - f * hh) - bb * (d - f * g) + c * (d * hh - e * g)
    if not np.isfinite(det) or abs(det) < 1e-9:
        return None
    n22 = a * e - bb * d
    if abs(n22) < 1e-12:
        return None
    inv = 1.0 / n22
    return np.array([
        (e - f * hh) * inv, (c * hh - bb) * inv, (bb * f - c * e) * inv,
        (f * g - d) * inv, (a - c * g) * inv, (c * d - a * f) * inv,
        (d * hh - e * g) * inv, (bb * g - a * hh) * inv,
    ])


def _power_pin_coons(q, u, v):
    """Vectorized Coons (bilinear) quad map: (u, v) arrays -> (..., 2)."""
    q0x, q0y = q[0]
    q1x, q1y = q[1]
    q2x, q2y = q[2]
    q3x, q3y = q[3]
    tx = q0x + (q1x - q0x) * u
    bx = q3x + (q2x - q3x) * u
    ty = q0y + (q1y - q0y) * u
    by = q3y + (q2y - q3y) * u
    return np.stack((tx + (bx - tx) * v, ty + (by - ty) * v), axis=-1)


def _power_pin_newton_inverse(q, px, py, u0, v0):
    """Vectorized Newton iteration for the bilinear-quad inverse (spec 3.4).
    Starts from the projective preimage; 8 iterations, per-pixel freeze on
    convergence (mirrors the JS break)."""
    e = 1e-4
    bu = u0.copy()
    bv = v0.copy()
    for _ in range(8):
        f = _power_pin_coons(q, bu, bv)
        rx = f[..., 0] - px
        ry = f[..., 1] - py
        fa = _power_pin_coons(q, bu + e, bv)
        fc = _power_pin_coons(q, bu, bv + e)
        j00 = (fa[..., 0] - f[..., 0]) / e
        j10 = (fa[..., 1] - f[..., 1]) / e
        j01 = (fc[..., 0] - f[..., 0]) / e
        j11 = (fc[..., 1] - f[..., 1]) / e
        det = j00 * j11 - j01 * j10
        with np.errstate(divide="ignore", invalid="ignore"):
            du = (j11 * rx - j01 * ry) / det
            dv = (-j10 * rx + j00 * ry) / det
        adv = (np.abs(rx) + np.abs(ry) >= 1e-6) & (np.abs(det) > 1e-12)
        bu = np.where(adv, bu - du, bu)
        bv = np.where(adv, bv - dv, bv)
    return bu, bv


def _power_pin_setup(pp, W, H):
    """Resolve a layer's powerPin into a warp setup (spec 3). None when
    absent or degenerate (collinear pins, zero-size expanded rect, NaN)."""
    if not isinstance(pp, dict):
        return None
    pins = pp.get("pins")
    if not isinstance(pins, (list, tuple)) or len(pins) != 4:
        return None
    try:
        q = [[float(pt[0]), float(pt[1])] for pt in pins]
        p = min(1.0, max(0.0, float(pp.get("perspective", 100)) / 100.0))
        eT = float(pp.get("expandTop", 0)) / 100.0
        eL = float(pp.get("expandLeft", 0)) / 100.0
        eR = float(pp.get("expandRight", 0)) / 100.0
        eB = float(pp.get("expandBottom", 0)) / 100.0
    except (TypeError, ValueError, IndexError):
        return None
    if any(not (math.isfinite(px) and math.isfinite(py)) for px, py in q):
        return None
    unstretch = bool(pp.get("unstretch", False))
    x0, y0 = -eL * W, -eT * H
    x1, y1 = W * (1.0 + eR), H * (1.0 + eB)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return None
    h = _power_pin_homography(q[0], q[1], q[2], q[3])
    if h is None:
        return None
    ox, oy, ow, oh = 0, 0, W, H
    inv = None
    if not unstretch:
        xs = [pt[0] for pt in q]
        ys = [pt[1] for pt in q]
        ox = int(math.floor(min(xs)))
        oy = int(math.floor(min(ys)))
        ow = max(1, int(math.ceil(max(xs))) - ox)
        oh = max(1, int(math.ceil(max(ys))) - oy)
        inv = _power_pin_homography_inverse(h)
        if inv is None:
            return None
    return {"q": q, "p": p, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "unstretch": unstretch, "ox": ox, "oy": oy, "ow": ow, "oh": oh,
            "h": h, "inv": inv, "W": W, "H": H}


def _apply_power_pin(img, pp):
    """CC Power Pin warp (spec 3). img = post-crop, post-mask PIL RGBA.
    Returns (warped_RGBA, ox, oy, ow, oh) or None when pp is absent / the quad
    is degenerate (caller renders the layer unpinned). Vectorized, row-tiled
    (inverse-map + gather needs no halo; tiles bound peak memory only)."""
    setup = _power_pin_setup(pp, img.width, img.height) if isinstance(pp, dict) else None
    if setup is None:
        return None
    W, H = img.width, img.height
    q, p = setup["q"], setup["p"]
    x0, y0, x1, y1 = setup["x0"], setup["y0"], setup["x1"], setup["y1"]
    unstretch, ox, oy, ow, oh = setup["unstretch"], setup["ox"], setup["oy"], setup["ow"], setup["oh"]
    src = np.asarray(img.convert("RGBA"), dtype=np.float32)  # (H, W, 4)
    out = np.zeros((oh, ow, 4), dtype=np.float32)
    xs = (np.arange(ow) + 0.5) + ox
    TILE = 256
    for ty0 in range(0, oh, TILE):
        ty1 = min(oh, ty0 + TILE)
        PX, PY = np.meshgrid(xs, (np.arange(ty0, ty1) + 0.5) + oy)
        if not unstretch:
            h = setup["inv"]
            w = h[6] * PX + h[7] * PY + 1.0
            valid = np.abs(w) > 1e-12
            with np.errstate(divide="ignore", invalid="ignore"):
                u = (h[0] * PX + h[1] * PY + h[2]) / w
                v = (h[3] * PX + h[4] * PY + h[5]) / w
            u = np.where(valid, u, 2.0)
            v = np.where(valid, v, 2.0)
            if p < 1.0:
                bu, bv = _power_pin_newton_inverse(q, PX, PY, u, v)
                u = bu + (u - bu) * p
                v = bv + (v - bv) * p
            inside = valid & (u >= -1e-9) & (u <= 1.0 + 1e-9) & (v >= -1e-9) & (v <= 1.0 + 1e-9)
            u = np.clip(np.where(inside, u, 0.0), 0.0, 1.0)
            v = np.clip(np.where(inside, v, 0.0), 0.0, 1.0)
            LX = x0 + u * (x1 - x0)
            LY = y0 + v * (y1 - y0)
        else:
            u = (PX - x0) / (x1 - x0)
            v = (PY - y0) / (y1 - y0)
            inside = (u >= -1e-9) & (u <= 1.0 + 1e-9) & (v >= -1e-9) & (v <= 1.0 + 1e-9)
            u = np.clip(np.where(inside, u, 0.0), 0.0, 1.0)
            v = np.clip(np.where(inside, v, 0.0), 0.0, 1.0)
            b = _power_pin_coons(q, u, v)
            h = setup["h"]
            w = h[6] * u + h[7] * v + 1.0
            with np.errstate(divide="ignore", invalid="ignore"):
                mx = (h[0] * u + h[1] * v + h[2]) / w
                my = (h[3] * u + h[4] * v + h[5]) / w
            okp = np.abs(w) > 1e-12
            LX = b[..., 0] + np.where(okp, (mx - b[..., 0]) * p, 0.0)
            LY = b[..., 1] + np.where(okp, (my - b[..., 1]) * p, 0.0)
            inside &= okp
        I0 = np.floor(LX - 0.5).astype(np.int32)
        J0 = np.floor(LY - 0.5).astype(np.int32)
        FX = np.clip(LX - (I0 + 0.5), 0.0, 1.0)
        FY = np.clip(LY - (J0 + 0.5), 0.0, 1.0)
        # Transparent padding (mirrors the JS neighbor guards): a neighbor
        # outside the source contributes 0, so exact pixel centers (identity)
        # sample pixel-exact and the source edge fades over the half pixel.
        good = inside & (I0 >= -1) & (J0 >= -1) & (I0 < W) & (J0 < H)
        I0g = np.clip(I0, 0, W - 1).astype(np.intp)
        J0g = np.clip(J0, 0, H - 1).astype(np.intp)
        I1g = np.clip(I0 + 1, 0, W - 1).astype(np.intp)
        J1g = np.clip(J0 + 1, 0, H - 1).astype(np.intp)
        c00 = np.where(((I0 >= 0) & (J0 >= 0) & good)[..., None], src[J0g, I0g], 0.0)
        c10 = np.where(((I0 < W - 1) & (J0 >= 0) & good)[..., None], src[J0g, I1g], 0.0)
        c01 = np.where(((I0 >= 0) & (J0 < H - 1) & good)[..., None], src[J1g, I0g], 0.0)
        c11 = np.where(((I0 < W - 1) & (J0 < H - 1) & good)[..., None], src[J1g, I1g], 0.0)
        FXe = FX[..., None]
        FYe = FY[..., None]
        samp = (c00 * (1.0 - FXe) + c10 * FXe) * (1.0 - FYe) + (c01 * (1.0 - FXe) + c11 * FXe) * FYe
        out[ty0:ty1] = np.where(good[..., None], samp, 0.0)
    return Image.fromarray(np.rint(out).astype(np.uint8), "RGBA"), ox, oy, ow, oh


def _apply_layer_transform(img, layer, doc_w, doc_h):
    """Apply layer transforms and return a doc-sized RGBA canvas."""
    nat_w, nat_h = img.size
    scale_x = abs(layer.get("scaleX", 1.0))
    scale_y = abs(layer.get("scaleY", 1.0))
    new_w = max(1, int(nat_w * scale_x))
    new_h = max(1, int(nat_h * scale_y))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    if layer.get("flippedX"):
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if layer.get("flippedY"):
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    rotation = layer.get("rotation", 0)
    if rotation:
        img = img.rotate(-rotation, expand=True, resample=Image.BICUBIC)

    opacity = layer.get("opacity", 1.0)
    if opacity < 1.0:
        r, g, b, a = img.split()
        a = a.point(lambda x: int(x * opacity))
        img = Image.merge("RGBA", (r, g, b, a))

    cx = layer.get("cx", doc_w / 2)
    cy = layer.get("cy", doc_h / 2)
    paste_x = int(cx - img.width / 2)
    paste_y = int(cy - img.height / 2)

    canvas = Image.new("RGBA", (doc_w, doc_h), (0, 0, 0, 0))
    canvas.paste(img, (paste_x, paste_y), img)
    return canvas
def _power_pin_center(layer, W, H, ox, oy, ow, oh):
    """Doc-space placement center for a warped layer (spec 3.8): C' = C + A(d),
    d = (ox - W/2 + ow/2, oy - H/2 + oh/2); A = the same scale-flip-rotate
    affine _apply_layer_transform applies (its PIL rotate(-rotation) equals
    the canvas rotate(rotation) screen math). Identity pin -> (cx, cy) exact."""
    dx = ox - W / 2.0 + ow / 2.0
    dy = oy - H / 2.0 + oh / 2.0
    x = dx * abs(float(layer.get("scaleX", 1.0) or 1.0))
    y = dy * abs(float(layer.get("scaleY", 1.0) or 1.0))
    if layer.get("flippedX"):
        x = -x
    if layer.get("flippedY"):
        y = -y
    rad = math.radians(float(layer.get("rotation", 0) or 0))
    cx = float(layer.get("cx", 0.0) or 0.0)
    cy = float(layer.get("cy", 0.0) or 0.0)
    return cx + x * math.cos(rad) - y * math.sin(rad), cy + x * math.sin(rad) + y * math.cos(rad)

class PixaromaImageComposition:
    DESCRIPTION = (
        "Image Composer Pixaroma - combine and arrange multiple images on a "
        "layered canvas. Drop an image onto the closed node body to add it as "
        "a new layer (opens the editor and stacks the dropped image on top of "
        "any existing layers, never replacing). Inside the fullscreen editor "
        "you can also add layers via drag-drop, paste from clipboard, or wire "
        "upstream IMAGE inputs into placeholder slots, then move / scale / "
        "rotate them with handles, and apply per-layer effects: opacity, blend "
        "modes (Normal, Multiply, Screen, Overlay, Darken, Lighten, Color "
        "Dodge, Color Burn, Hard Light, Soft Light, Difference, Exclusion, "
        "Hue, Saturation, Color, Luminosity), Gaussian blur, CC Power Pin (4-corner warp with perspective / expand / unstretch, [P]), and AI background "
        "removal (rembg-powered, picks the best installed model: birefnet -> "
        "isnet -> u2net auto-fallback).\n\n"
        "Eraser tool lets you mask any layer non-destructively. Per-layer blur "
        "uses a quadratic curve (slider 0-100 -> 0-50 px) so low values give "
        "fine control. Shift+Scroll on a selected layer scales it in place.\n\n"
        "Outputs the final composite as an IMAGE plus its width and height.\n\n"
        "Inputs are dynamic placeholder slots created by the editor when you "
        "add a placeholder layer - wire any upstream IMAGE source into the "
        "matching slot. Their tooltips live on the JS side."
    )

    @classmethod
    def INPUT_TYPES(self):
        return {
            "required": {},
            "optional": FlexibleOptionalInputType(any_type),
        }

    # ADDED: New Width and Height INT outputs!
    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("image", "width", "height")
    OUTPUT_TOOLTIPS = ("The composited image.", "Image width in pixels.", "Image height in pixels.")
    FUNCTION = "load_composite"
    CATEGORY = "👑 Pixaroma/🎨 Editors"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when the composite file on disk changes."""
        composition_data = kwargs.get("ComposerWidget")
        if not composition_data:
            return ""
        try:
            project_json = composition_data.get("project_json", "{}") if isinstance(composition_data, dict) else str(composition_data)
            meta = json.loads(project_json)
            composite_path = meta.get("composite_path", "")
            if composite_path:
                # safe_join, not os.path.join - see _path_guard.safe_join.
                # load_composite / _load_server_image were already guarded;
                # this IS_CHANGED twin was not.
                full_path = safe_join(folder_paths.get_input_directory(), composite_path)
                if full_path and os.path.exists(full_path):
                    return os.path.getmtime(full_path)
        except Exception:
            pass
        return str(composition_data)

    def load_composite(self, **kwargs):
        empty_image = torch.zeros((1, 1024, 1024, 3), dtype=torch.float32)

        composition_data = kwargs.get("ComposerWidget")
        if not composition_data:
            return (empty_image, 1024, 1024)

        project_json = composition_data.get("project_json", "{}") if isinstance(composition_data, dict) else str(composition_data)

        if not project_json or project_json == "" or project_json == "{}":
            return (empty_image, 1024, 1024)

        try:
            meta = json.loads(project_json)
            if not isinstance(meta, dict):
                return (empty_image, 1024, 1024)

            # Grab actual dimensions from JSON
            doc_w = int(meta.get("doc_w", 1024))
            doc_h = int(meta.get("doc_h", 1024))
            # A blank fallback sized to the ACTUAL document, so the returned
            # tensor shape always matches the reported width/height (a
            # hardcoded 1024x1024 here would mismatch doc_w/doc_h and break
            # downstream nodes that trust the INT outputs).
            doc_empty = torch.zeros((1, doc_h, doc_w, 3), dtype=torch.float32)

            layers = meta.get("layers", [])
            if not isinstance(layers, list):
                layers = []
            input_dir = os.path.realpath(folder_paths.get_input_directory())
            has_placeholders = any(l.get("isPlaceholder") for l in layers)
            has_auto_rembg = any(l.get("removeBgOnExec") for l in layers)

            has_masks = any(l.get("maskSrc") for l in layers)

            if has_placeholders or has_auto_rembg or has_masks:
                # Composite from scratch so placeholder slots are filled, rembg and masks applied.
                # Initialize with the user's saved BG color (project_json field
                # `bg_color`, added in 1.3.18). Without this the canvas was always
                # transparent here, and the eventual RGBA->RGB conversion in
                # _save_preview_png + the workflow output silently turned the BG
                # black even when the user had chosen a colour in the editor.
                # Older saves (pre-1.3.18) had no bg_color field — fall back to
                # the same default the editor's BG picker uses (#1e1e1e) so
                # output matches what users saw inside the editor.
                bg_hex = meta.get("bg_color") or "#1e1e1e"
                bg_rgba = _hex_to_rgba(bg_hex)
                canvas = Image.new("RGBA", (doc_w, doc_h), bg_rgba)
                for layer in layers:
                    if not layer.get("visible", True):
                        continue
                    # FX adjustment layer: grade everything composited so far
                    # (canvas = every layer below this one), then move on. The
                    # static fast path bakes the grade at save time; this branch
                    # re-applies it on the dynamic path (placeholder/rembg/mask)
                    # where live pixels differ from the baked composite.
                    if layer.get("isAdjustment"):
                        amount01 = float(layer.get("opacity", 1) or 0)
                        adj = layer.get("adjustments") or {}
                        if is_neutral(adj, amount01):
                            continue
                        seed = _fx_seed(layer.get("id", ""))
                        rgb = np.asarray(canvas.convert("RGB"), dtype=np.float32) / 255.0
                        graded = apply_fx(rgb, adj, amount01, seed)
                        graded_img = Image.fromarray(
                            (graded * 255.0).round().astype(np.uint8), "RGB"
                        ).convert("RGBA")
                        graded_img.putalpha(canvas.getchannel("A"))
                        canvas = graded_img
                        continue
                    if layer.get("isPlaceholder"):
                        ph_w = layer.get("naturalWidth", 512)
                        ph_h = layer.get("naturalHeight", 512)
                        img_input = kwargs.get(f"image_{layer.get('inputIndex', 1)}")
                        if img_input is not None:
                            layer_img = _tensor_to_pil(img_input)
                            if layer.get("removeBgOnExec"):
                                layer_img = _remove_background(layer_img, layer.get("bgRemovalQuality", "normal"))
                            # Quality fix: scale the slot up to source resolution
                            # if the upstream image is larger. Adjust scaleX/Y in
                            # a layer copy so the visual size matches the editor.
                            # Mirrors js/composer/placeholder.mjs previewPlaceholderInput.
                            sw, sh = layer_img.size
                            upscale = max(1.0, sw / ph_w, sh / ph_h)
                            if upscale > 1.0:
                                new_ph_w = int(round(ph_w * upscale))
                                new_ph_h = int(round(ph_h * upscale))
                                compensation = ph_w / new_ph_w
                                ph_w, ph_h = new_ph_w, new_ph_h
                                layer = {
                                    **layer,
                                    "scaleX": layer.get("scaleX", 1.0) * compensation,
                                    "scaleY": layer.get("scaleY", 1.0) * compensation,
                                }
                            layer_img = _fit_to_placeholder(layer_img, ph_w, ph_h, layer.get("fillMode", "cover"))
                        else:
                            color = _hex_to_rgba(layer.get("placeholderColor", "#808080"))
                            layer_img = Image.new("RGBA", (ph_w, ph_h), color)
                    else:
                        src = layer.get("src") or ""
                        if not src or src == "__placeholder__":
                            continue
                        layer_img = _load_server_image(src, input_dir)
                        if layer_img is None:
                            continue
                        if layer.get("removeBgOnExec"):
                            layer_img = _remove_background(layer_img, layer.get("bgRemovalQuality", "normal"))
                    # Apply non-destructive crop (cropRect is in source-image
                    # pixels). Order matches the JS bake: crop -> mask -> transform.
                    # The eraser mask is saved at the cropped size, so it must run
                    # AFTER the crop. (Crop-only layers without a placeholder/mask
                    # take the fast path and never reach here - their crop is
                    # already baked into the saved composite PNG.)
                    crop = layer.get("cropRect")
                    if crop and layer_img is not None:
                        try:
                            cx0 = max(0, int(crop["x"]))
                            cy0 = max(0, int(crop["y"]))
                            cx1 = min(layer_img.width, int(crop["x"]) + int(crop["w"]))
                            cy1 = min(layer_img.height, int(crop["y"]) + int(crop["h"]))
                            if cx1 > cx0 and cy1 > cy0:
                                layer_img = layer_img.crop((cx0, cy0, cx1, cy1))
                        except (KeyError, TypeError, ValueError):
                            pass
                    # Apply eraser mask if present (mask white = erased)
                    mask_src = layer.get("maskSrc")
                    if mask_src:
                        layer_img = _apply_eraser_mask(layer_img, mask_src, input_dir)
                    # Transform + blur + blend are the only unguarded throw-risk
                    # left in the loop (earlier branches already handle their own
                    # failures). Isolate them so one malformed layer is skipped
                    # instead of failing the whole composite.
                    try:
                        # CC Power Pin warp (post-crop, post-mask image
                        # pixels). Placement is re-anchored to C' so enabling
                        # or dragging pins never jumps the layer. Degenerate
                        # quad -> layer renders unpinned (one notice).
                        pp = layer.get("powerPin")
                        if pp:
                            nat_w, nat_h = layer_img.size
                            res = _apply_power_pin(layer_img, pp)
                            if res is None:
                                print("[Pixaroma] Composer: degenerate power-pin quad, layer rendered unpinned")
                            else:
                                layer_img, pox, poy, pow_, poh = res
                                cx2, cy2 = _power_pin_center(layer, nat_w, nat_h, pox, poy, pow_, poh)
                                layer = {**layer, "cx": cx2, "cy": cy2}
                        composed = _apply_layer_transform(layer_img, layer, doc_w, doc_h)
                        # Quadratic curve: slider 0-100 maps to actual blur 0-50px
                        # (finer control at low end). Mirrored in
                        # js/composer/render.mjs + js/composer/index.js.
                        blur_slider = layer.get("blur", 0)
                        if blur_slider and blur_slider > 0:
                            blur_radius = (blur_slider / 100.0) ** 2 * 50.0
                            composed = composed.filter(ImageFilter.GaussianBlur(radius=blur_radius))
                        canvas = _blend_over(canvas, composed, layer.get("blendMode", "Normal"))
                    except Exception as _le:
                        print(f"[Pixaroma] Composer: skipped a layer that failed to render: {_le}")
                        continue
                # Save the final composed image to temp so the node's
                # mini preview gets the exact executed result (including
                # auto-rembg / mask application). Without this, the JS
                # `rebuildPreview` re-composes the raw placeholder
                # images client-side and the mini preview looks nothing
                # like the Preview Image output downstream.
                #
                # IMPORTANT: we deliberately DON'T return {"ui":
                # {"images": [...]}} here — ComfyUI would then render
                # that image as a secondary preview panel below the
                # node widgets, on top of our custom top preview.
                # Instead we push a private WebSocket event that our
                # index.js listener picks up and applies to the top
                # preview only.
                preview_img = _save_preview_png(canvas, doc_w, doc_h)
                if preview_img:
                    try:
                        PromptServer.instance.send_sync(
                            "pixaroma-composer-preview",
                            {
                                "project_id": meta.get("project_id"),
                                "filename": preview_img["filename"],
                                "subfolder": preview_img["subfolder"],
                                "type": preview_img["type"],
                                "doc_w": doc_w,
                                "doc_h": doc_h,
                            },
                        )
                    except Exception as e:
                        print(f"[Pixaroma] preview WS send failed: {e}")
                return (_pil_to_tensor(canvas), doc_w, doc_h)

            # Fast path: load the pre-rendered composite PNG
            composite_path = meta.get("composite_path")
            if not composite_path:
                return (doc_empty, doc_w, doc_h)

            full_path = os.path.realpath(os.path.join(input_dir, composite_path))

            # Security: block path traversal — path must stay inside input_dir
            if not full_path.startswith(input_dir + os.sep):
                print(
                    "[Pixaroma] Security: composite_path escapes input directory, blocked."
                )
                return (doc_empty, doc_w, doc_h)

            if not os.path.exists(full_path):
                print(
                    f"[Pixaroma] Composite image not found (re-open the Image "
                    f"Composer and Save to regenerate it): {composite_path}"
                )
                return (doc_empty, doc_w, doc_h)

            img = Image.open(full_path).convert("RGB")
            img = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img)[None,]

            # Return Image + Dimensions
            return (img_tensor, doc_w, doc_h)

        except Exception as e:
            print(f"[Pixaroma] Fatal Load Error: {e}")
            # Best-effort: return a blank sized to the DOCUMENT if we can recover
            # its dims, so downstream nodes get the right width/height even when
            # compositing failed (a hardcoded 1024x1024 here mismatches doc_w/h
            # and breaks nodes that trust the INT outputs).
            try:
                _m = json.loads(project_json)
                _dw, _dh = int(_m.get("doc_w", 1024)), int(_m.get("doc_h", 1024))
                return (torch.zeros((1, _dh, _dw, 3), dtype=torch.float32), _dw, _dh)
            except Exception:
                return (empty_image, 1024, 1024)


NODE_CLASS_MAPPINGS = {
    "PixaromaImageComposition": PixaromaImageComposition,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixaromaImageComposition": "Image Composer Pixaroma",
}

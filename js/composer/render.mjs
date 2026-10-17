// Rendering, history/undo, restore — mixed into PixaromaEditor.prototype
import { PixaromaEditor } from "./core.mjs";
import { pixApiUrl } from "../shared/api_url.mjs";
import { applyFx, isNeutral, fxSeed, resolveClipTargets } from "./fx_engine.mjs";
import {
  powerPinSetup,
  pinnedPointsToDoc,
  powerPinAnchorShift,
  warpPowerPinCanvas,
  warpPowerPinCanvasAA,
} from "./power_pin.mjs";
import {
  featherRadiusPx,
  edgeBlurRadiusPx,
  borderFadeCanvas,
} from "./feather.mjs";

// Clone a layer for history restore. Shallow-copy the layer, but DEEP-copy the
// mutable sub-objects the panels edit IN PLACE (adjustments / textState /
// cropRect) so a live edit right after undo/redo can't corrupt the stored
// snapshot (the snapshots themselves are already deep via captureState).
function cloneLayerForHistory(l) {
  const c = { ...l };
  if (l.adjustments) c.adjustments = { ...l.adjustments };
  if (l.textState) c.textState = { ...l.textState };
  if (l.cropRect) c.cropRect = { ...l.cropRect };
  if (l.powerPin)
    c.powerPin = { ...l.powerPin, pins: l.powerPin.pins.map((pt) => [...pt]) };
  // Deep-copy the eraser mask canvas too: a shallow {...l} shares the canvas
  // with the history snapshot, so strokes drawn after an undo/redo would bleed
  // into the stored snapshot and corrupt history.
  if (l.eraserMaskCanvas_internal) {
    const cvs = document.createElement("canvas");
    cvs.width = l.eraserMaskCanvas_internal.width;
    cvs.height = l.eraserMaskCanvas_internal.height;
    const cctx = cvs.getContext("2d");
    cctx.drawImage(l.eraserMaskCanvas_internal, 0, 0);
    c.eraserMaskCanvas_internal = cvs;
    c.eraserMaskCtx_internal = cctx;
  }
  return c;
}

PixaromaEditor.prototype.pushHistory = function () {
  if (this.isRestoringHistory) return;
  this.history = this.history.slice(0, this.historyIndex + 1);
  this.history.push(this.captureState());
  this.historyIndex++;
  // Cap history to prevent unbounded memory growth
  const MAX_HISTORY = 50;
  if (this.history.length > MAX_HISTORY) {
    const excess = this.history.length - MAX_HISTORY;
    this.history = this.history.slice(excess);
    this.historyIndex -= excess;
  }
  this.ui.updateHistoryUI();
};

PixaromaEditor.prototype.undo = function () {
  // Ctrl+Z while cropping CANCELS the in-progress crop (discards the draft) and
  // consumes the keystroke - it does NOT also step back history. Otherwise the
  // layer you were about to crop gets removed (the step before "add layer").
  // Press Ctrl+Z again (now out of crop mode) to undo for real.
  if (this.activeMode === "crop") {
    this._clearCropState();
    this.draw();
    return;
  }
  if (this.historyIndex > 0) {
    this.historyIndex--;
    this.layers = this.history[this.historyIndex].map((l) =>
      cloneLayerForHistory(l),
    );
    this.verifySelection();
    this.isRestoringHistory = true;
    this.ui.updateActiveLayerUI();
    this.draw();
    this.isRestoringHistory = false;
    this.ui.updateHistoryUI();
  }
};

PixaromaEditor.prototype.redo = function () {
  // Mirror undo: a redo while cropping just cancels the crop draft.
  if (this.activeMode === "crop") {
    this._clearCropState();
    this.draw();
    return;
  }
  if (this.historyIndex < this.history.length - 1) {
    this.historyIndex++;
    this.layers = this.history[this.historyIndex].map((l) =>
      cloneLayerForHistory(l),
    );
    this.verifySelection();
    this.isRestoringHistory = true;
    this.ui.updateActiveLayerUI();
    this.draw();
    this.isRestoringHistory = false;
    this.ui.updateHistoryUI();
  }
};

// Blend mode map — defined once, reused every frame
const BLEND_MAP = {
  Normal: "source-over",
  Multiply: "multiply",
  Screen: "screen",
  Overlay: "overlay",
  Darken: "darken",
  Lighten: "lighten",
  "Color Dodge": "color-dodge",
  "Color Burn": "color-burn",
  "Hard Light": "hard-light",
  "Soft Light": "soft-light",
  Difference: "difference",
  Exclusion: "exclusion",
  Hue: "hue",
  Saturation: "saturation",
  Color: "color",
  Luminosity: "luminosity",
};

// draw() batches calls via rAF so multiple per frame collapse into one render.
// Pass cleanRender=true (save) for synchronous rendering.
PixaromaEditor.prototype.draw = function (cleanRender = false) {
  if (cleanRender) {
    this._drawImpl(true);
    return;
  }
  if (this._drawPending) return;
  this._drawPending = true;
  requestAnimationFrame(() => {
    this._drawPending = false;
    this._drawImpl(false);
    // If eraser preview was pending, draw it after the main render
    if (this._pendingEraserPreview) {
      this.drawEraserPreview(this._pendingEraserPreview);
      this._pendingEraserPreview = null;
    }
  });
};

// Lazily grow the selection overlay canvas so heavily-scaled or off-canvas
// layers don't have their bounding boxes clipped. Grow only — never shrink —
// to avoid resize churn during slider drag. Capped to keep peak memory sane.
PixaromaEditor.prototype._ensureSelPad = function () {
  const MIN_PAD = 500;
  const MAX_PAD = 4000; // ~ (4K + 8K)² × 4B ≈ 600 MB worst case; layer clips beyond this
  const HANDLE_BUFFER = 50; // room for corner handles + 30 px rotation handle stem
  let needed = MIN_PAD;
  for (const layer of this.layers) {
    if (!layer.visible || !this.selectedLayerIds.has(layer.id)) continue;
    if (!layer.img) continue;
    const W = layer.img.width, H = layer.img.height;
    let minX, maxX, minY, maxY;
    if (layer.powerPin && powerPinSetup(layer.powerPin, W, H)) {
      const pts = pinnedPointsToDoc(layer, W, H);
      minX = Math.min(...pts.map((p) => p.x));
      maxX = Math.max(...pts.map((p) => p.x));
      minY = Math.min(...pts.map((p) => p.y));
      maxY = Math.max(...pts.map((p) => p.y));
    } else {
      const w = W * Math.abs(layer.scaleX || 1);
      const h = H * Math.abs(layer.scaleY || 1);
      const r = ((layer.rotation || 0) * Math.PI) / 180;
      const cos = Math.abs(Math.cos(r));
      const sin = Math.abs(Math.sin(r));
      const aabbW = w * cos + h * sin;
      const aabbH = w * sin + h * cos;
      const cx = layer.cx ?? this.docWidth / 2;
      const cy = layer.cy ?? this.docHeight / 2;
      minX = cx - aabbW / 2;
      maxX = cx + aabbW / 2;
      minY = cy - aabbH / 2;
      maxY = cy + aabbH / 2;
    }
    needed = Math.max(
      needed,
      -minX + HANDLE_BUFFER,
      maxX - this.docWidth + HANDLE_BUFFER,
      -minY + HANDLE_BUFFER,
      maxY - this.docHeight + HANDLE_BUFFER,
    );
  }
  needed = Math.min(MAX_PAD, Math.ceil(needed));
  if (needed > this.selPad) {
    this.selPad = needed;
    this.selCanvas.width = this.docWidth + 2 * needed;
    this.selCanvas.height = this.docHeight + 2 * needed;
    this.selCanvas.style.left = -needed + "px";
    this.selCanvas.style.top = -needed + "px";
    if (this.selHitArea) {
      this.selHitArea.style.width = this.docWidth + 2 * needed + "px";
      this.selHitArea.style.height = this.docHeight + 2 * needed + "px";
      this.selHitArea.style.left = -needed + "px";
      this.selHitArea.style.top = -needed + "px";
    }
  }
};

PixaromaEditor.prototype._drawImpl = function (cleanRender) {
  if (this._transparentExport) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  } else {
    const bg = this._bgColor || "#1e1e1e";
    this.ctx.fillStyle = bg;
    this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
  }

  this.ctx.imageSmoothingEnabled = true;
  this.ctx.imageSmoothingQuality = "high";

  // Grow the selection overlay if needed BEFORE we clear it (resize wipes it
  // anyway, but doing it first keeps render order obvious).
  if (!cleanRender && this.selCanvas) this._ensureSelPad();

  // Clear selection overlay
  const oc = this.selCtx;
  const pad = this.selPad || 0;
  if (oc && !cleanRender) {
    oc.clearRect(0, 0, this.selCanvas.width, this.selCanvas.height);
  }

  // Clip-to-layer: a clipped FX adjustment grades only the nearest visible
  // non-adjustment layer below it. Precompute once; the loop applies the refs
  // to the target layer's own doc-sized pixels (offscreen, source-over) and
  // composites the result with the layer's own blend mode. Unclipped
  // adjustments keep the whole-canvas grade below. Mirrors
  // node_composition.py + index.js compositeAll.
  const clipFx = resolveClipTargets(this.layers);
  this.layers.forEach((layer, li) => {
    if (!layer.visible) return;
    // FX adjustment layer: grade everything composited so far (the pixels
    // already on the canvas = every layer below this one), then move on. Runs
    // for BOTH interactive and clean/save renders so the grade is baked into
    // the saved composite PNG (which keeps the Python fast path correct).
    if (layer.isAdjustment) {
      if (clipFx.consumed.has(li)) return; // already graded its target layer
      const amount01 = layer.opacity ?? 1;
      if (isNeutral(layer.adjustments, amount01)) return;
      const W = this.canvas.width, H = this.canvas.height;
      const id = this.ctx.getImageData(0, 0, W, H);
      applyFx(id.data, W, H, layer.adjustments, amount01, fxSeed(layer.id));
      this.ctx.putImageData(id, 0, 0);
      return;
    }
    // Defensive: layer.img can briefly be null during async upload / restore.
    // _ensureSelPad already skips this case; mirror that here.
    if (!layer.img) return;
    // In crop mode the active layer is drawn by drawCropOverlay (full source +
    // box), so skip it in the normal loop to avoid double-drawing. Only for the
    // interactive render — a clean render (save composite) must draw it normally
    // so the crop UI never bakes into the saved PNG.
    if (
      !cleanRender &&
      this.activeMode === "crop" &&
      this._cropLayer &&
      layer.id === this._cropLayer.id
    )
      return;

    const isSelected = this.selectedLayerIds.has(layer.id);
    // Clip-to-layer target: draw the layer's pixels onto a doc-sized offscreen
    // (source-over, opacity baked in), grade them with its clipped
    // adjustments, then composite with the layer's own blend mode + opacity.
    const clipRefs = (clipFx.byTarget.get(li) || []).filter(
      (r) => !isNeutral(r.adjustments, r.amount01),
    );
    if (clipRefs.length) {
      if (
        !this._fxClipCanvas ||
        this._fxClipCanvas.width !== this.canvas.width ||
        this._fxClipCanvas.height !== this.canvas.height
      ) {
        this._fxClipCanvas = document.createElement("canvas");
        this._fxClipCanvas.width = this.canvas.width;
        this._fxClipCanvas.height = this.canvas.height;
        this._fxClipCtx = this._fxClipCanvas.getContext("2d", {
          willReadFrequently: true,
        });
      }
      this._fxClipCtx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    }
    const layerCtx = clipRefs.length ? this._fxClipCtx : this.ctx;
    layerCtx.save();
    layerCtx.globalAlpha = layer.opacity;
    // Blend mode support (maps to canvas globalCompositeOperation). The clip
    // path keeps source-over here — the real blend mode is applied by the
    // final drawImage of the graded offscreen onto the main canvas.
    if (!clipRefs.length && layer.blendMode && BLEND_MAP[layer.blendMode])
      layerCtx.globalCompositeOperation = BLEND_MAP[layer.blendMode];

    // CC Power Pin: the placement frame is the pin-quad AABB, re-anchored at
    // C' so the layer never jumps on enable/drag. Degenerate pin -> normal
    // (unpinned) draw.
    const Wimg = layer.img.width, Him = layer.img.height;
    const ppSetup = layer.powerPin
      ? powerPinSetup(layer.powerPin, Wimg, Him)
      : null;
    // Snap to integer pixels so strokeRect (selection box) and drawImage land on
    // the same pixel grid — otherwise sub-pixel anti-aliasing makes the box look
    // 1-2px larger than the image on bottom/right edges
    let w = Math.round(Wimg * layer.scaleX);
    let h = Math.round(Him * layer.scaleY);
    if (ppSetup) {
      w = Math.round(ppSetup.ow * layer.scaleX);
      h = Math.round(ppSetup.oh * layer.scaleY);
    }
    const ppShift = ppSetup
      ? powerPinAnchorShift(layer, ppSetup, Wimg, Him)
      : null;

    layerCtx.translate(
      layer.cx + (ppShift ? ppShift.x : 0),
      layer.cy + (ppShift ? ppShift.y : 0),
    );
    layerCtx.rotate((layer.rotation * Math.PI) / 180);

    layerCtx.save();
    layerCtx.scale(layer.flippedX ? -1 : 1, layer.flippedY ? -1 : 1);

    // Border fade (feather + edge blur) in display space — post scale/flip,
    // pre rotation/paste, matching _apply_layer_transform's pre_display hook.
    // Treated canvas is cached; fast slider input renders at <=512px.
    const fadeF = featherRadiusPx(layer.feather);
    const fadeR = edgeBlurRadiusPx(layer.edgeBlur);
    const drawDisplay = (src) => {
      const reduced = !cleanRender &&
        performance.now() - (this._lastFadeInput || 0) < 150;
      const treated =
        fadeF > 0 || fadeR > 0
          ? this._borderFadeCanvas(layer, src, w, h, reduced)
          : null;
      // layerCtx, not this.ctx: the clip-to-layer path draws the layer onto
      // the doc-sized offscreen first (graded, then composited below).
      layerCtx.drawImage(treated || src, -w / 2, -h / 2, w, h);
    };

    // Per-layer Gaussian blur — non-destructive, applied to the final composited
    // (post-mask) result. Selection box stroke is on the overlay canvas (oc),
    // so it stays sharp regardless of this filter.
    // Slider value (0-100) is mapped via a quadratic curve to actual blur radius
    // (0-50 px) so the lower half of the slider gives finer control at small
    // blur amounts. Mirror in js/composer/index.js + nodes/node_composition.py.
    if (layer.blur && layer.blur > 0) {
      const r = Math.pow(layer.blur / 100, 2) * 50;
      layerCtx.filter = "blur(" + r + "px)";
    }

    if (ppSetup) {
      // Warp the (masked) image into the pin-quad AABB. Two-tier (spec 7):
      // <=512px preview while dragging, full-res on release / clean render.
      let srcForWarp = layer.img;
      if (layer.hasMask_internal && layer.eraserMaskCanvas_internal) {
        this.renderCanvas.width = Wimg;
        this.renderCanvas.height = Him;
        this.renderCtx.clearRect(0, 0, Wimg, Him);
        this.renderCtx.drawImage(layer.img, 0, 0);
        this.renderCtx.globalCompositeOperation = "destination-out";
        this.renderCtx.drawImage(layer.eraserMaskCanvas_internal, 0, 0);
        this.renderCtx.globalCompositeOperation = "source-over";
        srcForWarp = this.renderCanvas;
      }
      const dragging = !cleanRender && this.isMouseDown;
      const outScale = dragging
        ? Math.min(1, 512 / Math.max(ppSetup.ow, ppSetup.oh))
        : 1;
      // Drag: cheap preview. Idle / clean: cached full-quality warp (4x SS
      // when the layer's antialias toggle is on) - see _cachedPowerPinWarp.
      const warped = dragging
        ? warpPowerPinCanvas(srcForWarp, ppSetup, outScale)
        : this._cachedPowerPinWarp(layer, ppSetup, srcForWarp);
      drawDisplay(warped);
    } else if (layer.hasMask_internal && layer.eraserMaskCanvas_internal) {
      // NON-DESTRUCTIVE MASK RENDER
      this.renderCanvas.width = layer.img.width;
      this.renderCanvas.height = layer.img.height;
      this.renderCtx.clearRect(
        0,
        0,
        this.renderCanvas.width,
        this.renderCanvas.height,
      );

      this.renderCtx.drawImage(layer.img, 0, 0);
      this.renderCtx.globalCompositeOperation = "destination-out";
      this.renderCtx.drawImage(layer.eraserMaskCanvas_internal, 0, 0);
      this.renderCtx.globalCompositeOperation = "source-over";

      drawDisplay(this.renderCanvas);
    } else {
      drawDisplay(layer.img);
    }

    layerCtx.filter = "none";
    layerCtx.restore();
    layerCtx.restore();

    // Clip-to-layer: grade the layer's own pixels, then composite them onto
    // the main canvas with the layer's real blend mode (opacity is already
    // baked into the offscreen by the source-over draw above).
    if (clipRefs.length) {
      const Wc = this.canvas.width, Hc = this.canvas.height;
      const cid = this._fxClipCtx.getImageData(0, 0, Wc, Hc);
      for (const ref of clipRefs)
        applyFx(cid.data, Wc, Hc, ref.adjustments, ref.amount01, ref.seed);
      this._fxClipCtx.putImageData(cid, 0, 0);
      this.ctx.save();
      if (layer.blendMode && BLEND_MAP[layer.blendMode])
        this.ctx.globalCompositeOperation = BLEND_MAP[layer.blendMode];
      this.ctx.drawImage(this._fxClipCanvas, 0, 0);
      this.ctx.restore();
    }

    // Draw selection border & handles on overlay canvas (not clipped by main canvas)
    if (!cleanRender && isSelected && oc) {
      oc.save();
      if (ppSetup) {
        // Pinned: outline the quad through the pin doc positions (doc coords,
        // no rotation) - the 4 corner handles double as pin handles.
        const pinPts = pinnedPointsToDoc(layer, Wimg, Him);
        oc.translate(pad, pad);
        oc.strokeStyle = layer.locked
          ? "#888"
          : this.selectedLayerIds.size > 1
            ? "#0ea5e9"
            : "#f66744";
        oc.lineWidth = 1;
        if (layer.isPlaceholder) oc.setLineDash([6, 4]);
        oc.beginPath();
        oc.moveTo(pinPts[0].x, pinPts[0].y);
        oc.lineTo(pinPts[1].x, pinPts[1].y);
        oc.lineTo(pinPts[2].x, pinPts[2].y);
        oc.lineTo(pinPts[3].x, pinPts[3].y);
        oc.closePath();
        oc.stroke();
        oc.setLineDash([]);
        if (
          !layer.locked &&
          this.selectedLayerIds.size === 1 &&
          this.activeMode !== "eraser"
        ) {
          const sz = this.handleSize;
          oc.fillStyle = "#fff";
          oc.strokeStyle = "#f66744";
          oc.lineWidth = 1;
          pinPts.forEach((p) => {
            oc.beginPath();
            oc.arc(p.x, p.y, sz / 2, 0, Math.PI * 2);
            oc.fill();
            oc.stroke();
          });
        }
      } else {
        oc.translate(layer.cx + pad, layer.cy + pad);
        oc.rotate((layer.rotation * Math.PI) / 180);

        oc.strokeStyle = layer.locked
        ? "#888"
        : this.selectedLayerIds.size > 1
          ? "#0ea5e9"
          : "#f66744";
      oc.lineWidth = 1;
      if (layer.isPlaceholder) oc.setLineDash([6, 4]);
      // Offset by 0.5 so a 1px stroke lands on the integer pixel grid (crisp,
      // no anti-alias bleed). w/h are already integer-rounded above.
      oc.strokeRect(-w / 2 + 0.5, -h / 2 + 0.5, w - 1, h - 1);
      oc.setLineDash([]);

      if (
        !layer.locked &&
        this.selectedLayerIds.size === 1 &&
        this.activeMode !== "eraser"
      ) {
        const sz = this.handleSize;
        oc.fillStyle = "#fff";
        oc.strokeStyle = "#f66744";
        oc.lineWidth = 1;

        const corners = [
          { x: -w / 2, y: -h / 2 },
          { x: w / 2, y: -h / 2 },
          { x: w / 2, y: h / 2 },
          { x: -w / 2, y: h / 2 },
        ];
        corners.forEach((p) => {
          oc.beginPath();
          oc.arc(p.x, p.y, sz / 2, 0, Math.PI * 2);
          oc.fill();
          oc.stroke();
        });

        oc.fillStyle = "#fff";
        oc.lineWidth = 1;
        oc.beginPath();
        oc.moveTo(0, -h / 2);
        oc.lineTo(0, -h / 2 - 30);
        oc.stroke();
        oc.beginPath();
        oc.arc(0, -h / 2 - 30, sz / 1.2, 0, Math.PI * 2);
        oc.fillStyle = "#f66744";
        oc.fill();
        oc.stroke();
        oc.fillStyle = "#fff";
        oc.font = "12px Arial";
        oc.textAlign = "center";
        oc.textBaseline = "middle";
        oc.fillText("\u21bb", 0, -h / 2 - 29);

        oc.fillStyle = "#f66744";
        oc.strokeStyle = "#fff";
        oc.lineWidth = 1;
        const sides = [
          { x: -w / 2, y: 0 },
          { x: w / 2, y: 0 },
          { x: 0, y: -h / 2 },
          { x: 0, y: h / 2 },
        ];
        sides.forEach((p) => {
          oc.beginPath();
          oc.arc(p.x, p.y, sz / 2, 0, Math.PI * 2);
          oc.fill();
          oc.stroke();
        });
      }
      }
      oc.restore();
    }
  });

  // Crop-mode overlay: full source dimmed + bright kept region + box/handles.
  // Interactive render only — never bake the crop UI into a save composite.
  if (!cleanRender && this.activeMode === "crop" && this._cropLayer)
    this.drawCropOverlay();

  this.ctx.globalAlpha = 1.0;
};

// Full-quality pinned warp, cached per layer. The 4x supersample pass costs
// ~1s on a large layer, so the canvas is only rewarped when the source image,
// the mask (stamp + canvas identity), or the pin params change. Drag previews
// never touch this cache (they use the cheap outScale<1 path).
PixaromaEditor.prototype._cachedPowerPinWarp = function (layer, ppSetup, srcForWarp) {
  if (!this._ppWarpCache) this._ppWarpCache = new Map();
  const mask =
    layer.hasMask_internal && layer.eraserMaskCanvas_internal
      ? layer.eraserMaskCanvas_internal
      : null;
  const key = JSON.stringify([
    ppSetup.ox, ppSetup.oy, ppSetup.ow, ppSetup.oh,
    ppSetup.q, ppSetup.p,
    ppSetup.x0, ppSetup.y0, ppSetup.x1, ppSetup.y1,
    ppSetup.unstretch,
    layer.powerPin.aa ? 1 : 0,
    layer._ppMaskStamp || 0,
  ]);
  const hit = this._ppWarpCache.get(layer.id);
  if (hit && hit.key === key && hit.img === layer.img && hit.mask === mask)
    return hit.canvas;
  const canvas = layer.powerPin.aa
    ? warpPowerPinCanvasAA(srcForWarp, ppSetup, 4)
    : warpPowerPinCanvas(srcForWarp, ppSetup);
  this._ppWarpCache.set(layer.id, { key, img: layer.img, mask, canvas });
  return canvas;
};

// Full-quality border-fade canvas (feather + edge blur), cached per layer
// like the pinned warp. The per-pixel pass is too heavy for every frame on a
// large layer, so fast slider input (events <150ms apart, see
// _lastFadeInput in ui.mjs) renders the treatment at <=512px and a quiet
// render recomputes full quality.
PixaromaEditor.prototype._borderFadeCanvas = function (layer, src, w, h, reduced) {
  const F = featherRadiusPx(layer.feather);
  const R = edgeBlurRadiusPx(layer.edgeBlur);
  if (!(F > 0) && !(R > 0)) return null;
  const s = reduced ? Math.min(1, 512 / Math.max(w, h)) : 1;
  const rw = Math.max(1, Math.round(w * s));
  const rh = Math.max(1, Math.round(h * s));
  if (!this._bfCache) this._bfCache = new Map();
  const key = JSON.stringify([rw, rh, F, R, layer._ppMaskStamp || 0]);
  const hit = this._bfCache.get(layer.id);
  if (hit && hit.key === key && hit.src === src && hit.img === layer.img)
    return hit.canvas;
  const canvas = borderFadeCanvas(src, rw, rh, F * s, R * s);
  this._bfCache.set(layer.id, { key, src, img: layer.img, canvas });
  return canvas;
};

PixaromaEditor.prototype.attemptRestore = async function () {
  let savedData = null;
  const composerWidget = (this.node.widgets || []).find(
    (w) => w.name === "ComposerWidget",
  );
  if (
    composerWidget &&
    composerWidget.value &&
    composerWidget.value.project_json
  ) {
    savedData = composerWidget.value.project_json;
  }

  if (!savedData || savedData === "{}" || savedData === "") return;

  if (this.statusText)
    this.statusText.innerText = "\u23f3 Restoring session...";
  this.overlay.style.pointerEvents = "none";
  this.overlay.style.opacity = "0.5";

  try {
    if (typeof savedData !== "string") savedData = JSON.stringify(savedData);
    const meta = JSON.parse(savedData);

    this.docWidth = meta.doc_w;
    this.docHeight = meta.doc_h;
    // Restore the user's saved BG colour. Older saves (pre-bg_color
    // field) get the legacy default so existing projects keep
    // rendering identically to how they did before.
    if (typeof meta.bg_color === "string" && meta.bg_color) {
      this._bgColor = meta.bg_color;
      if (this._bgColorInput) this._bgColorInput.value = meta.bg_color;
    }
    if (this._canvasSettings)
      this._canvasSettings.setSize(meta.doc_w, meta.doc_h);

    this.canvasContainer.style.width = this.docWidth + "px";
    this.canvasContainer.style.height = this.docHeight + "px";
    this.canvas.width = this.docWidth;
    this.canvas.height = this.docHeight;
    if (this.selCanvas) {
      this.selCanvas.width = this.docWidth + 2 * this.selPad;
      this.selCanvas.height = this.docHeight + 2 * this.selPad;
    }
    if (this.selHitArea) {
      this.selHitArea.style.width = this.docWidth + 2 * this.selPad + "px";
      this.selHitArea.style.height = this.docHeight + 2 * this.selPad + "px";
    }

    const layersToLoad = meta.layers;
    let loadedCount = 0;
    if (!layersToLoad || layersToLoad.length === 0) {
      this.finishRestore();
      return;
    }

    this.layers = new Array(layersToLoad.length);

    layersToLoad.forEach((mLayer, i) => {
      // FX adjustment layer: no image — build synchronously. This single branch
      // covers all the restore paths (it never reaches img.onload/onerror).
      if (mLayer.isAdjustment) {
        this.layers[i] = {
          id: mLayer.id,
          name: mLayer.name || "Color Grade",
          isAdjustment: true,
          adjustments: { ...(mLayer.adjustments || {}) },
          presetId: mLayer.presetId || "Custom",
          visible: mLayer.visible !== false,
          opacity: mLayer.opacity ?? 1,
          locked: !!mLayer.locked,
          clipToLayer: !!mLayer.clipToLayer,
        };
        loadedCount++;
        if (loadedCount === layersToLoad.length) this.finishRestore();
        return;
      }
      // Placeholder layers don't need image loading — build canvas immediately
      if (mLayer.isPlaceholder) {
        const color = mLayer.placeholderColor || "#808080";
        const w = mLayer.naturalWidth || Math.round(this.docWidth / 2);
        const h = mLayer.naturalHeight || Math.round(this.docHeight / 2);
        this.layers[i] = {
          id: mLayer.id,
          name: mLayer.name,
          isPlaceholder: true,
          placeholderColor: color,
          inputIndex: mLayer.inputIndex || 1,
          fillMode: mLayer.fillMode || "cover",
          phRatio: mLayer.phRatio,
          removeBgOnExec: !!mLayer.removeBgOnExec,
          bgRemovalQuality: mLayer.bgRemovalQuality,
          img: this._makePlaceholderImage(w, h, color, mLayer.name, (bitmapImg) => {
            this.layers[i].img = bitmapImg;
          }),
          cx: mLayer.cx,
          cy: mLayer.cy,
          scaleX: mLayer.scaleX,
          scaleY: mLayer.scaleY,
          rotation: mLayer.rotation,
          opacity: mLayer.opacity,
          visible: mLayer.visible,
          locked: mLayer.locked,
          flippedX: mLayer.flippedX,
          flippedY: mLayer.flippedY,
          blendMode: mLayer.blendMode || "Normal",
          blur: mLayer.blur || 0,
          feather: mLayer.feather || 0,
          edgeBlur: mLayer.edgeBlur || 0,
          rawB64_internal: null,
          rawServerPath: "__placeholder__",
          savedOnServer: true,
          eraserMaskCanvas_internal: null,
          eraserMaskCtx_internal: null,
          hasMask_internal: false,
          savedMaskPath_internal: mLayer.maskSrc || null,
          cropRect: mLayer.cropRect || null,
          powerPin: mLayer.powerPin
            ? { ...mLayer.powerPin, pins: mLayer.powerPin.pins.map((pt) => [...pt]) }
            : null,
          sourceImg: null,
        };
        loadedCount++;
        if (loadedCount === layersToLoad.length) this.finishRestore();
        return;
      }

      const img = new Image();
      img.crossOrigin = "Anonymous";
      const fileNameOnly = mLayer.src ? mLayer.src.split(/[\\/]/).pop() : "";

      img.onload = () => {
        this.layers[i] = {
          id: mLayer.id,
          name: mLayer.name,
          img: img,
          // Text layer: its saved src IS the baked text bitmap, so it loads
          // through this normal image path; carry the text-ness + content so it
          // stays a re-editable text layer (panel swap, "T", crisp resize).
          isText: !!mLayer.isText,
          textState: mLayer.textState ? { ...mLayer.textState } : undefined,
          cx: mLayer.cx,
          cy: mLayer.cy,
          scaleX: mLayer.scaleX,
          scaleY: mLayer.scaleY,
          rotation: mLayer.rotation,
          opacity: mLayer.opacity,
          visible: mLayer.visible,
          locked: mLayer.locked,
          flippedX: mLayer.flippedX,
          flippedY: mLayer.flippedY,
          blendMode: mLayer.blendMode || "Normal",
          blur: mLayer.blur || 0,
          feather: mLayer.feather || 0,
          edgeBlur: mLayer.edgeBlur || 0,
          removeBgOnExec: !!mLayer.removeBgOnExec,
          bgRemovalQuality: mLayer.bgRemovalQuality,
          rawB64_internal: null,
          rawServerPath: mLayer.src,
          savedOnServer: true,
          savedMaskPath_internal: mLayer.maskSrc || null,
          cropRect: mLayer.cropRect || null,
          powerPin: mLayer.powerPin
            ? { ...mLayer.powerPin, pins: mLayer.powerPin.pins.map((pt) => [...pt]) }
            : null,
          sourceImg: null,
        };
        loadedCount++;
        if (loadedCount === layersToLoad.length) this.finishRestore();
      };

      img.onerror = () => {
        const tempCanvas = document.createElement("canvas");
        tempCanvas.width = 512;
        tempCanvas.height = 512;
        const tCtx = tempCanvas.getContext("2d");
        tCtx.fillStyle = "#333";
        tCtx.fillRect(0, 0, 512, 512);
        tCtx.strokeStyle = "red";
        tCtx.lineWidth = 10;
        tCtx.strokeRect(0, 0, 512, 512);
        tCtx.fillStyle = "white";
        tCtx.font = "bold 30px Arial";
        tCtx.fillText("Missing Image", 150, 256);
        const placeholder = new Image();
        placeholder.onload = () => {
          this.layers[i] = {
            id: mLayer.id,
            name: mLayer.name + " (Missing)",
            img: placeholder,
            isText: !!mLayer.isText,
            textState: mLayer.textState ? { ...mLayer.textState } : undefined,
            cx: mLayer.cx,
            cy: mLayer.cy,
            scaleX: mLayer.scaleX,
            scaleY: mLayer.scaleY,
            rotation: mLayer.rotation,
            opacity: mLayer.opacity,
            visible: mLayer.visible,
            locked: mLayer.locked,
            flippedX: mLayer.flippedX,
            flippedY: mLayer.flippedY,
            blendMode: mLayer.blendMode || "Normal",
            blur: mLayer.blur || 0,
            feather: mLayer.feather || 0,
            edgeBlur: mLayer.edgeBlur || 0,
            removeBgOnExec: !!mLayer.removeBgOnExec,
            bgRemovalQuality: mLayer.bgRemovalQuality,
            rawB64_internal: null,
            rawServerPath: mLayer.src,
            savedOnServer: true,
            savedMaskPath_internal: mLayer.maskSrc || null,
            // Do NOT crop the 512px "Missing Image" placeholder with the real
            // image's (out-of-bounds) rect - it would blank the marker. The
            // true cropRect stays in the saved JSON for when the file returns.
            cropRect: null,
            // The 512px marker is not the real image - its (real-image-pixel)
            // pins don't apply. The pin data stays in the saved JSON for when
            // the file returns.
            powerPin: null,
            sourceImg: null,
          };
          loadedCount++;
          if (loadedCount === layersToLoad.length) this.finishRestore(true);
        };
        placeholder.src = tempCanvas.toDataURL();
      };
      img.src = pixApiUrl(`/view?filename=${encodeURIComponent(fileNameOnly)}&type=input&subfolder=pixaroma&t=${Date.now()}`);
    });
  } catch (err) {
    console.error("Pixaroma Restore Error:", err);
    this.finishRestore(true);
  }
};

PixaromaEditor.prototype.finishRestore = function (hadError = false) {
  // Re-bake crops BEFORE loading masks, so the mask canvas (loaded next, sized
  // to layer.img) matches the cropped image. recenter=false: cx/cy already hold
  // the saved cropped center, so we must not shift them again.
  // Placeholders are skipped: their layer.img is the natural-sized (post-crop)
  // color block, and their cropRect applies to the WIRED INPUT (Python applies
  // it at exec time, the preview applies it on connect). Baking the
  // source-pixel crop onto the block would draw a small corner of it into a
  // mostly-transparent canvas (blue square top-left, not filling the pins).
  this.layers.forEach((l) => {
    if (l && l.cropRect && l.img && !l.isPlaceholder) {
      l.sourceImg = l.img;
      this.applyCropToLayer(l, false);
    }
  });

  this.layers.forEach((l) => {
    if (l.savedMaskPath_internal) {
      const maskFileName = l.savedMaskPath_internal.split(/[\\/]/).pop();
      const maskUrl = pixApiUrl(`/view?filename=${encodeURIComponent(maskFileName)}&type=input&subfolder=pixaroma&t=${Date.now()}`);
      this.prepareLayerMask(l, maskUrl);
    }
  });

  this.syncNodeInputs();

  this.overlay.style.pointerEvents = "auto";
  this.overlay.style.opacity = "1";
  if (this.statusText)
    this.statusText.innerText = hadError
      ? "Ready (Some images missing)."
      : "Session restored.";
  this.fitViewToWorkspace();
  this.ui.updateActiveLayerUI();
  this.draw();

  this.history = [];
  this.historyIndex = -1;
  this.pushHistory();

  // Auto-preview any placeholders that are already connected
  this.previewAllConnectedPlaceholders();
};

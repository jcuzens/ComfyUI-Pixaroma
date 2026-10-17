// Layer export/import: pull ONE layer (plus the FX layers clipped to it) out
// of a scene as a self-contained JSON file. Two purposes:
//   1. Restore - re-import into the same or another composer (applyLayerExport).
//   2. Read the details - corners, pins, transform, FX values are plain numbers
//      so other tools (Fusion, etc.) can reuse the same layout.
//
// The pure transforms (buildLayerExport / parseLayerExport / buildImportLayers /
// resolveInputIndex) are DOM-free and run in Node for parity tests. The editor
// glue (applyLayerExport) lives at the bottom and needs a live editor.

import { PixaromaLayers } from "./layers.mjs";
import { NEUTRAL, mergeAdjustments, resolveClipTargets } from "./fx_engine.mjs";

export const LAYER_EXPORT_FORMAT = "pixaroma/layer-export";
export const LAYER_EXPORT_VERSION = 1;

// Unique-per-process id suffix so two rapid imports can't collide on a
// same-millisecond Date.now(). (Same concern the add/duplicate paths note.)
let _seq = 0;
const freshId = (prefix) =>
  `${prefix}_${Date.now().toString(36)}_${(_seq++).toString(36)}${Math.random().toString(36).slice(2, 6)}`;

// Corners of a layer in DOCUMENT space (canvas px, top-left origin). Reuses the
// exact handle math so exported corners match what the editor draws - including
// the 4 power-pin positions when the layer is pinned.
function docCorners(layer) {
  const pts = PixaromaLayers.getTransformedPoints(layer);
  return pts.slice(0, 4).map((p) => [p.x, p.y]);
}

// The FX adjustment layers that grade THIS layer (clip-to-layer). Uses the same
// resolveClipTargets the render loop uses, so the export matches the pixels.
function clippedFxFor(layers, targetIndex) {
  const { byTarget } = resolveClipTargets(layers);
  const refs = byTarget.get(targetIndex) || [];
  return refs.map((r) => layers[r.index]);
}

export function buildLayerExport({ docWidth, docHeight, bgColor, layers, layerId }) {
  const targetIndex = layers.findIndex((l) => l.id === layerId);
  if (targetIndex === -1) throw new Error("Layer not found in scene");
  const layer = layers[targetIndex];
  if (layer.isAdjustment) throw new Error("FX layers can't be exported on their own");

  const entry = {
    id: layer.id,
    name: layer.name,
    cx: layer.cx,
    cy: layer.cy,
    scaleX: layer.scaleX,
    scaleY: layer.scaleY,
    rotation: layer.rotation,
    opacity: layer.opacity ?? 1,
    visible: layer.visible !== false,
    flippedX: !!layer.flippedX,
    flippedY: !!layer.flippedY,
    blendMode: layer.blendMode || "Normal",
    blur: layer.blur || 0,
    feather: layer.feather || 0,
    edgeBlur: layer.edgeBlur || 0,
    cropRect: layer.cropRect ? { ...layer.cropRect } : null,
    powerPin: layer.powerPin
      ? { ...layer.powerPin, pins: layer.powerPin.pins.map((p) => [...p]) }
      : null,
    maskSrc: layer.savedMaskPath_internal || null,
    corners: docCorners(layer),
  };
  if (layer.removeBgOnExec) entry.removeBgOnExec = true;
  if (layer.bgRemovalQuality && layer.bgRemovalQuality !== "normal")
    entry.bgRemovalQuality = layer.bgRemovalQuality;
  if (layer.isPlaceholder) {
    entry.isPlaceholder = true;
    entry.placeholderColor = layer.placeholderColor || "#808080";
    entry.inputIndex = layer.inputIndex ?? 1;
    entry.fillMode = layer.fillMode || "cover";
    if (layer.phRatio) entry.phRatio = layer.phRatio;
    entry.naturalWidth = layer.img ? layer.img.width : 512;
    entry.naturalHeight = layer.img ? layer.img.height : 512;
  } else if (layer.rawServerPath && layer.rawServerPath !== "__placeholder__") {
    entry.src = layer.rawServerPath;
  }
  if (layer.isText) {
    entry.isText = true;
    entry.textState = { ...layer.textState };
  }

  const effects = clippedFxFor(layers, targetIndex).map((fx) => ({
    name: fx.name || "Color Grade",
    presetId: fx.presetId || "Custom",
    amount: fx.opacity ?? 1,
    adjustments: mergeAdjustments(fx.adjustments),
    clipToLayer: true,
  }));

  return {
    format: LAYER_EXPORT_FORMAT,
    version: LAYER_EXPORT_VERSION,
    doc_w: docWidth,
    doc_h: docHeight,
    bg_color: bgColor || "#1e1e1e",
    layer: entry,
    effects,
  };
}

export function parseLayerExport(text) {
  if (typeof text !== "string") throw new Error("Expected a JSON string");
  let data;
  try {
    data = JSON.parse(text);
  } catch {
    throw new Error("Not valid JSON");
  }
  if (!data || data.format !== LAYER_EXPORT_FORMAT)
    throw new Error("Not a Pixaroma layer export");
  const L = data.layer;
  if (!L || typeof L !== "object") throw new Error("Missing layer");
  for (const k of ["cx", "cy", "scaleX", "scaleY", "rotation", "opacity"]) {
    if (typeof L[k] !== "number" || !Number.isFinite(L[k]))
      throw new Error(`Bad layer field: ${k}`);
  }
  if (typeof data.doc_w !== "number" || typeof data.doc_h !== "number")
    throw new Error("Bad doc size");
  if (L.isPlaceholder) {
    if (typeof L.naturalWidth !== "number" || typeof L.naturalHeight !== "number")
      throw new Error("Bad placeholder size");
    if (!Number.isInteger(L.inputIndex) || L.inputIndex < 1)
      throw new Error("Bad inputIndex");
  }
  if (!Array.isArray(data.effects)) throw new Error("Missing effects");
  return data;
}

// Rebuild live layer objects from a parsed export: [target, ...clippedFX].
// Fresh ids; target.img is left null for the editor glue to fill (placeholder
// block or /view reload). Order is target-then-FX so the clip still resolves.
export function buildImportLayers(data) {
  const L = data.layer;
  const target = {
    id: freshId(L.isPlaceholder ? "ph" : "imp"),
    name: L.name || "Imported",
    cx: L.cx,
    cy: L.cy,
    scaleX: L.scaleX,
    scaleY: L.scaleY,
    rotation: L.rotation,
    opacity: L.opacity ?? 1,
    visible: L.visible !== false,
    locked: false,
    flippedX: !!L.flippedX,
    flippedY: !!L.flippedY,
    blendMode: L.blendMode || "Normal",
    blur: L.blur || 0,
    feather: L.feather || 0,
    edgeBlur: L.edgeBlur || 0,
    cropRect: L.cropRect ? { ...L.cropRect } : null,
    powerPin: L.powerPin
      ? { ...L.powerPin, pins: L.powerPin.pins.map((p) => [...p]) }
      : null,
    removeBgOnExec: !!L.removeBgOnExec,
    bgRemovalQuality: L.bgRemovalQuality,
    rawB64_internal: null,
    rawServerPath: L.isPlaceholder ? "__placeholder__" : L.src || null,
    savedOnServer: true,
    savedMaskPath_internal: L.maskSrc || null,
    eraserMaskCanvas_internal: null,
    eraserMaskCtx_internal: null,
    hasMask_internal: false,
    sourceImg: null,
    img: null,
  };
  if (L.isPlaceholder) {
    target.isPlaceholder = true;
    target.placeholderColor = L.placeholderColor || "#808080";
    target.inputIndex = L.inputIndex;
    target.fillMode = L.fillMode || "cover";
    if (L.phRatio) target.phRatio = L.phRatio;
    target.naturalWidth = L.naturalWidth;
    target.naturalHeight = L.naturalHeight;
  }
  if (L.isText) {
    target.isText = true;
    target.textState = { ...L.textState };
  }

  const fxLayers = (data.effects || []).map((fx) => ({
    id: freshId("fx"),
    name: fx.name || "Color Grade",
    isAdjustment: true,
    adjustments: { ...NEUTRAL, ...(fx.adjustments || {}) },
    presetId: fx.presetId || "Custom",
    visible: true,
    opacity: fx.amount ?? 1,
    locked: false,
    clipToLayer: true,
  }));

  return [target, ...fxLayers];
}

// Keep the exported input slot when it's free in the target scene; otherwise
// take the next free one so two placeholders never share a socket.
export function resolveInputIndex(requested, usedIndices, nextFree) {
  return usedIndices.has(requested) ? nextFree : requested;
}

// ── Editor glue (DOM) ────────────────────────────────────────────────────────

// Import a parsed export into a live editor: fill the target's img (placeholder
// block or /view reload), push target + its FX onto the stack, re-bake crop /
// mask, add the input socket, select, redraw. Mirrors attemptRestore's tail.
export async function applyLayerExport(editor, data) {
  // Dynamic import: api_url.mjs pulls in ComfyUI core (scripts/api.js), which
  // is browser-only. The pure functions above must stay Node-importable.
  const { pixApiUrl } = await import("../shared/api_url.mjs");
  const built = buildImportLayers(data);
  const target = built[0];

  if (target.isPlaceholder) {
    const used = new Set(
      editor.layers.filter((l) => l.isPlaceholder).map((l) => l.inputIndex),
    );
    const idx = resolveInputIndex(
      target.inputIndex,
      used,
      editor._nextPlaceholderIndex(),
    );
    if (idx !== target.inputIndex) {
      target.inputIndex = idx;
      target.name = `image_${idx}`;
    }
    target.img = editor._makePlaceholderImage(
      target.naturalWidth,
      target.naturalHeight,
      target.placeholderColor,
      target.name,
      (bitmap) => {
        target.img = bitmap;
        editor.draw();
      },
    );
    finalizeImport(editor, built, pixApiUrl);
    return;
  }

  const fileName = (target.rawServerPath || "").split(/[\\/]/).pop();
  const img = new Image();
  img.crossOrigin = "Anonymous";
  img.onload = () => {
    target.img = img;
    finalizeImport(editor, built, pixApiUrl);
  };
  img.onerror = () => {
    if (editor._layout)
      editor._layout.setStatus(
        `Import failed - ${fileName} not found in input/pixaroma`,
        "warn",
      );
  };
  img.src = pixApiUrl(
    `/view?filename=${encodeURIComponent(fileName)}&type=input&subfolder=pixaroma&t=${Date.now()}`,
  );
}

function finalizeImport(editor, built, pixApiUrl) {
  const target = built[0];
  editor.layers.push(...built);
  if (target.cropRect && target.img && !target.isPlaceholder) {
    target.sourceImg = target.img;
    editor.applyCropToLayer(target, false);
  }
  if (target.savedMaskPath_internal) {
    const maskFileName = target.savedMaskPath_internal.split(/[\\/]/).pop();
    editor.prepareLayerMask(
      target,
      pixApiUrl(
        `/view?filename=${encodeURIComponent(maskFileName)}&type=input&subfolder=pixaroma&t=${Date.now()}`,
      ),
    );
  }
  editor.syncNodeInputs();
  editor.selectedLayerIds = new Set([target.id]);
  editor.syncActiveLayerIndex();
  editor.ui.updateActiveLayerUI();
  editor.draw();
  editor.pushHistory();
  editor.previewAllConnectedPlaceholders?.();
}

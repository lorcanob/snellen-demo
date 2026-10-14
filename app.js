// Snellen web spike: can the desktop app's Python core run in a browser, fast enough?
//
// - Pyodide runs the `snellen` package (geometry, face spans, card detection, scoring),
//   installed from the wheel that web/build.py puts in dist/.
// - MediaPipe's JavaScript Face Landmarker finds the face; its landmarks go straight to
//   snellen.face, and the camera frame to snellen.card.find_card.
// - A chart line is drawn on a canvas at the size snellen.geometry computes, so it can
//   be checked with a ruler.
//
// Screening toy only - not a medical test. Nothing is uploaded: all processing is local.

import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";
import {
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/vision_bundle.mjs";

const MEDIAPIPE_WASM = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.0/wasm";
const FACE_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";
const CARD_WIDTH_MM = 85.6;
const DEFAULT_PX_PER_MM = 96 / 25.4; // the CSS reference pixel; real screens differ
const STATS_WINDOW = 30; // frames averaged in the stats panel
const STORAGE_KEY = "snellen.pxPerMm";

const $ = (id) => document.getElementById(id);
const status = (text, isError = false) => {
  $("status").textContent = text;
  $("status").className = isError ? "error" : "muted";
};

/** Load Pyodide, its packages, our wheel and glue.py; returns the glue module. */
async function loadPython() {
  status("Loading Python (Pyodide)...");
  const pyodide = await loadPyodide();
  status("Loading numpy and OpenCV...");
  await pyodide.loadPackage(["numpy", "opencv-python", "micropip"]);
  const manifest = await (await fetch("dist/manifest.json")).json();
  const wheelUrl = new URL(`dist/${manifest.wheel}`, location.href).href;
  status("Installing the snellen package...");
  // deps=False: the wheel lists the desktop app's dependencies (MediaPipe, pygame),
  // which don't exist in the browser and aren't needed by the modules used here.
  await pyodide.runPythonAsync(`
import micropip
await micropip.install(${JSON.stringify(wheelUrl)}, deps=False)
`);
  const site = pyodide.runPython("import site; site.getsitepackages()[0]");
  pyodide.FS.writeFile(`${site}/glue.py`, await (await fetch("glue.py")).text());
  return { pyodide, glue: pyodide.pyimport("glue"), geometry: pyodide.pyimport("snellen.geometry") };
}

/** Load MediaPipe's face landmarker for video frames. */
async function loadLandmarker() {
  status("Loading the face model...");
  const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM);
  return FaceLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: FACE_MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
  });
}

/** Convert a Python result (a PyProxy dict) to a plain JS object and free it. */
function toJs(proxy) {
  const value = proxy.toJs({ dict_converter: Object.fromEntries });
  proxy.destroy();
  return value;
}

/** Rolling averages for the stats panel. */
class Rolling {
  constructor(size) {
    this.size = size;
    this.values = {};
  }
  add(name, value) {
    if (value === undefined || value === null || Number.isNaN(value)) return;
    const list = (this.values[name] ??= []);
    list.push(value);
    if (list.length > this.size) list.shift();
  }
  mean(name) {
    const list = this.values[name] ?? [];
    return list.length ? list.reduce((a, b) => a + b, 0) / list.length : NaN;
  }
}

// --- 1. camera ---------------------------------------------------------------------

function startCameraLoop(glue, landmarker) {
  const video = $("video");
  const overlay = $("overlay");
  const grab = document.createElement("canvas");
  const grabCtx = grab.getContext("2d", { willReadFrequently: true });
  const stats = new Rolling(STATS_WINDOW);
  let lastVideoTime = -1;
  let lastFrameAt = performance.now();

  async function start() {
    $("start-camera").disabled = true;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      video.srcObject = stream;
      await video.play();
    } catch (error) {
      status(`Camera unavailable: ${error.message}`, true);
      $("start-camera").disabled = false;
      return;
    }
    grab.width = overlay.width = video.videoWidth;
    grab.height = overlay.height = video.videoHeight;
    status("Running. Hold a bank card flat on your forehead to test card detection.");
    requestAnimationFrame(tick);
  }

  function tick() {
    requestAnimationFrame(tick);
    if (video.currentTime === lastVideoTime) return;
    lastVideoTime = video.currentTime;
    const now = performance.now();
    stats.add("frame_ms", now - lastFrameAt);
    lastFrameAt = now;

    const w = video.videoWidth;
    const h = video.videoHeight;
    const t0 = performance.now();
    const faces = landmarker.detectForVideo(video, now).faceLandmarks;
    const t1 = performance.now();
    let rgba; // undefined, not null: Pyodide turns undefined into None but null into JsNull
    if ($("find-card").checked) {
      grabCtx.drawImage(video, 0, 0, w, h);
      rgba = grabCtx.getImageData(0, 0, w, h).data;
    }
    const t2 = performance.now();
    const result = toJs(glue.measure(faces.length ? faces[0] : undefined, w, h, rgba));
    const t3 = performance.now();

    stats.add("landmarks_ms", t1 - t0);
    stats.add("grab_ms", t2 - t1);
    stats.add("python_ms", t3 - t2);
    stats.add("copy_ms", result.copy_ms);
    stats.add("card_ms", result.card_ms);
    stats.add("face", result.pupils ? 1 : 0);
    if (rgba) stats.add("card", result.card ? 1 : 0);
    stats.add("card_mm_ipd", result.card_mm_ipd);
    drawOverlay(overlay, result);
    showStats(result, stats, w, h);
  }

  $("start-camera").addEventListener("click", start);
  $("start-camera").disabled = false;
}

function drawOverlay(canvas, result) {
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 3;
  if (result.card) {
    ctx.strokeStyle = "#28dc5a";
    ctx.beginPath();
    result.card.corners.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
    ctx.closePath();
    ctx.stroke();
  }
  if (result.pupils) {
    ctx.strokeStyle = "#28a0ff";
    for (const [x, y] of result.pupils) {
      ctx.beginPath();
      ctx.arc(x, y, 6, 0, 2 * Math.PI);
      ctx.stroke();
    }
  }
}

function showStats(result, stats, w, h) {
  const ms = (name) => stats.mean(name).toFixed(1).padStart(6);
  const pct = (name) => `${Math.round(100 * stats.mean(name))}%`;
  const fps = 1000 / stats.mean("frame_ms");
  const card = result.card ? `${result.card.width_px.toFixed(1)} px` : "-";
  const ipd = result.ipd_px ? `${result.ipd_px.toFixed(1)} px` : "-";
  const implied = stats.mean("card_mm_ipd");
  $("stats").textContent = [
    `camera ${w}x${h}   ${fps.toFixed(1)} fps   face found ${pct("face")}   card found ${pct("card")}`,
    `per frame (ms): landmarks ${ms("landmarks_ms")}  grab ${ms("grab_ms")}  ` +
      `python ${ms("python_ms")}  (copy ${ms("copy_ms")}, find_card ${ms("card_ms")})`,
    `pupil distance ${ipd}   card ${card}` +
      (Number.isNaN(implied) ? "" : `   -> pupil distance ${implied.toFixed(1)} mm`),
  ].join("\n");
}

// --- 2. screen size ------------------------------------------------------------------

function loadPxPerMm() {
  try {
    const saved = Number(localStorage.getItem(STORAGE_KEY));
    return saved > 0 ? saved : DEFAULT_PX_PER_MM;
  } catch {
    return DEFAULT_PX_PER_MM;
  }
}

function savePxPerMm(value) {
  try {
    localStorage.setItem(STORAGE_KEY, String(value));
  } catch {
    // private window or blocked storage: the value just isn't remembered
  }
}

// --- 3. chart ------------------------------------------------------------------------

/** Cap height of Optician Sans as a share of its font size (E has a flat top and bottom). */
function capHeightRatio(ctx) {
  ctx.font = "100px OpticianSans";
  const m = ctx.measureText("E");
  return (m.actualBoundingBoxAscent + m.actualBoundingBoxDescent) / 100;
}

function drawChart(glue, pxPerMm, seed) {
  const canvas = $("chart");
  const dpr = window.devicePixelRatio || 1;
  const cssW = canvas.clientWidth;
  const cssH = canvas.clientHeight;
  canvas.width = Math.round(cssW * dpr);
  canvas.height = Math.round(cssH * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0); // draw in CSS pixels; the browser anti-aliases
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, cssW, cssH);

  const line = toJs(
    glue.chart_line(
      $("chart-type").value,
      Number($("denominator").value),
      Number($("distance").value),
      pxPerMm,
      seed,
    ),
  );
  const h = line.height_px;
  $("ruler").textContent =
    `Each ${line.kind === "letters" ? "letter's cap height" : "optotype"} should measure ` +
    `${line.height_mm.toFixed(2)} mm with a ruler (${h.toFixed(2)} CSS px at ` +
    `${pxPerMm.toFixed(3)} px/mm).`;

  const gap = h; // one optotype width between them, as in the desktop app
  const rowW = line.shapes.length * (h + gap) - gap;
  const left0 = (cssW - rowW) / 2;
  const top = (cssH - h) / 2;
  const capRatio = capHeightRatio(ctx);
  ctx.fillStyle = "#141414";
  line.shapes.forEach((shape, i) => {
    const left = left0 + i * (h + gap);
    const cell = h / 5;
    if (shape.letter) {
      ctx.font = `${h / capRatio}px OpticianSans`;
      const m = ctx.measureText(shape.letter);
      const inkW = m.actualBoundingBoxLeft + m.actualBoundingBoxRight;
      ctx.fillText(shape.letter, left + (h - inkW) / 2 + m.actualBoundingBoxLeft, top + h);
    } else if (shape.cells) {
      for (const [x, y, w, hh] of shape.cells) {
        ctx.fillRect(left + x * cell, top + y * cell, w * cell, hh * cell);
      }
    } else {
      const [outer, inner] = shape.radii;
      const cx = left + 2.5 * cell;
      const cy = top + 2.5 * cell;
      ctx.beginPath();
      ctx.arc(cx, cy, outer * cell, 0, 2 * Math.PI);
      ctx.arc(cx, cy, inner * cell, 0, 2 * Math.PI, true);
      ctx.fill("evenodd");
      const [x, y, w, hh] = shape.gap;
      ctx.fillStyle = "#fff";
      ctx.fillRect(left + x * cell, top + y * cell, w * cell, hh * cell);
      ctx.fillStyle = "#141414";
    }
  });
}

// --- start-up -------------------------------------------------------------------------

async function main() {
  if (!window.isSecureContext) {
    status("The camera needs HTTPS (or localhost). Open this page over https://.", true);
  }
  const loadStarted = performance.now();
  const [{ glue, geometry }, landmarker] = await Promise.all([loadPython(), loadLandmarker()]);
  await document.fonts.load("100px OpticianSans");
  const loadSeconds = ((performance.now() - loadStarted) / 1000).toFixed(1);
  status(`Ready (loaded in ${loadSeconds} s). Start the camera, or check the letter sizes below.`);

  let pxPerMm = loadPxPerMm();
  let seed = 1;
  const redraw = () => {
    $("card-bar").style.width = `${CARD_WIDTH_MM * pxPerMm}px`;
    $("px-per-mm").textContent = `${pxPerMm.toFixed(3)} px/mm`;
    drawChart(glue, pxPerMm, seed);
  };
  for (const button of document.querySelectorAll("[data-step]")) {
    button.addEventListener("click", () => {
      pxPerMm = Math.max(1, pxPerMm + Number(button.dataset.step));
      savePxPerMm(pxPerMm);
      redraw();
    });
  }
  for (const name of glue.chart_names().toJs()) {
    $("chart-type").add(new Option(name, name));
  }
  for (const d of geometry.SNELLEN_DENOMINATORS.toJs()) {
    $("denominator").add(new Option(String(d), String(d)));
  }
  for (const id of ["chart-type", "denominator", "distance"]) {
    $(id).addEventListener("input", redraw);
  }
  $("reshuffle").addEventListener("click", () => {
    seed += 1;
    redraw();
  });
  window.addEventListener("resize", redraw);
  redraw();
  startCameraLoop(glue, landmarker);
}

main().catch((error) => {
  console.error(error);
  status(`Failed to start: ${error.message}`, true);
});

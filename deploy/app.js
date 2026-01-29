const BASE_MODEL_URL = "./conv_max_512_final.tflite";
const SUB_MODEL_URL = "./conv_max_512_final_sub.tflite";
const HALLWAYS_MODEL_URL = "./conv_512_hallways.tflite";
const ENTRANCES_MODEL_URL = "./conv_512_entrances.tflite";
const ROOMS_11_12_MODEL_URL = "./conv_512_rooms_11_12.tflite";
const ROOMS_19_20_MODEL_URL = "./conv_512_rooms_19_20.tflite";

const BASE_CLASSES_URL = "./classes.json";
const SUB_CLASSES_URL = "./classes_512_final_sub.json";
const HALLWAYS_CLASSES_URL = "./classes_512_hallways.json";
const ENTRANCES_CLASSES_URL = "./classes_512_entrances.json";
const ROOMS_11_12_CLASSES_URL = "./classes_512_rooms_11_12.json";
const ROOMS_19_20_CLASSES_URL = "./classes_512_rooms_19_20.json";

const IMG_SIZE = 512;
const MEAN = [121.302245, 122.731548, 123.377391];
const STD = [62.151375, 60.068601, 61.300509];

const video = document.getElementById("video");
const captureCanvas = document.getElementById("capture");
const captureBtn = document.getElementById("captureBtn");
const retakeBtn = document.getElementById("retakeBtn");
const uploadBtn = document.getElementById("uploadBtn");
const uploadInput = document.getElementById("uploadInput");
const switchBtn = document.getElementById("switchBtn");
const statusEl = document.getElementById("status");
const resultsBaseEl = document.getElementById("resultsBase");
const resultsHierEl = document.getElementById("resultsHier");
const togglePredBtn = document.getElementById("togglePredBtn");
const groupToggleBtn = document.getElementById("groupToggleBtn");
const panelBase = document.getElementById("panelBase");
const panelHier = document.getElementById("panelHier");
const placeholderEl = document.getElementById("placeholder");

let baseModel = null;
let subModel = null;
let baseClasses = [];
let subClasses = [];
let groupModels = {};
let groupClasses = {};
const modelHasRun = new WeakSet();
let stream = null;
let facingMode = "environment";
let modelReady = false;
let cameraReady = false;
let busy = false;
let activePanel = "base";
let useGroupModels = false;
let lastBaseItems = null;
let lastHierItems = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function updateButtons() {
  const canCapture = modelReady && cameraReady && !busy;
  captureBtn.disabled = !canCapture;
  uploadBtn.disabled = !modelReady || busy;
  switchBtn.disabled = !cameraReady || busy;
}

function clearResults() {
  const empty =
    '<li class="rounded-lg border border-dashed border-slate-200 bg-white/70 px-3 py-2">No results yet.</li>';
  resultsBaseEl.innerHTML = empty;
  resultsHierEl.innerHTML = empty;
  lastBaseItems = null;
  lastHierItems = null;
}

function setPlaceholder(text) {
  if (text) {
    placeholderEl.textContent = text;
    placeholderEl.classList.remove("hidden");
  } else {
    placeholderEl.classList.add("hidden");
  }
}

function showCaptureView() {
  video.classList.add("hidden");
  captureCanvas.classList.remove("hidden");
  setPlaceholder("");
  captureBtn.classList.add("hidden");
  retakeBtn.classList.remove("hidden");
}

function showLiveView() {
  captureCanvas.classList.add("hidden");
  captureBtn.classList.remove("hidden");
  retakeBtn.classList.add("hidden");
  if (cameraReady) {
    video.classList.remove("hidden");
    setPlaceholder("");
  } else {
    video.classList.add("hidden");
    setPlaceholder("Camera unavailable. Upload a photo.");
  }
}

function updatePanels() {
  const wide = window.matchMedia("(min-width: 768px)").matches;
  if (wide) {
    panelBase.classList.remove("hidden");
    panelHier.classList.remove("hidden");
    return;
  }
  panelBase.classList.toggle("hidden", activePanel !== "base");
  panelHier.classList.toggle("hidden", activePanel !== "hier");
}

function updateGroupToggle() {
  groupToggleBtn.textContent = useGroupModels ? "Disable Groups" : "Enable Groups";
  if (!useGroupModels) {
    renderMessage(resultsHierEl, "Groups disabled. Enable to run.");
  } else if (lastHierItems) {
    renderList(resultsHierEl, lastHierItems);
  } else {
    renderMessage(resultsHierEl, "Groups enabled. Capture to run.");
  }
}

function getSourceSize(source) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  return { sw, sh };
}

function drawToSquare(source, canvas) {
  const ctx = canvas.getContext("2d");
  const { sw, sh } = getSourceSize(source);
  if (!sw || !sh) return false;

  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;

  // Pad with dataset mean so standardized padding becomes zero (matches training).
  ctx.fillStyle = `rgb(${MEAN[0]}, ${MEAN[1]}, ${MEAN[2]})`;
  ctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);

  const scale = Math.min(IMG_SIZE / sw, IMG_SIZE / sh);
  const dw = Math.round(sw * scale);
  const dh = Math.round(sh * scale);
  const dx = Math.floor((IMG_SIZE - dw) / 2);
  const dy = Math.floor((IMG_SIZE - dh) / 2);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, dx, dy, dw, dh);
  return true;
}

function makeInputTensor(canvas) {
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const { data } = ctx.getImageData(0, 0, width, height);
  const input = new Float32Array(width * height * 3);
  let j = 0;
  for (let i = 0; i < data.length; i += 4) {
    input[j++] = (data[i] - MEAN[0]) / STD[0];
    input[j++] = (data[i + 1] - MEAN[1]) / STD[1];
    input[j++] = (data[i + 2] - MEAN[2]) / STD[2];
  }
  return tf.tensor4d(input, [1, height, width, 3], "float32");
}

const MAX_LABEL_LEN = 28;

function shortenLabel(name, mode) {
  const firstIdx = name.indexOf("__");
  if (firstIdx === -1) return name;
  const firstLetter = name.charAt(0);
  if (mode === 1) return firstLetter + name.slice(firstIdx);
  const secondIdx = name.indexOf("__", firstIdx + 2);
  if (secondIdx === -1) return firstLetter + name.slice(firstIdx);
  return firstLetter + name.slice(secondIdx);
}

function shortenIfNeeded(name) {
  let out = name;
  if (out.length > MAX_LABEL_LEN) out = shortenLabel(out, 1);
  if (out.length > MAX_LABEL_LEN) out = shortenLabel(out, 2);
  return out;
}

function formatPct(value, digits = 2) {
  return `${(value * 100).toFixed(digits)}%`;
}

function makeLabelSpan(name) {
  const label = document.createElement("span");
  label.className = "block min-w-0 flex-1";
  label.textContent = shortenIfNeeded(name);
  label.title = name;
  label.style.overflow = "hidden";
  label.style.textOverflow = "ellipsis";
  label.style.whiteSpace = "nowrap";
  label.style.flex = "1 1 0%";
  return label;
}

function renderMessage(listEl, text) {
  listEl.innerHTML = `<li class="rounded-lg border border-dashed border-slate-200 bg-white/70 px-3 py-2">${text}</li>`;
}

function waitForPaint() {
  return new Promise((resolve) =>
    requestAnimationFrame(() => requestAnimationFrame(resolve))
  );
}

function renderList(listEl, items) {
  listEl.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.className =
      "rounded-lg border border-slate-200 bg-white/70 px-3 py-2 overflow-hidden";

    const row = document.createElement("div");
    row.className = "flex w-full min-w-0 items-center gap-3";
    const label = makeLabelSpan(item.name);
    const prob = document.createElement("span");
    prob.className = "shrink-0 text-xs font-semibold text-slate-600";
    prob.textContent = formatPct(item.prob);
    row.append(label, prob);
    li.appendChild(row);

    if (item.children && item.children.length) {
      const subList = document.createElement("ul");
      subList.className =
        "mt-2 space-y-1 border-l border-slate-200 pl-3 text-xs text-slate-600";
      item.children.forEach((child) => {
        const subLi = document.createElement("li");
        subLi.className = "flex w-full min-w-0 items-center gap-2";
        const subLabel = makeLabelSpan(child.name);
        const subProb = document.createElement("span");
        subProb.className = "shrink-0 text-[11px] font-semibold text-slate-500";
        subProb.textContent = `${formatPct(child.overall)} (${formatPct(
          child.prob,
          1
        )} of group)`;
        subLi.append(subLabel, subProb);
        subList.appendChild(subLi);
      });
      li.appendChild(subList);
    }

    listEl.appendChild(li);
  });
}

async function loadJson(url) {
  const res = await fetch(url);
  return res.json();
}

async function loadModelAndClasses() {
  try {
    setStatus("Loading models...");
    await tf.ready();
    tflite.setWasmPath(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/"
    );
    const [
      baseM,
      subM,
      hallsM,
      entrM,
      rooms11M,
      rooms19M,
      baseC,
      subC,
      hallsC,
      entrC,
      rooms11C,
      rooms19C,
    ] = await Promise.all([
      tflite.loadTFLiteModel(BASE_MODEL_URL, { numThreads: 1 }),
      tflite.loadTFLiteModel(SUB_MODEL_URL, { numThreads: 1 }),
      tflite.loadTFLiteModel(HALLWAYS_MODEL_URL, { numThreads: 1 }),
      tflite.loadTFLiteModel(ENTRANCES_MODEL_URL, { numThreads: 1 }),
      tflite.loadTFLiteModel(ROOMS_11_12_MODEL_URL, { numThreads: 1 }),
      tflite.loadTFLiteModel(ROOMS_19_20_MODEL_URL, { numThreads: 1 }),
      loadJson(BASE_CLASSES_URL),
      loadJson(SUB_CLASSES_URL),
      loadJson(HALLWAYS_CLASSES_URL),
      loadJson(ENTRANCES_CLASSES_URL),
      loadJson(ROOMS_11_12_CLASSES_URL),
      loadJson(ROOMS_19_20_CLASSES_URL),
    ]);

    baseModel = baseM;
    subModel = subM;
    baseClasses = baseC.class_names || [];
    subClasses = subC.class_names || [];
    groupModels = {
      "Lise-Meitner-Str-9_9377__Indoor__9377_Hallways": hallsM,
      "Lise-Meitner-Str-9_9377__Outdoor__Entrances": entrM,
      "Willy-Messerschmitt-5_9387__Indoor__RM_011_012": rooms11M,
      "Willy-Messerschmitt-5_9387__Indoor__RM_019_020": rooms19M,
    };
    groupClasses = {
      "Lise-Meitner-Str-9_9377__Indoor__9377_Hallways":
        hallsC.class_names || [],
      "Lise-Meitner-Str-9_9377__Outdoor__Entrances":
        entrC.class_names || [],
      "Willy-Messerschmitt-5_9387__Indoor__RM_011_012":
        rooms11C.class_names || [],
      "Willy-Messerschmitt-5_9387__Indoor__RM_019_020":
        rooms19C.class_names || [],
    };
    modelReady = true;
    setStatus("Models ready. Use camera or upload.");
    updateButtons();
  } catch (err) {
    setStatus("Model load failed. Check console.");
    console.error(err);
  }
}

async function startCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      cameraReady = false;
      updateButtons();
      showLiveView();
      setStatus("Camera not supported. Use upload.");
      return;
    }
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: facingMode } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    cameraReady = true;
    updateButtons();
    showLiveView();
    setStatus(modelReady ? "Ready." : "Loading models...");
  } catch (err) {
    cameraReady = false;
    updateButtons();
    showLiveView();
    setStatus("Camera access failed. Use upload.");
    console.error(err);
  }
}

async function runModel(model, canvas) {
  const runOnce = async () => {
    const output = tf.tidy(() => {
      const input = makeInputTensor(canvas);
      const result = model.predict(input);
      return Array.isArray(result) ? result[0] : result;
    });
    const probs = await output.data();
    output.dispose();
    return probs;
  };

  let probs = await runOnce();
  let sum = 0;
  for (let i = 0; i < probs.length; i += 1) {
    sum += probs[i];
  }
  if (!modelHasRun.has(model) && sum < 1e-6) {
    probs = await runOnce();
  }
  modelHasRun.add(model);
  return probs;
}

async function buildBaseItems(canvas) {
  const probs = await runModel(baseModel, canvas);
  const items = baseClasses.map((name, index) => ({
    name,
    prob: probs[index] || 0,
  }));
  items.sort((a, b) => b.prob - a.prob);
  return items;
}

async function buildHierItems(canvas) {
  const probs = await runModel(subModel, canvas);
  const items = subClasses.map((name, index) => ({
    name,
    prob: probs[index] || 0,
  }));
  items.sort((a, b) => b.prob - a.prob);

  const enriched = await Promise.all(
    items.map(async (item) => {
      const groupModel = groupModels[item.name];
      if (!groupModel) return item;
      const childProbs = await runModel(groupModel, canvas);
      const classes = groupClasses[item.name] || [];
      const children = classes.map((name, idx) => ({
        name,
        prob: childProbs[idx] || 0,
        overall: (childProbs[idx] || 0) * item.prob,
      }));
      children.sort((a, b) => b.prob - a.prob);
      return { ...item, children };
    })
  );

  return enriched;
}

async function classifySource(source) {
  if (!modelReady || !baseModel || !subModel) {
    setStatus("Model not ready.");
    return;
  }
  if (busy) return;
  busy = true;
  updateButtons();
  clearResults();
  setStatus("Capturing...");

  const prepped = drawToSquare(source, captureCanvas);
  if (!prepped) {
    setStatus("Capture failed.");
    busy = false;
    updateButtons();
    return;
  }

  showCaptureView();
  renderMessage(resultsBaseEl, "Running base model...");
  if (useGroupModels) {
    renderMessage(resultsHierEl, "Running group models...");
  } else {
    renderMessage(resultsHierEl, "Groups disabled. Enable to run.");
  }
  await waitForPaint();
  setStatus(
    useGroupModels ? "Running base + group models..." : "Running base model..."
  );

  try {
    const baseItems = await buildBaseItems(captureCanvas);
    lastBaseItems = baseItems;
    renderList(resultsBaseEl, baseItems);

    if (useGroupModels) {
      const hierItems = await buildHierItems(captureCanvas);
      lastHierItems = hierItems;
      renderList(resultsHierEl, hierItems);
    } else {
      lastHierItems = null;
      renderMessage(resultsHierEl, "Groups disabled. Enable to run.");
    }
    setStatus("Done.");
  } catch (err) {
    setStatus("Inference failed. Check console.");
    console.error(err);
  } finally {
    busy = false;
    updateButtons();
  }
}

async function captureAndClassify() {
  if (!cameraReady || !video.videoWidth) {
    setStatus("Camera not ready.");
    return;
  }
  await classifySource(video);
}

function loadImageFromFile(file) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      URL.revokeObjectURL(url);
      resolve(img);
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(url);
      reject(err);
    };
    img.src = url;
  });
}

function retake() {
  clearResults();
  showLiveView();
  if (!modelReady) {
    setStatus("Loading models...");
  } else if (cameraReady) {
    setStatus("Ready.");
  } else {
    setStatus("Upload a photo.");
  }
}

captureBtn.addEventListener("click", captureAndClassify);
retakeBtn.addEventListener("click", retake);
uploadBtn.addEventListener("click", () => uploadInput.click());
uploadInput.addEventListener("change", async (event) => {
  const file = event.target.files && event.target.files[0];
  uploadInput.value = "";
  if (!file) return;
  try {
    const img = await loadImageFromFile(file);
    await classifySource(img);
  } catch (err) {
    setStatus("Upload failed. Check console.");
    console.error(err);
  }
});
switchBtn.addEventListener("click", async () => {
  facingMode = facingMode === "user" ? "environment" : "user";
  await startCamera();
  showLiveView();
});
togglePredBtn.addEventListener("click", () => {
  activePanel = activePanel === "base" ? "hier" : "base";
  updatePanels();
});
groupToggleBtn.addEventListener("click", () => {
  useGroupModels = !useGroupModels;
  updateGroupToggle();
  setStatus(useGroupModels ? "Groups enabled." : "Groups disabled.");
});
window.addEventListener("resize", updatePanels);

clearResults();
updatePanels();
updateGroupToggle();
loadModelAndClasses();
startCamera();

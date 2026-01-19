const MODEL_URL = "./conv_max_model_a.tflite";
const CLASSES_URL = "./classes.json";
const IMG_SIZE = 256;
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
const resultsEl = document.getElementById("results");
const placeholderEl = document.getElementById("placeholder");

const preprocessCanvas = document.createElement("canvas");

let model = null;
let classNames = [];
let stream = null;
let facingMode = "environment";
let modelReady = false;
let cameraReady = false;
let busy = false;

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
  resultsEl.innerHTML =
    '<li class="rounded-lg border border-dashed border-slate-200 bg-white/70 px-3 py-2">No results yet.</li>';
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

function getSourceSize(source) {
  const sw = source.videoWidth || source.naturalWidth || source.width;
  const sh = source.videoHeight || source.naturalHeight || source.height;
  return { sw, sh };
}

function drawToCanvas(source, canvas) {
  const ctx = canvas.getContext("2d");
  const { sw, sh } = getSourceSize(source);
  if (!sw || !sh) return false;
  canvas.width = sw;
  canvas.height = sh;
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(source, 0, 0, sw, sh);
  return true;
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

function renderResults(items) {
  resultsEl.innerHTML = "";
  items.forEach((item) => {
    const li = document.createElement("li");
    li.className =
      "flex items-center justify-between gap-3 rounded-lg border border-slate-200 bg-white/70 px-3 py-2";
    li.innerHTML = `
      <span class="truncate">${item.name}</span>
      <span class="text-xs font-semibold text-slate-600">${(item.prob * 100).toFixed(2)}%</span>
    `;
    resultsEl.appendChild(li);
  });
}

async function loadModelAndClasses() {
  try {
    setStatus("Loading model...");
    await tf.ready();
    tflite.setWasmPath(
      "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.10/wasm/"
    );
    model = await tflite.loadTFLiteModel(MODEL_URL, { numThreads: 1 });
    const res = await fetch(CLASSES_URL);
    const data = await res.json();
    classNames = data.class_names || [];
    modelReady = true;
    setStatus("Model ready. Use camera or upload.");
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
    setStatus(modelReady ? "Ready." : "Loading model...");
  } catch (err) {
    cameraReady = false;
    updateButtons();
    showLiveView();
    setStatus("Camera access failed. Use upload.");
    console.error(err);
  }
}

async function classifySource(source) {
  if (!modelReady || !model) {
    setStatus("Model not ready.");
    return;
  }
  if (busy) return;
  busy = true;
  updateButtons();
  clearResults();
  setStatus("Capturing...");

  const drawn = drawToCanvas(source, captureCanvas);
  const prepped = drawToSquare(source, preprocessCanvas);
  if (!drawn || !prepped) {
    setStatus("Capture failed.");
    busy = false;
    updateButtons();
    return;
  }

  showCaptureView();
  setStatus("Running inference...");

  try {
    const output = tf.tidy(() => {
      const input = makeInputTensor(preprocessCanvas);
      const result = model.predict(input);
      return Array.isArray(result) ? result[0] : result;
    });

    const probs = await output.data();
    output.dispose();

    const items = Array.from(probs).map((prob, index) => ({
      name: classNames[index] || `Class ${index}`,
      prob,
    }));
    items.sort((a, b) => b.prob - a.prob);
    renderResults(items);
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
    setStatus("Loading model...");
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

clearResults();
loadModelAndClasses();
startCamera();

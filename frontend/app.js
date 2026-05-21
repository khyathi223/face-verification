const tabs = document.querySelectorAll(".tab");
const panes = {
  verify: document.querySelector("#verifyForm"),
  detect: document.querySelector("#detectForm"),
  register: document.querySelector("#registerForm"),
  webcam: document.querySelector("#webcamPane"),
};
const resultText = document.querySelector("#resultText");
const faceCount = document.querySelector("#faceCount");
const processedImage = document.querySelector("#processedImage");
const emptyState = document.querySelector("#emptyState");
const facesList = document.querySelector("#facesList");
const downloadLink = document.querySelector("#downloadLink");
const usersList = document.querySelector("#usersList");
const modelStatus = document.querySelector("#modelStatus");
const video = document.querySelector("#video");
const snapshot = document.querySelector("#snapshot");

let stream = null;
let webcamTimer = null;
let webcamBusy = false;

tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    tabs.forEach((item) => item.classList.remove("active"));
    tab.classList.add("active");
    Object.values(panes).forEach((pane) => pane.classList.remove("active"));
    panes[tab.dataset.tab].classList.add("active");
  });
});

document.querySelector("#verifyForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitForm("/verify", event.currentTarget);
});

document.querySelector("#detectForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitForm("/detect", event.currentTarget);
});

document.querySelector("#registerForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await submitForm("/register", event.currentTarget);
  await loadUsers();
});

document.querySelector("#refreshUsers").addEventListener("click", loadUsers);
document.querySelector("#startCamera").addEventListener("click", startWebcam);
document.querySelector("#stopCamera").addEventListener("click", stopWebcam);

async function submitForm(endpoint, form) {
  const data = new FormData(form);
  setLoading("Processing...");
  try {
    const response = await fetch(endpoint, { method: "POST", body: data });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Request failed.");
    }
    renderResult(payload);
  } catch (error) {
    showError(error.message);
  }
}

function setLoading(message) {
  resultText.textContent = message;
  resultText.classList.remove("error");
}

function showError(message) {
  resultText.textContent = message;
  resultText.classList.add("error");
}

function renderResult(payload) {
  resultText.classList.remove("error");
  resultText.textContent = payload.message || "Done";
  faceCount.textContent = payload.face_count ?? 0;
  facesList.innerHTML = "";

  if (payload.processed_image_url) {
    processedImage.src = `${payload.processed_image_url}?t=${Date.now()}`;
    processedImage.style.display = "block";
    emptyState.style.display = "none";
    downloadLink.href = payload.download_url || payload.processed_image_url;
    downloadLink.style.display = "flex";
  }

  for (const face of payload.faces || []) {
    const item = document.createElement("div");
    item.className = "face-item";
    const label = face.status || "Face";
    const score = face.score === null || face.score === undefined ? "" : `Score ${face.score}`;
    item.innerHTML = `
      <div><strong>${label}</strong><br>${face.matched_name || "No registered match"}</div>
      <div>${Math.round(face.confidence * 100)}% confidence<br>${score}</div>
    `;
    facesList.appendChild(item);
  }
}

async function loadUsers() {
  const response = await fetch("/users");
  const users = await response.json();
  usersList.innerHTML = "";
  if (!users.length) {
    usersList.innerHTML = '<div class="user-item">No registered users yet.</div>';
    return;
  }
  for (const user of users) {
    const item = document.createElement("div");
    item.className = "user-item";
    item.innerHTML = `<strong>${user.name}</strong><span>${user.user_id}</span>`;
    usersList.appendChild(item);
  }
}

async function loadHealth() {
  const response = await fetch("/health");
  const health = await response.json();
  modelStatus.textContent = `Model: ${health.model}`;
}

async function startWebcam() {
  if (stream) return;
  stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
  video.srcObject = stream;
  webcamTimer = window.setInterval(sendWebcamFrame, 1200);
}

function stopWebcam() {
  if (webcamTimer) window.clearInterval(webcamTimer);
  webcamTimer = null;
  if (stream) {
    stream.getTracks().forEach((track) => track.stop());
  }
  stream = null;
  video.srcObject = null;
}

async function sendWebcamFrame() {
  if (!stream || webcamBusy || video.videoWidth === 0) return;
  webcamBusy = true;
  snapshot.width = video.videoWidth;
  snapshot.height = video.videoHeight;
  snapshot.getContext("2d").drawImage(video, 0, 0);
  snapshot.toBlob(async (blob) => {
    try {
      const data = new FormData();
      data.append("image", blob, "webcam.jpg");
      const response = await fetch("/verify", { method: "POST", body: data });
      const payload = await response.json();
      if (response.ok) renderResult(payload);
    } finally {
      webcamBusy = false;
    }
  }, "image/jpeg", 0.9);
}

loadHealth().catch(() => {
  modelStatus.textContent = "Backend unavailable";
});
loadUsers();


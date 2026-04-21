import { execFile } from "child_process";
import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import fs from "fs";
import multer from "multer";
import OpenAI from "openai";
import path from "path";
import { Resend } from "resend";
import { promisify } from "util";

dotenv.config();

const execFileAsync = promisify(execFile);

// ─── Config ───────────────────────────────────────────────────────────────────

const PORT = Number(process.env.PORT || 3001);
const EXEC_MAX_BUFFER = 1024 * 1024 * 50;
const PYTHON_TIMEOUT_MS = 1000 * 60 * 12;
const JOB_TTL_MS = 1000 * 60 * 60;
const POLL_INTERVAL_MS = 2000;

const pythonExePath =
  process.env.PYTHON_PATH ||
  (process.platform === "win32" ? "python" : "python3");

const pythonScriptPath = path.join(process.cwd(), "python", "analyze_run.py");
const uploadsDir = path.join(process.cwd(), "uploads");
const JOB_STORE_FILE = path.join(process.cwd(), "job-store.json");

// ─── Boot Log ─────────────────────────────────────────────────────────────────

console.log("===== SERVER START =====");
console.log("[BOOTED]", new Date().toISOString());
console.log("Port:", PORT);
console.log("Python:", pythonExePath);
console.log("Script:", pythonScriptPath);
console.log("Uploads:", uploadsDir);
console.log("========================");

// ─── OpenAI ───────────────────────────────────────────────────────────────────

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

// ─── Resend ───────────────────────────────────────────────────────────────────

const resend = new Resend(process.env.RESEND_API_KEY);

// ─── Firebase Admin ───────────────────────────────────────────────────────────

let adminAuth = null;
let adminDb = null;

try {
  const { initializeApp, cert, getApps } = await import("firebase-admin/app");
  const { getAuth } = await import("firebase-admin/auth");
  const { getFirestore } = await import("firebase-admin/firestore");

  if (!getApps().length) {
    const serviceAccount = process.env.FIREBASE_SERVICE_ACCOUNT
      ? JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT)
      : null;

    if (serviceAccount) {
      initializeApp({ credential: cert(serviceAccount) });
      adminAuth = getAuth();
      adminDb = getFirestore();
      console.log("[FIREBASE ADMIN] Initialized successfully");
    } else {
      console.warn("[FIREBASE ADMIN] FIREBASE_SERVICE_ACCOUNT env not set — reject deletion will be skipped");
    }
  }
} catch (err) {
  console.warn("[FIREBASE ADMIN] Could not initialize:", err.message);
}

// ─── Express Setup ────────────────────────────────────────────────────────────

const app = express();
app.use(cors());
app.use(express.json({ limit: "2mb" }));

// ─── Uploads Directory ────────────────────────────────────────────────────────

try {
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }
} catch (err) {
  console.error("Failed to create uploads directory:", err);
}

// ─── Multer ───────────────────────────────────────────────────────────────────

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, uploadsDir),
  filename: (_req, file, cb) => {
    const ext = path.extname(file.originalname || ".mp4") || ".mp4";
    cb(null, `${Date.now()}-${Math.random().toString(36).slice(2)}${ext}`);
  },
});

const upload = multer({
  storage,
  limits: { fileSize: 600 * 1024 * 1024 },
  fileFilter: (_req, file, cb) => {
    const mime = String(file.mimetype || "").toLowerCase();
    const ext = String(path.extname(file.originalname || "")).toLowerCase();
    const allowedMime = ["video/mp4", "video/quicktime", "video/x-m4v", "video/mpeg", "video/webm", "video/3gpp", "application/octet-stream"];
    const allowedExt = [".mp4", ".mov", ".m4v", ".mpeg", ".mpg", ".webm", ".3gp"];
    if (allowedMime.includes(mime) || allowedExt.includes(ext)) return cb(null, true);
    cb(new Error("Unsupported video file type."));
  },
});

// ─── JSON Utilities ───────────────────────────────────────────────────────────

function safeParseJson(input, fallback = null) {
  try {
    if (typeof input === "object" && input !== null) return input;
    if (typeof input !== "string" || !input.trim()) return fallback;
    return JSON.parse(input);
  } catch {
    return fallback;
  }
}

function extractLastJsonObject(text) {
  const raw = String(text || "").trim();
  if (!raw) return null;
  try { return JSON.parse(raw); } catch { /* continue */ }
  const end = raw.lastIndexOf("}");
  if (end === -1) return null;
  for (let start = raw.lastIndexOf("{", end); start !== -1; start = raw.lastIndexOf("{", start - 1)) {
    try { return JSON.parse(raw.slice(start, end + 1)); } catch { /* continue */ }
  }
  return null;
}

function parseModelJson(outputText) {
  const cleaned = String(outputText || "")
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  return JSON.parse(cleaned);
}

function preview(text, max = 200) {
  return String(text || "").slice(0, max);
}

// ─── File Cleanup ─────────────────────────────────────────────────────────────

function safeCleanup(paths) {
  for (const targetPath of paths) {
    try {
      if (!targetPath || !fs.existsSync(targetPath)) continue;
      const normalized = path.resolve(targetPath);
      const normalizedUploads = path.resolve(uploadsDir);
      const normalizedCwd = path.resolve(process.cwd());
      if (!normalized.startsWith(normalizedCwd)) {
        console.warn("[CLEANUP] Skipping path outside project:", normalized);
        continue;
      }
      const stats = fs.statSync(normalized);
      if (stats.isDirectory()) {
        fs.rmSync(normalized, { recursive: true, force: true });
      } else if (normalized.startsWith(normalizedUploads) || normalized.includes(`${path.sep}python${path.sep}`)) {
        fs.unlinkSync(normalized);
      } else {
        console.warn("[CLEANUP] Skipping unexpected file:", normalized);
      }
    } catch (err) {
      console.warn("[CLEANUP] Warning:", err.message);
    }
  }
}

function getPythonGeneratedPaths(pythonResult) {
  const paths = [];
  if (!pythonResult || typeof pythonResult !== "object") return paths;
  if (pythonResult.path_map_path) paths.push(pythonResult.path_map_path);
  if (Array.isArray(pythonResult.sampled_frames)) {
    for (const frame of pythonResult.sampled_frames) {
      if (frame?.image_path) paths.push(frame.image_path);
      if (frame?.overlay_image_path) paths.push(frame.overlay_image_path);
    }
  }
  return [...new Set(paths)];
}

// ─── Job Store ────────────────────────────────────────────────────────────────

const jobs = new Map();
const pendingConfirmations = new Map();
const confirmedUsers = new Map();
const rejectedUsers = new Set();
const cleanupTimers = new Map();

// ─── Guardian Store (Firestore-backed) ────────────────────────────────────────

async function persistGuardianStore() {
  if (!adminDb) return;
  try {
    await adminDb.doc("barrel_pro_system/guardian_store").set({
      savedAt: new Date().toISOString(),
      pendingConfirmations: Array.from(pendingConfirmations.entries()).map(([token, record]) => ({ token, ...record })),
      confirmedUsers: Array.from(confirmedUsers.entries()).map(([userId, data]) => ({ userId, ...data })),
      rejectedUsers: Array.from(rejectedUsers),
    });
  } catch (err) {
    console.error("[GUARDIAN PERSIST] Failed:", err.message);
  }
}

async function restoreGuardianStore() {
  if (!adminDb) {
    console.warn("[GUARDIAN RESTORE] Firebase Admin not ready — skipping restore");
    return;
  }
  try {
    const snap = await adminDb.doc("barrel_pro_system/guardian_store").get();
    if (!snap.exists) {
      console.log("[GUARDIAN RESTORE] No stored data found");
      return;
    }
    const data = snap.data();
    const sevenDays = 1000 * 60 * 60 * 24 * 7;

    if (Array.isArray(data.pendingConfirmations)) {
      for (const record of data.pendingConfirmations) {
        if (record.token && Date.now() - record.createdAt < sevenDays) {
          const { token, ...rest } = record;
          pendingConfirmations.set(token, rest);
        }
      }
    }
    if (Array.isArray(data.confirmedUsers)) {
      for (const record of data.confirmedUsers) {
        if (record.userId) {
          const { userId, ...rest } = record;
          confirmedUsers.set(userId, rest);
        }
      }
    }
    if (Array.isArray(data.rejectedUsers)) {
      for (const userId of data.rejectedUsers) {
        rejectedUsers.add(userId);
      }
    }
    console.log("[GUARDIAN RESTORE] pending:", pendingConfirmations.size, "confirmed:", confirmedUsers.size, "rejected:", rejectedUsers.size);
  } catch (err) {
    console.error("[GUARDIAN RESTORE] Failed:", err.message);
  }
}

function persistJobs() {
  try {
    fs.writeFileSync(
      JOB_STORE_FILE,
      JSON.stringify({ savedAt: new Date().toISOString(), jobs: Array.from(jobs.values()) }, null, 2),
      "utf8"
    );
  } catch (err) {
    console.error("[JOB PERSIST] Failed:", err.message);
  }
}

function clearCleanupTimer(jobId) {
  const existing = cleanupTimers.get(jobId);
  if (existing) { clearTimeout(existing); cleanupTimers.delete(jobId); }
}

function deleteJob(jobId) {
  clearCleanupTimer(jobId);
  if (jobs.delete(jobId)) {
    console.log("[JOB DELETE]", jobId);
    persistJobs();
  }
}

function scheduleJobCleanup(jobId) {
  clearCleanupTimer(jobId);
  const job = jobs.get(jobId);
  if (!job) return;
  const delay = Math.max(0, new Date(job.createdAt).getTime() + JOB_TTL_MS - Date.now());
  cleanupTimers.set(jobId, setTimeout(() => deleteJob(jobId), delay));
}

function createJob({ kind, run, videoPath = null }) {
  const jobId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const job = {
    id: jobId, kind, run, videoPath,
    status: "queued", progress: 0, stage: "Queued",
    createdAt: new Date().toISOString(),
    startedAt: null, completedAt: null,
    error: null, result: null,
  };
  jobs.set(jobId, job);
  scheduleJobCleanup(jobId);
  persistJobs();
  console.log("[JOB CREATED]", jobId, "kind:", kind, "total jobs:", jobs.size);
  return job;
}

function updateJob(jobId, updates) {
  const job = jobs.get(jobId);
  if (!job) {
    console.warn("[JOB UPDATE MISSED]", jobId);
    return null;
  }
  Object.assign(job, updates);
  persistJobs();
  console.log("[JOB UPDATE]", jobId, "status:", job.status, "progress:", job.progress, "stage:", job.stage);
  return job;
}

function restoreJobs() {
  try {
    if (!fs.existsSync(JOB_STORE_FILE)) return;
    const parsed = safeParseJson(fs.readFileSync(JOB_STORE_FILE, "utf8"));
    if (!parsed?.jobs || !Array.isArray(parsed.jobs)) return;

    let restored = 0;
    for (const job of parsed.jobs) {
      if (!job?.id || !job?.createdAt) continue;
      if (Date.now() - new Date(job.createdAt).getTime() >= JOB_TTL_MS) continue;

      const restoredJob = { ...job };
      if (restoredJob.status === "queued" || restoredJob.status === "running") {
        restoredJob.status = "failed";
        restoredJob.stage = "Failed";
        restoredJob.completedAt = new Date().toISOString();
        restoredJob.error = "Server restarted while analysis was running. Please re-analyze.";
      }

      jobs.set(restoredJob.id, restoredJob);
      scheduleJobCleanup(restoredJob.id);
      restored++;
    }

    console.log("[JOB RESTORE] restored:", restored);
    persistJobs();
  } catch (err) {
    console.error("[JOB RESTORE] Failed:", err.message);
  }
}

// ─── Python Runner ────────────────────────────────────────────────────────────

async function runPythonAnalysis(videoPath, run = {}) {
  console.log("[PYTHON] Starting analysis:", videoPath);

  try {
    const { stdout, stderr } = await execFileAsync(
      pythonExePath,
      [pythonScriptPath, videoPath, JSON.stringify(run)],
      {
        maxBuffer: EXEC_MAX_BUFFER,
        timeout: PYTHON_TIMEOUT_MS,
        env: { ...process.env, PYTHONUNBUFFERED: "1", YOLO_CONFIG_DIR: "/tmp/Ultralytics" },
      }
    );

    if (stderr?.trim()) console.warn("[PYTHON STDERR]", preview(stderr, 400));
    console.log("[PYTHON] stdout length:", String(stdout || "").length);

    const result = safeParseJson(stdout) || extractLastJsonObject(stdout);
    if (!result) {
      console.error("[PYTHON] Invalid output:", preview(stdout));
      throw new Error("Python returned invalid JSON.");
    }
    if (!result.ok) throw new Error(result.error || "Python analysis failed.");

    return result;
  } catch (err) {
    if (err?.stdout) {
      const recovered = safeParseJson(err.stdout) || extractLastJsonObject(err.stdout);
      if (recovered?.ok) return recovered;
      if (recovered) throw new Error(recovered.error || "Python analysis failed.");
    }

    if (err?.killed || err?.signal === "SIGTERM") throw new Error("Python analysis timed out.");
    if (err?.code === "ETIMEDOUT") throw new Error("Python analysis timed out.");
    throw err;
  }
}

// ─── Frame Selection ──────────────────────────────────────────────────────────
// Smart strategic frame selection — 30 frames total.
// If rider set manual splits, use split timestamps to anchor zones precisely.
// If no splits, use percentage-based zones (no alley — wasted frames).
//
// When splits are available:
//   - videoDuration from pythonResult tells us total video length in seconds
//   - barrel_video_timestamps (b1, b2, b3) are video positions when rider tapped each barrel
//   - We build zones centered on those exact timestamps
//
// The barrel_video_timestamps also tell CV exactly WHERE each barrel is in the video timeline,
// which is passed to Python for more accurate barrel detection anchoring.

function selectFramePaths(sampledFrames, maxFrames = 30, run = null, videoDuration = null) {
  const usable = (sampledFrames || [])
    .filter((f) => f?.read_success && (f?.overlay_image_path || f?.image_path))
    .map((f) => ({
      path: f.overlay_image_path || f.image_path,
      percent: f.percent ?? 0,
      timeSec: f.time_sec ?? null,
      dense: !!f.dense_pass,
    }))
    .filter((f) => f.path);

  if (usable.length === 0) return [];
  if (usable.length <= maxFrames) return usable.map((f) => f.path);

  // ── SPLIT-AWARE SELECTION ─────────────────────────────────────────────────
  // If rider set manual splits AND we have barrel video timestamps, build zones
  // anchored to the exact video positions where each barrel was reached.
  const barrelTimestamps = run?.barrel_location_hints || run?.manualSplits?.barrel_video_timestamps || null;
  const splits = run?.manualSplits || null;
  const totalDuration = videoDuration || null;

  if (splits && barrelTimestamps && totalDuration && totalDuration > 0) {
    const b1t = barrelTimestamps.barrel1; // video seconds when rider tapped B1
    const b2t = barrelTimestamps.barrel2;
    const b3t = barrelTimestamps.barrel3;
    const startT = splits.start_to_barrel1_seconds != null ? 0 : null; // run starts at 0
    const finishT = totalDuration;

    // Convert video timestamp → percent of total video
    const toPercent = (t) => Math.max(0, Math.min(1, t / totalDuration));
    const window = (t, before, after) => ({ min: toPercent(t - before), max: toPercent(t + after) });

    // Build split-anchored zones — no alley, focused on the action
    // START of run = after rider clicked START (skip pre-run)
    const runStartPercent = splits.start_to_barrel1_seconds != null
      ? Math.max(0, toPercent((b1t || 0) - splits.start_to_barrel1_seconds - 1))
      : 0.10; // fallback: skip first 10% (alley)

    const zones = [];

    // Pre-run approach (from START click to B1 approach) — 2 frames
    if (b1t != null) {
      zones.push({ min: runStartPercent, max: toPercent(b1t - 1.5), count: 2, label: "run start→B1 approach" });
      // B1 apex — centered on exact timestamp, ±1.5s — 5 frames
      zones.push({ ...window(b1t, 1.5, 1.5), count: 5, label: "B1 apex" });
      // B1 exit — 1.5s after B1 — 2 frames
      zones.push({ ...window(b1t + 1.5, 0, 1.5), count: 2, label: "B1 exit" });
    }

    if (b2t != null) {
      // B2 approach — between B1 exit and B2 — 3 frames
      const b2ApproachMin = b1t != null ? toPercent(b1t + 2.5) : runStartPercent;
      zones.push({ min: b2ApproachMin, max: toPercent(b2t - 1.0), count: 3, label: "B2 approach" });
      // B2 apex — 5 frames
      zones.push({ ...window(b2t, 1.5, 1.5), count: 5, label: "B2 apex" });
      // B2 exit — 2 frames
      zones.push({ ...window(b2t + 1.5, 0, 1.5), count: 2, label: "B2 exit" });
    }

    if (b3t != null) {
      // B3 approach — 3 frames
      const b3ApproachMin = b2t != null ? toPercent(b2t + 2.5) : runStartPercent;
      zones.push({ min: b3ApproachMin, max: toPercent(b3t - 1.0), count: 3, label: "B3 approach" });
      // B3 apex — 5 frames
      zones.push({ ...window(b3t, 1.5, 1.5), count: 5, label: "B3 apex" });
      // B3 exit + home run — 3 frames
      zones.push({ ...window(b3t + 1.5, 0, toPercent(finishT) - toPercent(b3t + 1.5)), count: 3, label: "B3 exit→home" });
    }

    console.log(`[FRAMES] Using SPLIT-ANCHORED selection — ${zones.length} zones from barrel timestamps`);

    const selected = new Set();
    for (const zone of zones) {
      if (zone.min >= zone.max) continue;
      const inZone = usable.filter((f) => f.percent >= zone.min && f.percent <= zone.max);
      if (inZone.length === 0) continue;
      const dense = inZone.filter((f) => f.dense);
      const pool = dense.length > 0 ? dense : inZone;
      const count = Math.min(zone.count, pool.length);
      for (let i = 0; i < count; i++) {
        const idx = Math.round((i * (pool.length - 1)) / Math.max(count - 1, 1));
        selected.add(pool[idx].path);
      }
    }

    // Fill any remaining slots evenly from non-alley portion of video
    if (selected.size < maxFrames) {
      const remaining = maxFrames - selected.size;
      const nonAlley = usable.filter((f) => f.percent >= runStartPercent);
      for (let i = 0; i < remaining; i++) {
        const idx = Math.round((i * (nonAlley.length - 1)) / Math.max(remaining - 1, 1));
        selected.add(nonAlley[idx].path);
      }
    }

    console.log(`[FRAMES] Split-anchored: ${selected.size} frames selected`);
    return [...selected].slice(0, maxFrames);
  }

  // ── PERCENTAGE-BASED FALLBACK (no splits) ────────────────────────────────
  // No alley — skip first 12% of video (pre-run approach).
  // Redistribute those 2 frames to barrels instead.
  console.log(`[FRAMES] Using PERCENTAGE-BASED selection (no manual splits)`);

  const zones = [
    { min: 0.12, max: 0.25, count: 3 },  // barrel 1 approach (skip alley)
    { min: 0.23, max: 0.36, count: 5 },  // barrel 1 apex
    { min: 0.34, max: 0.43, count: 3 },  // barrel 1 exit
    { min: 0.40, max: 0.52, count: 3 },  // barrel 2 approach
    { min: 0.50, max: 0.62, count: 5 },  // barrel 2 apex
    { min: 0.60, max: 0.68, count: 3 },  // barrel 2 exit
    { min: 0.65, max: 0.76, count: 3 },  // barrel 3 approach
    { min: 0.74, max: 0.84, count: 5 },  // barrel 3 apex
    { min: 0.82, max: 1.00, count: 3 },  // barrel 3 exit + home run
  ];

  // Total allocation = 33 slots for 30 frames — zones will compete for frames naturally

  const selected = new Set();
  for (const zone of zones) {
    const inZone = usable.filter((f) => f.percent >= zone.min && f.percent <= zone.max);
    if (inZone.length === 0) continue;
    const dense = inZone.filter((f) => f.dense);
    const pool = dense.length > 0 ? dense : inZone;
    const count = Math.min(zone.count, pool.length);
    for (let i = 0; i < count; i++) {
      const idx = Math.round((i * (pool.length - 1)) / Math.max(count - 1, 1));
      selected.add(pool[idx].path);
    }
  }

  // Fill remaining slots from non-alley frames
  if (selected.size < maxFrames) {
    const remaining = maxFrames - selected.size;
    const nonAlley = usable.filter((f) => f.percent >= 0.12);
    for (let i = 0; i < remaining; i++) {
      const idx = Math.round((i * (nonAlley.length - 1)) / Math.max(remaining - 1, 1));
      selected.add(nonAlley[idx].path);
    }
  }

  return [...selected].slice(0, maxFrames);
}

function buildImageInputs(framePaths) {
  // Read and encode frames one at a time, immediately discarding the buffer
  // after encoding to prevent OOM on Render Standard (2GB RAM limit).
  // GPT-4o "low" detail mode processes images at 512x512 equivalent —
  // sending full 1080p frames wastes memory with no quality benefit.
  const inputs = [];
  let totalBytes = 0;

  for (const p of framePaths) {
    try {
      const buf = fs.readFileSync(p);
      totalBytes += buf.length;

      // Hard stop if we are approaching memory limits
      // Each base64 string is ~1.33x the original file size
      if (totalBytes > 150 * 1024 * 1024) { // 150MB raw = ~200MB base64
        console.warn(`[FRAMES] Memory limit reached at frame ${inputs.length} — stopping early`);
        break;
      }

      const b64 = buf.toString("base64");
      inputs.push({
        type: "image_url",
        image_url: {
          url: `data:image/jpeg;base64,${b64}`,
          detail: "high",
        },
      });
    } catch (err) {
      console.warn("[FRAMES] Could not read frame:", p, err.message);
    }
  }

  console.log(`[FRAMES] Loaded ${inputs.length}/${framePaths.length} frames — ${Math.round(totalBytes/1024/1024)}MB raw`);
  return inputs;
}

// ─── Historical Context Builder ───────────────────────────────────────────────

function buildHistoricalContext(run) {
  const history = run?.runHistory || [];
  const horseName = run?.horse || "this horse";

  if (!history || history.length === 0) {
    return `No previous run history available for ${horseName}. This appears to be the first logged run.`;
  }

  const horseRuns = history.filter(r => r.horse === horseName && r.time && !isNaN(parseFloat(r.time)));
  
  if (horseRuns.length === 0) {
    return `No previous timed runs found for ${horseName}.`;
  }

  const times = horseRuns.map(r => parseFloat(r.time)).filter(t => !isNaN(t));
  const currentTime = parseFloat(run?.time);
  
  const best = times.length > 0 ? Math.min(...times) : null;
  const avg = times.length > 0 ? (times.reduce((a, b) => a + b, 0) / times.length).toFixed(3) : null;
  const worst = times.length > 0 ? Math.max(...times) : null;

  // Trend: compare last 3 runs to previous 3
  const recent3 = times.slice(0, 3);
  const prior3 = times.slice(3, 6);
  let trend = "insufficient data for trend";
  if (recent3.length >= 2 && prior3.length >= 2) {
    const recentAvg = recent3.reduce((a, b) => a + b, 0) / recent3.length;
    const priorAvg = prior3.reduce((a, b) => a + b, 0) / prior3.length;
    const diff = (recentAvg - priorAvg).toFixed(3);
    if (recentAvg < priorAvg) trend = `improving (${Math.abs(diff)}s faster recently)`;
    else if (recentAvg > priorAvg) trend = `slower recently (${diff}s off pace)`;
    else trend = "consistent";
  }

  // Compare this run to personal best
  let vsPersonalBest = "";
  if (best && !isNaN(currentTime)) {
    const diff = (currentTime - best).toFixed(3);
    if (diff <= 0) vsPersonalBest = `This is a NEW PERSONAL BEST by ${Math.abs(diff)}s!`;
    else vsPersonalBest = `This run is ${diff}s off the personal best.`;
  }

  // Arena condition history
  const arenaHistory = horseRuns
    .filter(r => r.arenaCondition)
    .slice(0, 5)
    .map(r => `${r.arenaCondition}: ${r.time}s`)
    .join(", ");

  // Recent show results
  const recentShows = horseRuns
    .filter(r => r.showName)
    .slice(0, 5)
    .map(r => {
      const parts = [`${r.showName}: ${r.time}s`];
      if (r.placing) parts.push(`placed ${r.placing}`);
      if (r.earnings) parts.push(`earned $${r.earnings}`);
      return parts.join(", ");
    })
    .join(" | ");

  // Rider notes from recent runs — gold for the AI
  const recentNotes = horseRuns
    .filter(r => r.riderFeedback || r.notes)
    .slice(0, 4)
    .map(r => {
      const parts = [];
      if (r.riderFeedback) parts.push(`Rider felt: "${r.riderFeedback}"`);
      if (r.notes) parts.push(`Notes: "${r.notes}"`);
      return parts.join(" | ");
    })
    .join("\n  ");

  return `
HORSE HISTORY FOR ${horseName.toUpperCase()} (${horseRuns.length} logged runs):
- Personal best time: ${best ? best + "s" : "unknown"}
- Average time: ${avg ? avg + "s" : "unknown"}
- Worst time: ${worst ? worst + "s" : "unknown"}
- Performance trend: ${trend}
- ${vsPersonalBest}

Arena condition performance:
  ${arenaHistory || "No arena condition data available"}

Recent show results:
  ${recentShows || "No show data available"}

Recent rider feedback and notes:
  ${recentNotes || "No recent notes available"}
  `.trim();
}

// ─── AI Prompts ───────────────────────────────────────────────────────────────

// ── Barrel Racing Knowledge Base ─────────────────────────────────────────────
// Expert coaching knowledge injected into every prompt so the AI coaches
// like a professional with deep barrel racing expertise.

// ─── Arena Distance Normalization ────────────────────────────────────────────
// Default: WPRA Pattern A distances. User can override with their arena's
// actual measurements for accurate ft/sec calculations.

const PATTERN_A_DISTANCES = {
  "Start to 1st Barrel":   60,
  "1st to 2nd Barrel":     90,
  "2nd to 3rd Barrel":    105,
  "3rd Barrel to Finish": 120,
};

function getArenaDistances(run) {
  // Use user-entered distances if provided, otherwise Pattern A defaults
  const d = run?.arenaDistances;
  if (d && d.startToB1 && d.b1ToB2 && d.b2ToB3 && d.b3ToFinish) {
    return {
      "Start to 1st Barrel":   Number(d.startToB1),
      "1st to 2nd Barrel":     Number(d.b1ToB2),
      "2nd to 3rd Barrel":     Number(d.b2ToB3),
      "3rd Barrel to Finish":  Number(d.b3ToFinish),
    };
  }
  return PATTERN_A_DISTANCES;
}

function normalizeSplitsToSpeed(splits, distances) {
  // splits = { label: seconds }, distances = { label: feet }
  // returns array sorted slowest→fastest by ft/s
  const dist = distances || PATTERN_A_DISTANCES;
  const results = [];
  for (const [label, seconds] of Object.entries(splits)) {
    const distance = dist[label];
    if (distance == null || !Number.isFinite(Number(seconds)) || Number(seconds) <= 0) continue;
    const speed = distance / Number(seconds);
    results.push({ label, seconds: Number(seconds), distance, speed: parseFloat(speed.toFixed(2)) });
  }
  results.sort((a, b) => a.speed - b.speed);
  return results;
}

const BARREL_RACING_KNOWLEDGE_BASE = `
=== BARREL RACING EXPERT COACHING KNOWLEDGE ===
You are an elite barrel racing coach. Apply this knowledge to every analysis.

── CORE PHILOSOPHY ────────────────────────────────────────────────────────────
"Speed is a byproduct of correctness. If you fix the mechanics, the clock will follow."
"A great run is a conversation, not a fight. The best riders do 90% of their work in the training pen so that during the 15-second run, they can simply stay out of the horse's way."
"In barrel racing, speed is the goal, but precision is the engine."
The straightaways are for speed. The turns are for precision. If you don't set up the approach, you've lost the turn before you even get there.
The "go fast to be fast" trap: Many riders push for raw speed, which results in sloppy lines and blown turns. To clock in the 1D, you have to stop fighting the clock and start working with your horse's mechanics. Consistency beats intensity every single day.
The turn doesn't start AT the barrel — it starts 15 feet before it. Every single time.

── THE APPROACH & THE POCKET ──────────────────────────────────────────────────
Source: "Phase 1 — The Anatomy of the Approach"
Core principle: "The straightaways are for speed, but the turns are for the check. If you don't set up the approach, you've lost the turn before you even get there."
The system is built around "The Pocket."

THE FIRST BARREL — "The Money Barrel"
Highest speed entry of the run. Sets up everything that follows.
Use a "J" approach — aim for a point 5 to 8 feet to the side of the barrel. Never run straight at it. Running straight forces a hard check that kills all momentum.
The Line: Run slightly to the outside of the barrel. Your path should create a smooth arc into the turn, not a sharp hook.
The Pocket: The horse needs enough room to arc its ribcage cleanly through the turn. Too tight = hind end clips the backside of the barrel.

THE SECOND AND THIRD BARRELS — Cross-Firing Your Vision
These require "cross-firing" your vision.
As you leave the first barrel, you are NOT looking at the second barrel. You are looking at the ENTRY POINT of the second barrel — where you want the horse's feet to go on approach.
"Don't look at the barrel. If you look at the barrel, you'll hit it. Look where you want the horse's feet to go."
This applies leaving every barrel — eyes to the entry point of the next, not the barrel itself.

── THE RATE POINT ─────────────────────────────────────────────────────────────
Rate = shifting weight from the front end to the hocks to prepare for the turn.
Cue: As the horse's nose reaches the barrel, "sit deep in your pockets." Sink your weight into your seat bones. This is the physical cue — not a verbal cue, not a rein cue — for the horse to shift its weight to its hocks.
Stop driving with your legs at the rate point. Use a slight command rein to ask the horse to break at the poll and gather its stride.
Rate too early = lose speed unnecessarily. Rate too late = "going long" — overshooting the turn, having to hook back, losing both time and line.
Fix for running past: Stop the horse at the rate point every practice run until they learn to wait for the seat cue.

── THE TURN ───────────────────────────────────────────────────────────────────
Keep the horse's inside shoulder UP. A dropped shoulder causes slicing, loss of arc, and knocked barrels.
The horse should bend around your inside leg — ribcage arc ensures hind feet follow front feet in a single track for maximum traction.
Do NOT commit to the turn until your leg (cinch area) has passed the barrel.
At the 3/4 mark of the turn, snap your eyes to the next barrel's entry point — this leads the horse's momentum out of the turn naturally.
Finish the turn completely. Leaving early causes a poor line to the next barrel and loses the momentum built through the apex.

── RIDER HAND POSITION ────────────────────────────────────────────────────────
Source: "Phase 2 — Rider Body Mechanics. Your horse is a mirror. If you are unbalanced, they are unbalanced. If you are tense, they are stiff."

Keep hands LOW at all times. High hands = high-headed horse. A high-headed horse cannot see the ground or turn efficiently.
Inside Hand: Guides the nose. Creates a soft arc into the turn — never a sharp pull. Pulling hard on the inside rein pulls the horse off-balance and prevents hindquarter power through the turn.
Outside Hand: "The Wall." This hand keeps the horse from drifting out and keeps the inside shoulder upright. Often more important than the inside hand.
Never "hunt" for the horn until AFTER you have sat deep in your seat. Grabbing the horn too early pulls your chest forward, which tips the horse's weight onto its front end — exactly the opposite of what you need at the rate point.
One-handed principle: Barrel racing is more about legs and seat than steering. If the horse won't respond without heavy hands, return to lateral groundwork — side-passing, leg-yielding — before returning to the pattern.

── RIDER SEAT & BODY POSITION ─────────────────────────────────────────────────
Source: "Phase 2 — The Proper Seat"

Straightaways: Stand slightly in your stirrups (two-point position). Take weight off the horse's back, allowing them to fully extend their stride. Do not sit and drive on the straightaway — let the horse run.
The Rate: As the horse's nose reaches the barrel, sit DEEP in your pockets. Sink your weight into your seat bones. This IS the physical cue for the horse to shift weight to its hocks. This is not just balance — it is communication.
Through the turn: Spine aligned with horse's spine. Do NOT lean (motorcycle lean) — shifting weight to the inside shoulder causes diving, slicing, and knocked barrels.
After the turn: Immediately forward and athletic. Push the horse to the next spot. Getting left behind (leaning back during acceleration) effectively hits the horse in the mouth through rein tension and kills the drive to the next barrel.

── COMMON FAULTS & EXACT FIXES ────────────────────────────────────────────────
Source: "Championship Barrel Racing — The Professional Coach's Guide to Corrective Training"
Core principle: "Speed is a byproduct of correctness. If you fix the mechanics, the clock will follow."

1. DIVING INTO THE TURN (Pocket Killer #1)
The #1 pocket killer. Riders lean their body or pull the horse's nose toward the barrel too early, causing the horse to "shoulder in" — hitting the barrel or losing momentum entirely.
What it looks like: Rider leans inward before leg reaches the barrel. Horse's inside shoulder drops and they cut the turn short. Horse may clip barrel with shoulder or rider's knee.
Fix drill: "Square the Barrel" —
  Execution: Approach the barrel at a TROT. Instead of riding a circle, ride a literal square around the barrel.
  The Focus: Ride past your "point" (where you start your turn) and keep the horse's shoulders upright. Only begin the turn when you can see the BACKSIDE of the barrel.
  Why it works: The square forces the rider to wait — you physically cannot turn early when riding a square. The horse learns to hold their shoulders until released by the rider's cue.
  Progression: Start at a trot, then slow lope, then full speed once the habit is built.

2. LOOKING AT THE BARREL
It's human nature to look at what you're trying to avoid — but in barrel racing, your horse follows your eyes. If you stare at the barrel, your body weight shifts, your shoulder drops, and the horse follows directly into a collision.
What it looks like: Rider's head drops or turns toward barrel on approach. Body weight tips inside. Shoulder drops. Horse drifts toward the barrel.
Fix drill: "The Look Ahead / Horizon Drill" —
  Execution: While working at a controlled lope, focus on a spot about 10 feet PAST each barrel as you approach it.
  The Focus: Once you are in the turn, snap your eyes immediately to the next destination — the next barrel's entry point, or the timer line if it's the last barrel. By looking where you want to go, your hips and shoulders naturally align there.
  Key cue: Don't look at the next barrel until your horse's hip has cleared the current one. Looking too early pulls you out of the turn prematurely.

3. SHOULDERS DROPPING ("Washing Out")
Allowing the horse to drop their inside shoulder, losing leverage and power in the turn. The horse loses its arc, risking a knockdown or missed apex.
What it looks like: Horse's inside shoulder visibly drops mid-turn. Hind end swings out. Turn loses arc and traction.
Fix drill: "Counter-Bending Circles" — Circle the barrel while bending the horse's nose AWAY from the barrel. This lifts the inside shoulder and engages the hindquarters. Repeat until horse can hold the lift through the entire circle.

4. OVER-HANDLING / SAWING ON THE BIT
Many riders "saw" on the bit or pull the inside rein throughout the entire turn, pulling the horse off-balance and preventing them from using their hind end to power out of the turn.
What it looks like: Rider's hands active and pulling throughout the turn. Horse's head elevated, jaw tight, movement choppy and broken. Horse cannot drive from hindquarters because front end is being held.
Fix drill A: "The One-Handed Guide" —
  Execution: Set up the standard pattern and work it at a trot or slow lope using ONLY your dominant riding hand on the reins.
  The Focus: Use your legs and seat to guide the horse's ribs and hips through each turn. This forces you to use your body to steer rather than your hands.
  If the horse won't turn without a heavy hand: Return to basic lateral work — side-passing and leg yielding — before returning to the pattern. The problem is a training gap, not a speed problem.
Fix drill B: "Loose Rein Loping" —
  Execution: Lope a large circle around the barrel on a completely loose rein, using only weight and legs.
  Reward the horse for holding the path. This resets the horse to respond to seat cues, not hand cues.

5. FAILING TO FINISH THE TURN (Early Exit)
Leaving the barrel too early, resulting in a compromised line to the next barrel.
What it looks like: Horse and rider peel away from the barrel before the turn is complete. Next approach is compromised because the turn was not finished.
Fix drill: "The One-and-a-Half" — Make a full circle around the barrel PLUS another half-turn before heading to the next. This teaches the horse to keep turning until specifically told to leave. Builds patience and finish through the turn.

6. IMPROPER POCKET — RUNNING TOO TIGHT
Horse clips the barrel with shoulder, hip, or hind end. Often caused by diving too early, dropped inside shoulder, or poor approach angle.
What it looks like: Horse brushes or knocks the barrel. Hind end swings into the backside.
Fix drill: "The Pinwheel" — Set up 4 cones around a barrel at 5-foot intervals. Practice spiraling in and out at a trot to master barrel awareness.

7. GETTING AHEAD OF THE HORSE
Leaning forward over the neck before the horse has finished the turn, causing them to stumble or lose hind-end engagement.
What it looks like: Rider's weight tips forward mid-turn or at exit. Horse's front end heavy, hindquarters disengage. Horse loses drive and forward momentum out of the turn.
Fix drill: "The Deep Sit Stop" — Lope the pattern and ask for a complete stop at the backside of every barrel. You must sit DEEP in your pockets to cue the stop. This trains both horse and rider to maintain hindquarter engagement through and past the apex.

8. LACK OF RATE (Running Past)
The horse "blows" past the barrel because they didn't gear down or prepare for the turn.
What it looks like: Horse approaches at full speed with no collection. Has to hook back after passing barrel or knocks it going past.
Fix drill: "Transition Points" — Pick a point 15 feet before the barrel. Every time you hit that point, you MUST transition from a lope to a trot. This builds the "rate" muscle memory. Once consistent, move the transition point back gradually until the horse rates off your seat alone.

9. SHOULDERING IN
Horse leans into barrel, knocks it with shoulder or rider's knee.
What it looks like: Horse's body angle wrong — ribcage pushes into the barrel rather than arcing around it.
Fix: More inside leg at the girth, more outside rein. Keep horse "square" until the actual turning point.
Fix drill: "Counter-Bending Circles" (same as fault #3 — this drill addresses both shoulder drop and shouldering in).

10. INCONSISTENT ALLEYWAY BEHAVIOR
Fighting in the alley causes stressed entry, poor alignment, and a rushed approach to first barrel.
What it looks like: Horse jigging, spinning, refusing to stand in alley. Rider tense. Entry rushed and crooked.
Fix drill: "Quiet Entry" — Walk into alley, stop, back up, sit quietly until horse exhales, then walk out calmly. Repeat. Alley must represent calm, not launch. A stressed alley entry ruins the approach to first barrel every time.

── FAULT DIAGNOSIS FROM SPLIT DATA ────────────────────────────────────────────
When you see a slow split, connect it to the specific fault it most likely represents:
Slow alley→1st: Alley stress, wrong approach angle, failure to rate at the right point, or diving too early.
Slow 1st→2nd or 2nd→3rd: Horse not driving between barrels. Rider getting left behind (leaning back during acceleration). Horse still in rate mode — not extending on the straightaway. Early exit from previous barrel causing bad line.
Slow 3rd→home: Rider not in two-point pushing forward. Horse not rated out cleanly. Fatigue. Or rider mentally "done" before the run is.
One split dramatically slower than others: Problem is SPECIFIC to that barrel. Examine that barrel's turn grade, approach angle, and exit drive data closely.
All splits slow but consistent: Horse not extending on straightaways, OR rider restraining horse through entire run rather than galloping between barrels.

── SPLIT TIME INTERPRETATION ──────────────────────────────────────────────────
Slow alley-to-first: Late to rate, approach angle too straight, wrong rate point, or alley stress.
Slow first-to-second or second-to-third: Horse not driving between barrels. Rider sitting back (getting left behind), horse still in rate mode, not extending on the straightaway.
Slow third-to-home: Horse not rated out cleanly, tired, or rider not in two-point pushing forward.
One slow split vs others: Problem is specific to that barrel — check turn grade, approach angle, and exit drive data for that barrel specifically.
A run that feels fast but clocks slow = time is being lost in the turns or straightaways — check exit drive and approach efficiency.

── HORSE BODY MECHANICS ───────────────────────────────────────────────────────
Inside shoulder dropped = slicing the turn, loss of arc, likely knockdown.
High head/nose out = on the forehand, cannot collect or turn efficiently.
Breaking at poll with soft jaw = correctly rated and collected.
Ribcage bending around inside leg = single-track movement, maximum traction.
Exit drive = horse pushes hard from hindquarters within 2-3 strides of apex. Drifting or coasting out = losing significant time.
A horse that anticipates the rate and shuts down early = over-trained rate, needs more forward work.
`;

function buildBarrelCoachingData(run, pythonResult) {
  const barrelMetrics = pythonResult?.barrel_metrics || {};
  const speedSummary = pythonResult?.speed_summary || null;
  const splits = pythonResult?.splits || {};
  const insights = Array.isArray(pythonResult?.insights) ? pythonResult.insights : [];
  const barrelLabels = { barrel1: "First", barrel2: "Second", barrel3: "Third" };
  const barrelTimestamps = run?.barrel_location_hints || run?.manualSplits?.barrel_video_timestamps || null;

  const barrelReport = ["barrel1", "barrel2", "barrel3"].map(name => {
    const bm = barrelMetrics[name];
    const label = barrelLabels[name];
    if (!bm || !bm.detected) return `${label} barrel: not detected in video`;
    const tightness = bm.turn_tightness || {};
    const approach = bm.approach || {};
    const exitDrive = bm.exit_drive || null;
    const knocked = bm.potential_knockdown;
    const lines = [`${label} barrel:`];
    if (tightness.grade) lines.push(`  - Turn tightness: Grade ${tightness.grade} (${tightness.label}) — ${tightness.coaching_note}`);
    if (tightness.min_distance_px !== null && tightness.min_distance_px !== undefined) lines.push(`  - Closest approach: ${tightness.min_distance_px}px from barrel center`);
    if (approach.angle_degrees !== null && approach.angle_degrees !== undefined) lines.push(`  - Approach angle: ${approach.angle_degrees}° (ideal: 20-40°) — ${approach.coaching_note}`);
    if (exitDrive) {
      lines.push(`  - Exit drive: ${exitDrive.coaching_note}`);
      if (exitDrive.apex_speed_px_per_sec && exitDrive.exit_speed_px_per_sec) {
        const ratio = exitDrive.acceleration_ratio;
        lines.push(`  - Speed at turn: ${exitDrive.apex_speed_px_per_sec}px/s → exit: ${exitDrive.exit_speed_px_per_sec}px/s (${ratio >= 1.0 ? "+" : ""}${((ratio - 1) * 100).toFixed(0)}%)`);
      }
    }
    if (knocked) lines.push(`  - ⚠️ POSSIBLE KNOCKDOWN DETECTED (confidence: ${Math.round((bm.knockdown_confidence || 0) * 100)}%)`);
    if (bm.summary_tags && bm.summary_tags.length > 0) lines.push(`  - Tags: ${bm.summary_tags.join(", ")}`);
    return lines.join("\n");
  }).join("\n\n");

  let speedReport = "Speed data: not available";
  if (speedSummary) {
    const lines = ["Run speed profile:"];
    if (speedSummary.slowest_section_label) lines.push(`  - SLOWEST section: ${speedSummary.slowest_section_label}`);
    if (speedSummary.fastest_section_label) lines.push(`  - Fastest section: ${speedSummary.fastest_section_label}`);
    if (speedSummary.section_speeds) {
      const s = speedSummary.section_speeds;
      lines.push(`  - Alley→1st: ${s.alley_to_barrel1 ?? "n/a"}px/s | 1st→2nd: ${s.barrel1_to_barrel2 ?? "n/a"}px/s | 2nd→3rd: ${s.barrel2_to_barrel3 ?? "n/a"}px/s | 3rd→Home: ${s.barrel3_to_home ?? "n/a"}px/s`);
    }
    speedReport = lines.join("\n");
  }

  // Always prefer manual splits (already scaled to official time) over CV splits
  // CV splits are raw video estimates — manual splits are ground truth
  const manualSplits = run?.manualSplits || null;
  const activeSplits = manualSplits || splits || {};
  const splitsSource = manualSplits ? "user-marked and scaled to official time" : "CV-estimated";

  // Scale CV splits to official time if no manual splits available
  let s1 = activeSplits?.start_to_barrel1_seconds ?? null;
  let s2 = activeSplits?.barrel1_to_barrel2_seconds ?? null;
  let s3 = activeSplits?.barrel2_to_barrel3_seconds ?? null;
  let s4 = activeSplits?.barrel3_to_home_seconds ?? null;

  // If using CV splits, scale them to the official run time
  if (!manualSplits && run?.time) {
    const officialTime = parseFloat(run.time);
    const cvTotal = [s1, s2, s3, s4].reduce((sum, v) => sum + (Number.isFinite(Number(v)) ? Number(v) : 0), 0);
    if (cvTotal > 0 && officialTime > 0 && Math.abs(cvTotal - officialTime) > 0.1) {
      const scale = officialTime / cvTotal;
      if (s1 != null) s1 = parseFloat((Number(s1) * scale).toFixed(2));
      if (s2 != null) s2 = parseFloat((Number(s2) * scale).toFixed(2));
      if (s3 != null) s3 = parseFloat((Number(s3) * scale).toFixed(2));
      if (s4 != null) s4 = parseFloat((Number(s4) * scale).toFixed(2));
    }
  }

  // Normalize to ft/s using Pattern A distances
  const cvNormalizedMap = {
    "Start to 1st Barrel":   s1,
    "1st to 2nd Barrel":     s2,
    "2nd to 3rd Barrel":     s3,
    "3rd Barrel to Finish":  s4,
  };
  const cvArenaDist = getArenaDistances(run);
  const cvNormalized = normalizeSplitsToSpeed(
    Object.fromEntries(Object.entries(cvNormalizedMap).filter(([,v]) => v != null && Number.isFinite(Number(v)))),
    cvArenaDist
  );
  const slowestBySpeed = cvNormalized[0] || null;
  const fastestBySpeed = cvNormalized[cvNormalized.length - 1] || null;

  const splitReport = `Split times by speed — ${splitsSource} (Pattern A distances):
${cvNormalized.map(s => `  - ${s.label}: ${s.seconds.toFixed(2)}s / ${s.distance}ft = ${s.speed.toFixed(1)} ft/s`).join("\n")}
${slowestBySpeed ? `  ► SLOWEST by speed: ${slowestBySpeed.label} at ${slowestBySpeed.speed.toFixed(1)} ft/s` : ""}
${fastestBySpeed ? `  ► FASTEST by speed: ${fastestBySpeed.label} at ${fastestBySpeed.speed.toFixed(1)} ft/s` : ""}`;

  return `
=== COMPUTER VISION COACHING DATA ===

Run overview:
- Duration: ${pythonResult?.duration_seconds ?? "unknown"}s | Pattern: ${pythonResult?.pattern_direction ?? "unknown"}-first
- Horse tracked: ${pythonResult?.horse_detected_frames ?? "?"} frames | Frames analyzed: ${pythonResult?.tracking_quality?.sampled_frame_count ?? "?"}

${splitReport}

${speedReport}

Per-barrel coaching data:
${barrelReport}

CV insights:
${insights.length ? insights.map(i => `- ${i}`).join("\n") : "- None"}

Run data:
- Horse: ${run?.horse || "not provided"}
- Official time: ${run?.time || "not provided"}s
- Show: ${run?.showName || "not provided"}
- Location: ${run?.location || "not provided"}
- Arena: ${run?.arenaCondition || "not provided"}
- Placing: ${run?.placing || "not provided"}
- Earnings: $${run?.earnings || "0"}
- Rider felt: "${run?.riderFeedback || "no feedback provided"}"
- Notes: "${run?.notes || "none"}"
- Manual splits set: ${run?.manualSplits ? "YES — user-marked splits are highly accurate" : "No"}
${(() => {
  const bt = run?.barrel_location_hints || run?.manualSplits?.barrel_video_timestamps;
  if (!bt) return "- Barrel video timestamps: not available";
  const dur = pythonResult?.duration_seconds;
  const pct = (t) => dur ? ` (${Math.round((t/dur)*100)}% through video)` : "";
  return [
    "- Barrel video timestamps (rider tapped when horse reached each barrel):",
    bt.barrel1 != null ? `  • 1st barrel reached at ${bt.barrel1.toFixed(2)}s${pct(bt.barrel1)}` : null,
    bt.barrel2 != null ? `  • 2nd barrel reached at ${bt.barrel2.toFixed(2)}s${pct(bt.barrel2)}` : null,
    bt.barrel3 != null ? `  • 3rd barrel reached at ${bt.barrel3.toFixed(2)}s${pct(bt.barrel3)}` : null,
    "  These timestamps anchor WHERE each barrel is in the video. Frames around these times show the horse at each barrel.",
  ].filter(Boolean).join("\n");
})()}
  `.trim();
}

function buildVideoPrompt(run, pythonResult) {
  const coachingData = buildBarrelCoachingData(run, pythonResult);
  const historicalContext = buildHistoricalContext(run);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";

  return `
${BARREL_RACING_KNOWLEDGE_BASE}

=== YOUR TASK ===

You are analyzing ${riderName}'s run on ${horseName}. You have up to 30 video frames covering the run — all three barrel approaches, apexes, exits, and the home run. When manual splits were set by the rider, frames are anchored precisely to each barrel timestamp. You also have detailed computer vision data.

WHAT TO LOOK FOR IN THE FRAMES — check every frame carefully:

RIDER POSITION:
- Is the rider sitting deep (pockets in the saddle) at the rate point, or still standing in the stirrups?
- Is the rider leaning into the barrel (motorcycle lean) or staying centered/spine-aligned?
- Are the rider's hands quiet and low, or up near the chin pulling?
- Is the rider in a forward power position on the straightaways, or leaning back?
- Is the rider looking at the barrel (bad) or looking ahead to the next one (good)?
- Is the rider grabbing the horn before completing the rate?

HORSE POSITION:
- Is the inside shoulder up or dropped through the turns?
- Is the horse's head up with nose out (on the forehand) or breaking at the poll (collected)?
- Is the horse driving hard out of each barrel with hindquarter engagement, or drifting?
- Is the horse on the correct lead for each turn?
- Does the horse look balanced and collected approaching each barrel, or strung out?

BARREL PROXIMITY:
- Did the horse appear at risk of clipping any barrel (running dangerously tight)?
- Did any barrel appear disturbed or moved?

COACHING RULES:
- Use the knowledge base above — sound like a real barrel racing professional
- Reference what you actually SEE in the frames, not what you assume
- Use specific barrel racing language: rate point, pocket, collection, drive, two-tracking, shoulder drop, on the forehand, breaking at the poll, motorcycle lean, power position, snap the turn
- Call barrels by name: first barrel, second barrel, third barrel
- If data shows a Grade D or F turn — say it plainly and explain why based on the knowledge base
- If the rider is leaning in (motorcycle lean) and you can see it in the frames — call it out by name
- If the horse's inside shoulder is dropped — explain what that costs them
- Compare the three barrels — which was cleanest, which needs work
- If manual splits were set, they are highly accurate — use them as ground truth
- Every drill must connect to a specific observed problem — use the named drills from the knowledge base when applicable: Square the Barrel, Horizon Focus, One-Handed Guide, Counter-Bending Circles, Transition Points, One-and-a-Half, Pinwheel, Deep Sit Stop, Quiet Entry, Loose Rein Loping
- NO generic advice. Every sentence must earn its place. Sound like a real professional coach, not an AI.
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- Return ONLY valid JSON. No markdown. No extra text.

${coachingData}

${historicalContext}

Return ONLY this exact JSON:

{
  "summary": "2-3 punchy sentences at the gate. Lead with the single most important finding. Reference the specific barrel and what you saw. Sound like a real coach.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "One specific actionable coaching cue — the highest priority fix",
  "speedInsight": "Use the SLOWEST split already identified in the CV data above — do not recalculate. Explain specifically why that section was slow based on what you observed in the frames and CV metrics",
  "splitAnalysis": "Read each split time using the knowledge base — what does each split tell you about what happened at that barrel",
  "patternNotes": "What the approach angles, tightness grades, and exit drives tell you about how this horse runs the pattern — use proper terminology",
  "visualObservations": "What you specifically saw in the video frames — rider position, horse shoulder, hands, seat, head position. Reference what you actually saw.",
  "accuracyNotes": "Honest note on data quality and what you could see clearly vs what was estimated",
  "strengths": ["Specific strength tied to data or what you saw in frames", "Another specific strength with detail"],
  "issues": ["Specific issue with barrel name and visual observation — e.g. Motorcycle lean at the second barrel — rider's upper body clearly leaning in, dropping the horse's inside shoulder", "Another specific issue with detail"],
  "workOns": ["Specific work-on tied to an observed problem with proper terminology", "Another targeted work-on"],
  "drills": ["Specific drill that directly addresses an observed issue — e.g. Walk and trot the pattern focusing on sitting deep the moment your horse's nose hits the barrel", "Another targeted drill tied to what you saw"]
}
  `.trim();
}

// ─── PASS 1: Vision Pass Prompt ──────────────────────────────────────────────
// GPT-4o sees all 60 frames. Only job: describe exactly what it sees.
// No coaching. No drills. Just clinical visual observation per barrel.

function buildVisionPassPrompt(run, pythonResult) {
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";
  const barrelTimestamps = run?.barrel_location_hints || run?.manualSplits?.barrel_video_timestamps || null;
  const manualSplits = run?.manualSplits || null;

  // Pre-identify slowest split so vision pass knows where to focus
  // Normalize splits to ft/s using Pattern A distances
  // The slowest ft/s section is where time was genuinely lost — not the longest raw time
  const rawSplitMap = {
    "Start to 1st Barrel":   manualSplits?.start_to_barrel1_seconds,
    "1st to 2nd Barrel":     manualSplits?.barrel1_to_barrel2_seconds,
    "2nd to 3rd Barrel":     manualSplits?.barrel2_to_barrel3_seconds,
    "3rd Barrel to Finish":  manualSplits?.barrel3_to_home_seconds,
  };
  const arenaDist = getArenaDistances(run);
  const normalizedSplits = normalizeSplitsToSpeed(
    Object.fromEntries(Object.entries(rawSplitMap).filter(([,v]) => v != null)),
    arenaDist
  );
  const slowestSection = normalizedSplits.length > 0 ? normalizedSplits[0].label : null;
  const slowestTime = normalizedSplits.length > 0 ? normalizedSplits[0].seconds.toFixed(2) : null;
  const slowestSpeed = normalizedSplits.length > 0 ? normalizedSplits[0].speed.toFixed(1) : null;

  const timestampContext = barrelTimestamps ? `
Barrel timestamps (rider tapped when horse reached each barrel):
${barrelTimestamps.barrel1 != null ? `- 1st barrel: ${barrelTimestamps.barrel1.toFixed(2)}s into video` : ""}
${barrelTimestamps.barrel2 != null ? `- 2nd barrel: ${barrelTimestamps.barrel2.toFixed(2)}s into video` : ""}
${barrelTimestamps.barrel3 != null ? `- 3rd barrel: ${barrelTimestamps.barrel3.toFixed(2)}s into video` : ""}
Frames near these timestamps show the horse at each barrel.` : "";

  const slowestFocus = slowestSection
    ? `
SLOWEST SECTION BY SPEED: "${slowestSection}" — ${slowestTime}s covering ${PATTERN_A_DISTANCES[slowestSection]}ft = ${slowestSpeed} ft/s. This is where the horse was genuinely slowest. Give this section extra scrutiny.`
    : "";

  return `You are watching a barrel racing run by ${riderName} on ${horseName}.
${timestampContext}
${slowestFocus}

WHAT TO LOOK FOR — only report what you can clearly see in the frames.
If something is not visible or unclear, say "not visible in frames."

FOCUS ON THESE 5 THINGS — use the exact terms below:

RIDER POSITION:
- "Neutral Spine" = upright, balanced in the saddle ✓
- "Quiet Hands" = hands low and steady ✓
- "The Tilter" = rider's upper body leaning toward the barrel ✗
- "The Reacher" = hands reaching forward pulling on the front end ✗

HORSE BODY:
- "Banana Shape" = horse's body curves around the barrel ✓
- "Log Stiff" = horse's body is straight, no arc ✗
- "Shoulder-Pop" = horse's shoulder pushes outward ✗

HORSE RATE (weight shift approaching barrel):
- "Sit & Squat" = hindquarters lower, horse collecting before the turn ✓
- "Downhill Run" = horse running heavy on the front end through the turn ✗

HORSE EXIT:
- "Square Exit" = horse drives cleanly out of the turn ✓
- "The Fishtail" = hind end swings out on exit ✗
- "Wandering" = horse drifts instead of driving to the next barrel ✗

BARREL CONTACT:
- Did any barrel appear to be knocked or disturbed? (yes/no/not visible)

REPORT FORMAT:
For each barrel write 2-3 sentences:
1. Which terms apply (positive and negative)
2. What you specifically saw in the frames
3. Your confidence level: HIGH (clearly visible) / MEDIUM (partially visible) / LOW (unclear)

Mark which barrel showed the CLEAREST FAULT — the coaching pass will focus there.`;
}

// ─── PASS 2: Coaching Pass Prompt ────────────────────────────────────────────
// NO images. Gets vision observations from Pass 1 + CV data + knowledge base.
// Produces short, accurate, factual report. No per-barrel sections. No degrees.

function buildCoachingPassPrompt(run, pythonResult, visionObservations) {
  const coachingData = buildBarrelCoachingData(run, pythonResult);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";
  const manualSplits = run?.manualSplits || null;

  // Penalty time calculation — must be declared BEFORE split scaling uses baseTime
  const baseTime = parseFloat(run?.time) || 0;
  const penaltySeconds = run?.knockedPenalty === "+5" && run?.knockedBarrels?.length > 0
    ? run.knockedBarrels.length * 5 : 0;
  const officialTime = baseTime + penaltySeconds;

  // Use manual splits if available (already scaled to official time in the app)
  // Fall back to CV splits and scale them to official time if needed
  const cvSplits = pythonResult?.splits || {};
  const rawSplits = manualSplits || cvSplits;
  let sp1 = rawSplits?.start_to_barrel1_seconds ?? null;
  let sp2 = rawSplits?.barrel1_to_barrel2_seconds ?? null;
  let sp3 = rawSplits?.barrel2_to_barrel3_seconds ?? null;
  let sp4 = rawSplits?.barrel3_to_home_seconds ?? null;

  // If using CV splits (no manual), scale to official time
  if (!manualSplits && baseTime > 0) {
    const cvTotal = [sp1, sp2, sp3, sp4].reduce((sum, v) => sum + (Number.isFinite(Number(v)) ? Number(v) : 0), 0);
    if (cvTotal > 0 && Math.abs(cvTotal - baseTime) > 0.1) {
      const scale = baseTime / cvTotal;
      if (sp1 != null) sp1 = parseFloat((Number(sp1) * scale).toFixed(2));
      if (sp2 != null) sp2 = parseFloat((Number(sp2) * scale).toFixed(2));
      if (sp3 != null) sp3 = parseFloat((Number(sp3) * scale).toFixed(2));
      if (sp4 != null) sp4 = parseFloat((Number(sp4) * scale).toFixed(2));
    }
  }

  // Normalize to ft/s using Pattern A distances — slowest ft/s is genuinely slowest
  const splitRawMap = {
    "Start to 1st Barrel":   sp1,
    "1st to 2nd Barrel":     sp2,
    "2nd to 3rd Barrel":     sp3,
    "3rd Barrel to Finish":  sp4,
  };
  const arenaDist = getArenaDistances(run);
  const normalizedSections = normalizeSplitsToSpeed(
    Object.fromEntries(Object.entries(splitRawMap).filter(([,v]) => v != null && Number.isFinite(Number(v)))),
    arenaDist
  );
  const slowestSplit = normalizedSections[0] || null;
  const fastestSplit = normalizedSections[normalizedSections.length - 1] || null;
  const hasSplits = normalizedSections.length > 0;
  const splitsLabel = manualSplits ? "user-marked, scaled to official time" : "CV-estimated, scaled to official time";

  return `${BARREL_RACING_KNOWLEDGE_BASE}

=== VISION OBSERVATIONS ===
${visionObservations}

=== RUN DATA ===
Horse: ${horseName} | Rider: ${riderName}
Official time: ${run?.knockedPenalty === "nt" ? "N-T (barrel knocked — no time)" : officialTime > 0 ? officialTime.toFixed(2) + "s" + (penaltySeconds > 0 ? ` (includes +${penaltySeconds}s penalty)` : "") : "not recorded"}
Show: ${run?.showName || "—"} | Location: ${run?.location || "—"} | Date: ${run?.runDate || "—"}
Arena: ${run?.arenaCondition || "not provided"} | Placing: ${run?.placing || "—"}
Knocked barrels: ${run?.knockedBarrels?.length > 0 ? `Barrel ${run.knockedBarrels.join(", ")} — ${run.knockedPenalty === "nt" ? "N-T" : `+${run.knockedBarrels.length * 5}s`}` : "none"}
Rider felt: "${run?.riderFeedback || "no feedback"}"
Notes: "${run?.notes || "none"}"
${hasSplits ? `
SPLIT TIMES WITH SPEED NORMALIZATION (${splitsLabel}):
Arena distances used — Start→B1: ${arenaDist["Start to 1st Barrel"]}ft | B1→B2: ${arenaDist["1st to 2nd Barrel"]}ft | B2→B3: ${arenaDist["2nd to 3rd Barrel"]}ft | B3→Finish: ${arenaDist["3rd Barrel to Finish"]}ft${run?.arenaDistances ? " (user-entered)" : " (WPRA Pattern A defaults)"}

${normalizedSections.map(s =>
  `  ${s.label}: ${s.seconds.toFixed(2)}s over ${s.distance}ft = ${s.speed.toFixed(1)} ft/s`
).join("\n")}

► SLOWEST BY SPEED: "${slowestSplit.label}" at ${slowestSplit.speed.toFixed(1)} ft/s (${slowestSplit.seconds.toFixed(2)}s over ${slowestSplit.distance}ft) — THIS IS WHERE THE HORSE WAS GENUINELY SLOWEST
► FASTEST BY SPEED: "${fastestSplit.label}" at ${fastestSplit.speed.toFixed(1)} ft/s

NOTE: Slowest is determined by ft/s, not raw time. Start→B1 (60ft) will always have the lowest raw time but may be the slowest speed. Base all coaching on the ft/s ranking above.` : "No splits available — use rider feedback and run data only."}

=== YOUR TASK ===
You are a barrel racing coach. Your job is simple:
1. Identify what went wrong at the SLOWEST split
2. Explain why using the fault framework
3. Give one specific, executable fix

FAULT FRAMEWORK — use these terms when they apply:
- Rider: "The Tilter" (leaning in) | "The Reacher" (hands forward/pulling)
- Horse body: "Log Stiff" (no arc) | "Shoulder-Pop" (shoulder pushes out)
- Rate: "Downhill Run" (on forehand) | "Stiff-Legged" (choppy, no collection)
- Exit: "The Fishtail" (hind swings out) | "Wandering" (drifts off line)

CORRELATION RULE — combine faults when they appear together:
- "The Tilter" causing "Downhill Run" = rider disrupted the rate cue
- "Log Stiff" causing "The Fishtail" = no rib bend led to hind end loss
- "The Reacher" causing "Shoulder-Pop" = pulling front end knocked horse off track
Only use a correlation if BOTH faults were actually observed.

WHAT GOOD OUTPUT LOOKS LIKE:
- Summary names the primary fault and the slowest split time
- timeLost[0] is about the SLOWEST split — specific fault, specific cause
- timeLost[1] is about a DIFFERENT section with a genuinely different fault
- timeLost[2] only exists if a third real problem was observed
- Each improvement matches its timeLost item exactly
- Drills are named and executable, not vague

WHAT BAD OUTPUT LOOKS LIKE (never do this):
- Repeating the same issue three different ways
- Saying time was lost somewhere the splits show was fast
- Generic advice with no specific barrel or drill named
- Inventing faults not supported by vision observations or split data

STRICT OUTPUT RULES:
- Do NOT say a turn or approach was wide or narrow — ever
- Do NOT mention degrees, angles, or pixel measurements
- Do NOT contradict the split data
- If only 1 or 2 genuine time losses exist, return 1 or 2 — do not pad
- Return ONLY valid JSON. No markdown. No extra text.

Return ONLY this JSON:
{
  "summary": "2-3 sentences. Reference the official time, the show or location if given, the slowest split, and the primary fault by name. Sound like a coach at the gate — direct and specific.",
  "timeLost": [
    "PRIMARY: [fault term] at [barrel] ([split time]s) — explain the cause in one sentence using barrel racing language.",
    "SECONDARY: A genuinely different fault at a DIFFERENT barrel/section from point 1. Never repeat the same barrel. Supported by vision or split data.",
    "THIRD: Only if a third distinct, real problem exists — otherwise omit this entry entirely."
  ],
  "improvements": [
    "Directly fixes timeLost[0]: name the drill, which barrel, how to execute it.",
    "Directly fixes timeLost[1]: specific and executable this week.",
    "Directly fixes timeLost[2] only if that entry exists."
  ],
}`.trim();
}

function buildTextOnlyPrompt(run) {
  const historicalContext = buildHistoricalContext(run);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";

  return `
${BARREL_RACING_KNOWLEDGE_BASE}

=== YOUR TASK ===

No video available. Coach ${riderName} on ${horseName} using their run data, their own feedback, and their history. Apply the knowledge base above — sound like a real barrel racing professional.

RULES:
- Acknowledge once that there's no video, then move on immediately
- Address the rider's own feedback directly using proper terminology from the knowledge base
- Use history context — personal best, trend, arena conditions
- Reference the knowledge base when explaining problems — use real barrel racing language
- Every sentence must be useful
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- Return ONLY valid JSON. No markdown. No extra text.

THIS RUN:
- Horse: ${horseName}
- Time: ${run?.time || "not provided"}s
- Show: ${run?.showName || "not provided"}
- Location: ${run?.location || "not provided"}
- Arena: ${run?.arenaCondition || "not provided"}
- Placing: ${run?.placing || "not provided"}
- Earnings: $${run?.earnings || "0"}
- Rider felt: "${run?.riderFeedback || "no feedback provided"}"
- Notes: "${run?.notes || "none"}"

${historicalContext}

STRICT RULES:
- Build the entire report around the rider's data — splits, feedback, arena condition, knocked barrels
- Use the 5-point framework terminology where applicable — even without video, rider feedback often reveals the fault category
- NEVER say a turn was "wide" — no exceptions
- NEVER contradict the split data
- NEVER produce generic advice — every point must reference a specific section of this run
- NEVER repeat the same problem across all 3 time loss points
- No degrees, no technical numbers
- Return ONLY valid JSON. No markdown. No extra text.

Return ONLY this JSON:
{
  "summary": "2-3 sentences. Reference the actual time, show/location, the slowest split, and what the rider felt. Sound like a real coach at the gate — specific to this run.",
  "timeLost": [
    "Time loss 1 — identify the specific section (name the barrel/split), what likely happened there based on the split time or rider feedback, explained with barrel racing terminology. Use framework terms if applicable.",
    "Time loss 2 — a DIFFERENT section or fault from point 1. Must be genuinely supported by data.",
    "Third time loss only if a third genuine loss is identifiable — otherwise return 2 honest points."
  ],
  "improvements": [
    "Directly fixes time loss 1 — name a specific drill and how to execute it for this exact fault.",
    "Directly fixes time loss 2 — specific and executable at their next training session.",
    "Directly fixes time loss 3 if applicable — tied to a named drill."
  ],
}
  `.trim();
}


// ─── Analysis Output Sanitizer ────────────────────────────────────────────────

function sanitizeAnalysis(parsed) {
  return {
    summary: parsed.summary || "",
    timeLost: Array.isArray(parsed.timeLost) ? parsed.timeLost : [],
    improvements: Array.isArray(parsed.improvements) ? parsed.improvements : [],
    drills: Array.isArray(parsed.drills) ? parsed.drills : [],
    // Legacy fields for backward compat with old saved analyses
    observations: Array.isArray(parsed.observations) ? parsed.observations : [],
    strengths: Array.isArray(parsed.strengths) ? parsed.strengths : [],
    issues: Array.isArray(parsed.issues) ? parsed.issues : [],
  };
}

function sanitizeBarrelMetrics(barrelMetrics) {
  if (!barrelMetrics || typeof barrelMetrics !== "object") return null;
  return barrelMetrics;
}

function sanitizePythonForClient(pythonResult) {
  if (!pythonResult || typeof pythonResult !== "object") return null;
  return {
    ok: !!pythonResult.ok,
    message: pythonResult.message || "",
    frame_count: pythonResult.frame_count ?? 0,
    fps: pythonResult.fps ?? 0,
    width: pythonResult.width ?? 0,
    height: pythonResult.height ?? 0,
    duration_seconds: pythonResult.duration_seconds ?? 0,
    horse_detected_frames: pythonResult.horse_detected_frames ?? 0,
    raw_trajectory_point_count: pythonResult.raw_trajectory_point_count ?? 0,
    accepted_trajectory_point_count: pythonResult.accepted_trajectory_point_count ?? 0,
    smoothed_trajectory_point_count: pythonResult.smoothed_trajectory_point_count ?? 0,
    normalized_smoothed_path_points: pythonResult.normalized_smoothed_path_points || [],
    tracking_quality: pythonResult.tracking_quality || null,
    barrel_detection_summary: pythonResult.barrel_detection_summary || null,
    identified_barrels: pythonResult.identified_barrels || { barrel1: null, barrel2: null, barrel3: null },
    turns: pythonResult.turns || { barrel1: null, barrel2: null, barrel3: null },
    splits: pythonResult.splits || {
      start_to_barrel1_seconds: null,
      barrel1_to_barrel2_seconds: null,
      barrel2_to_barrel3_seconds: null,
      barrel3_to_home_seconds: null,
      splits_method: "no_data",
    },
    pattern_direction: pythonResult.pattern_direction || null,
    normalized_actual_template_path: pythonResult.normalized_actual_template_path || [],
    ideal_template_path: pythonResult.ideal_template_path || [],
    insights: Array.isArray(pythonResult.insights) ? pythonResult.insights : [],
    highlights: pythonResult.highlights || null,
    barrel_metrics: pythonResult.barrel_metrics || null,
    speed_summary: pythonResult.speed_summary || null,
    sampled_frames: Array.isArray(pythonResult.sampled_frames)
      ? pythonResult.sampled_frames.map((f) => ({
          percent: f?.percent ?? null,
          frame_index: f?.frame_index ?? null,
          timestamp_seconds: f?.timestamp_seconds ?? null,
          read_success: !!f?.read_success,
          horse_detection: f?.horse_detection || null,
          barrel_detection_count: f?.barrel_detection_count ?? 0,
          rejection_reason: f?.rejection_reason || null,
          dense_pass: !!f?.dense_pass,
        }))
      : [],
    frame_metrics: Array.isArray(pythonResult.frame_metrics)
      ? pythonResult.frame_metrics.map((m) => ({
          frame_index: m?.frame_index ?? null,
          timestamp_seconds: m?.timestamp_seconds ?? null,
          horse_detected: !!m?.horse_detected,
          horse_center: m?.horse_center || null,
          nearest_barrel: m?.nearest_barrel || null,
          nearest_barrel_distance_px: m?.nearest_barrel_distance_px ?? null,
          dist_to_barrel1_px: m?.dist_to_barrel1_px ?? null,
          dist_to_barrel2_px: m?.dist_to_barrel2_px ?? null,
          dist_to_barrel3_px: m?.dist_to_barrel3_px ?? null,
        }))
      : [],
  };
}

// ─── Job Processors ───────────────────────────────────────────────────────────

async function processVideoJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) {
    console.warn("[VIDEO JOB] Job missing at start:", jobId);
    return;
  }

  const videoPath = job.videoPath;
  let pythonGeneratedPaths = [];

  try {
    updateJob(jobId, { status: "running", progress: 5, stage: "Starting analysis", startedAt: new Date().toISOString() });

    updateJob(jobId, { progress: 8, stage: "Running computer vision — tracking horse and barrels" });
    const pythonResult = await runPythonAnalysis(videoPath, job.run);
    pythonGeneratedPaths = getPythonGeneratedPaths(pythonResult);

    updateJob(jobId, { progress: 55, stage: "Computer vision complete — selecting key frames" });
    const framePaths = selectFramePaths(
      pythonResult.sampled_frames || [],
      30,
      job.run,
      pythonResult.duration_seconds || null
    );
    if (!framePaths.length) throw new Error("Python did not return any usable frame images.");

    updateJob(jobId, { progress: 62, stage: "Preparing frames for AI coach" });
    const imageInputs = buildImageInputs(framePaths);
    console.log(`[MEMORY] After frame load — ${imageInputs.length} frames ready for GPT-4o`);

    updateJob(jobId, { progress: 70, stage: "Pass 1 — AI vision coach analyzing frames" });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before AI analysis.");

    // ── PASS 1: VISION PASS ───────────────────────────────────────────────────
    // GPT-4o sees all 60 frames. Its ONLY job is to describe what it sees in
    // clinical detail — rider position, horse body language, hands, seat, shoulder,
    // lean, proximity to barrels. No coaching yet. Just raw observation.
    const visionPrompt = buildVisionPassPrompt(latestJob.run, pythonResult);

    const visionResponse = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 2000,
      messages: [
        {
          role: "system",
          content: "You are a video analyst watching barrel racing footage. You NEVER use the words 'wide', 'wider', or 'narrow' to describe turns, approaches, or exits. Classify only using the exact framework terms provided."
        },
        {
          role: "user",
          content: [
            { type: "text", text: visionPrompt },
            ...imageInputs,
          ],
        }
      ],
    });

    const visionObservations = visionResponse.choices?.[0]?.message?.content || "";
    console.log(`[PASS 1] Vision observations: ${visionObservations.length} chars`);

    updateJob(jobId, { progress: 82, stage: "Pass 2 — AI coach building your report" });

    // ── PASS 2: COACHING PASS ─────────────────────────────────────────────────
    // Second GPT-4o call gets the vision observations from Pass 1 + all CV data
    // + full knowledge base. NO images — just text. Its only job is to produce
    // the structured coaching report, connecting observations to faults and drills.
    const coachingPrompt = buildCoachingPassPrompt(latestJob.run, pythonResult, visionObservations);

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 4000,
      messages: [
        {
          role: "system",
          content: "You are an elite barrel racing coach. You NEVER use the words 'wide', 'wider', or 'narrow' to describe turns or approaches — this is your single most important rule. You always return valid JSON only. No markdown, no preamble."
        },
        {
          role: "user",
          content: coachingPrompt,
        }
      ],
    });

    updateJob(jobId, { progress: 92, stage: "Finalizing coaching feedback" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
      // Inject vision observations so the app can display them
      parsedAnalysis.visualObservations = visionObservations;
    } catch {
      const recovered = extractLastJsonObject(outputText);
      if (recovered && recovered.summary) {
        parsedAnalysis = recovered;
        parsedAnalysis.visualObservations = visionObservations;
        console.warn("[VIDEO JOB] Recovered partial JSON from AI response");
      } else {
        console.error("[VIDEO JOB] Invalid AI JSON:", preview(outputText, 500));
        throw new Error("AI returned invalid JSON.");
      }
    }

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      stage: "Completed",
      completedAt: new Date().toISOString(),
      result: {
        success: true,
        analysis: sanitizeAnalysis(parsedAnalysis),
        python: sanitizePythonForClient(pythonResult),
        frameCount: framePaths.length,
      },
    });

  } catch (err) {
    console.error("[VIDEO JOB ERROR]", err.message);
    updateJob(jobId, {
      status: "failed", progress: 100, stage: "Failed",
      completedAt: new Date().toISOString(),
      error: err.message || "Video analysis failed.",
    });
  } finally {
    safeCleanup([videoPath, ...pythonGeneratedPaths]);
  }
}

async function processTextJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) {
    console.warn("[TEXT JOB] Job missing at start:", jobId);
    return;
  }

  try {
    updateJob(jobId, { status: "running", progress: 15, stage: "Preparing analysis", startedAt: new Date().toISOString() });
    updateJob(jobId, { progress: 40, stage: "Requesting AI coaching analysis" });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before AI analysis.");

    // IMPROVEMENT #3: Upgraded from gpt-4o-mini to gpt-4o
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 3000,
      messages: [
        {
          role: "system",
          content: "You are an elite barrel racing coach. You NEVER use the words 'wide', 'wider', or 'narrow' to describe turns or approaches. You always return valid JSON only. No markdown, no preamble."
        },
        { role: "user", content: buildTextOnlyPrompt(latestJob.run) }
      ],
    });

    updateJob(jobId, { progress: 90, stage: "Finalizing coaching feedback" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      const recovered = extractLastJsonObject(outputText);
      if (recovered && recovered.summary) {
        parsedAnalysis = recovered;
        console.warn("[TEXT JOB] Recovered partial JSON from AI response");
      } else {
        console.error("[TEXT JOB] Invalid AI JSON:", preview(outputText, 500));
        throw new Error("AI returned invalid JSON.");
      }
    }

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      stage: "Completed",
      completedAt: new Date().toISOString(),
      result: {
        success: true,
        analysis: sanitizeAnalysis(parsedAnalysis),
        python: null,
        frameCount: 0,
      },
    });

  } catch (err) {
    console.error("[TEXT JOB ERROR]", err.message);
    updateJob(jobId, {
      status: "failed", progress: 100, stage: "Failed",
      completedAt: new Date().toISOString(),
      error: err.message || "Analysis failed.",
    });
  }
}

function startJobProcessing(job) {
  console.log("[JOB START]", job.id, "kind:", job.kind);
  if (job.kind === "video") void processVideoJob(job.id);
  else if (job.kind === "text") void processTextJob(job.id);
}

// ─── Expired Minor Account Cleanup (runs every 6 hours) ──────────────────────

async function cleanupExpiredMinorAccounts() {
  const sevenDays = 1000 * 60 * 60 * 24 * 7;
  const now = Date.now();
  const expired = [];

  for (const [token, record] of pendingConfirmations.entries()) {
    if (now - record.createdAt >= sevenDays) {
      expired.push({ token, record });
    }
  }

  if (!expired.length) {
    console.log("[CLEANUP] No expired minor accounts found");
    return;
  }

  console.log(`[CLEANUP] Found ${expired.length} expired minor account(s) — deleting`);

  for (const { token, record } of expired) {
    const { userId, minorEmail } = record;
    pendingConfirmations.delete(token);
    console.log(`[CLEANUP] Deleting expired account: ${minorEmail} (${userId})`);

    if (adminAuth) {
      try {
        await adminAuth.deleteUser(userId);
        console.log(`[CLEANUP] Deleted Auth user: ${userId}`);
      } catch (err) {
        console.warn(`[CLEANUP] Could not delete Auth user ${userId}:`, err.message);
      }
    }

    if (adminDb) {
      try {
        const collections = ["runs", "profile", "account", "consent"];
        for (const col of collections) {
          try {
            const snapshot = await adminDb.collection(`users/${userId}/${col}`).get();
            const batch = adminDb.batch();
            snapshot.docs.forEach((d) => batch.delete(d.ref));
            if (!snapshot.empty) await batch.commit();
          } catch { /* silent per collection */ }
        }
        await adminDb.doc(`users/${userId}`).delete();
        console.log(`[CLEANUP] Deleted Firestore data for: ${userId}`);
      } catch (err) {
        console.warn(`[CLEANUP] Firestore cleanup error for ${userId}:`, err.message);
      }
    }
  }

  await persistGuardianStore();
  console.log(`[CLEANUP] Expired account cleanup complete`);
}

setInterval(() => {
  cleanupExpiredMinorAccounts().catch((err) =>
    console.error("[CLEANUP] Error:", err.message)
  );
}, 1000 * 60 * 60 * 6);

restoreJobs();
restoreGuardianStore().catch((err) => console.error("[BOOT] Guardian restore failed:", err.message));

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/", (_req, res) => res.json({ ok: true, message: "Barrel Pro AI Server running", activeJobs: jobs.size }));
app.get("/health", (_req, res) => res.json({ ok: true, message: "Barrel Pro AI Server running", activeJobs: jobs.size }));

app.get("/debug/jobs", (_req, res) => {
  res.json({
    ok: true,
    count: jobs.size,
    jobs: Array.from(jobs.values()).map((j) => ({
      id: j.id, kind: j.kind, status: j.status, progress: j.progress,
      stage: j.stage, createdAt: j.createdAt, startedAt: j.startedAt,
      completedAt: j.completedAt, error: j.error, hasResult: !!j.result,
    })),
  });
});

app.post("/analyze-run-video/start", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No video file uploaded." });

    const run = safeParseJson(req.body?.runData ?? req.body?.run ?? "{}");
    if (!run || typeof run !== "object") {
      safeCleanup([req.file.path]);
      return res.status(400).json({ error: "Run data was missing or invalid." });
    }

    const job = createJob({ kind: "video", run, videoPath: req.file.path });
    updateJob(job.id, { progress: 5, stage: "Upload received" });
    startJobProcessing(job);

    console.log("[START VIDEO]", job.id);
    return res.json({ ok: true, jobId: job.id });
  } catch (err) {
    console.error("[START VIDEO ERROR]", err.message);
    return res.status(500).json({ error: "Could not start video analysis.", details: err.message });
  }
});

app.post("/analyze-run/start", async (req, res) => {
  try {
    const run = req.body || {};
    const job = createJob({ kind: "text", run });
    updateJob(job.id, { progress: 5, stage: "Analysis request received" });
    startJobProcessing(job);

    console.log("[START TEXT]", job.id);
    return res.json({ ok: true, jobId: job.id });
  } catch (err) {
    console.error("[START TEXT ERROR]", err.message);
    return res.status(500).json({ error: "Could not start analysis.", details: err.message });
  }
});

app.get("/analysis-status/:jobId", (req, res) => {
  const jobId = String(req.params.jobId || "").trim();
  console.log("[POLL]", jobId, "exists:", jobs.has(jobId), "total:", jobs.size);

  const job = jobs.get(jobId);
  if (!job) {
    return res.status(404).json({
      ok: false,
      error: "Job not found.",
      requestedJobId: jobId,
      activeJobCount: jobs.size,
    });
  }

  return res.json({
    ok: true,
    jobId: job.id,
    status: job.status,
    progress: job.progress,
    stage: job.stage,
    createdAt: job.createdAt,
    startedAt: job.startedAt,
    completedAt: job.completedAt,
    error: job.error,
    result: job.status === "completed" ? job.result : null,
  });
});

// ─── Guardian Routes ──────────────────────────────────────────────────────────

app.post("/send-guardian-email", async (req, res) => {
  try {
    const { guardianEmail, guardianName, minorEmail, minorAge, userId } = req.body;

    if (!guardianEmail || !guardianName || !minorEmail || !userId) {
      return res.status(400).json({ error: "Missing required fields." });
    }

    const token = `${userId}-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    pendingConfirmations.set(token, {
      userId, minorEmail, guardianEmail, guardianName, createdAt: Date.now(),
    });
    persistGuardianStore();

    const baseUrl = process.env.API_BASE_URL || "https://barrel-backend-gyyd.onrender.com";
    const confirmUrl = `${baseUrl}/confirm-guardian?token=${token}`;
    const rejectUrl = `${baseUrl}/reject-guardian?token=${token}`;

    await resend.emails.send({
      from: "Barrel Pro <noreply@fabhorsewear.com>",
      to: guardianEmail,
      subject: "Action Required — Confirm Your Child's Barrel Pro Account",
      html: `
        <div style="font-family: sans-serif; max-width: 600px; margin: 0 auto; padding: 24px;">
          <h2 style="color: #1ecad3;">Barrel Pro — Parental Approval Required</h2>
          <p>Hello ${guardianName},</p>
          <p>A Barrel Pro account was created for a user under 18 with you listed as parent or guardian.</p>
          <p><strong>Your child cannot access the app until you confirm below.</strong></p>
          <div style="text-align: center; margin: 32px 0; display: flex; gap: 16px; justify-content: center; flex-wrap: wrap;">
            <a href="${confirmUrl}" style="background: #1ecad3; color: #fff; padding: 16px 32px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 16px; display: inline-block; margin: 8px;">✅ Approve Account</a>
            <a href="${rejectUrl}" style="background: #b91c1c; color: #fff; padding: 16px 32px; border-radius: 8px; text-decoration: none; font-weight: 700; font-size: 16px; display: inline-block; margin: 8px;">❌ Reject & Delete Account</a>
          </div>
          <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
            <tr><td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">Account Email</td><td style="padding: 8px; border: 1px solid #e5e7eb;">${minorEmail}</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">User Age</td><td style="padding: 8px; border: 1px solid #e5e7eb;">${minorAge} years old</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">Guardian Name</td><td style="padding: 8px; border: 1px solid #e5e7eb;">${guardianName}</td></tr>
            <tr><td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">Date</td><td style="padding: 8px; border: 1px solid #e5e7eb;">${new Date().toLocaleDateString()}</td></tr>
          </table>
          <p style="color: #6b7280; font-size: 13px;">If you click <strong>Reject</strong>, the account and all associated data will be permanently deleted.</p>
          <p>Questions? Contact us at <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a></p>
          <p style="color: #9ca3af; font-size: 13px;">Barrel Pro — Built for barrel racers</p>
        </div>
      `,
    });

    console.log("[GUARDIAN EMAIL] Sent to:", guardianEmail, "token:", token);
    return res.json({ ok: true, token });
  } catch (err) {
    console.error("[GUARDIAN EMAIL ERROR]", err.message);
    return res.status(500).json({ error: "Failed to send guardian email.", details: err.message });
  }
});

app.get("/confirm-guardian", async (req, res) => {
  try {
    const token = String(req.query.token || "").trim();
    if (!token) {
      return res.status(400).send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#b91c1c;">Invalid Link</h2><p>This confirmation link is invalid.</p></div>`);
    }

    const record = pendingConfirmations.get(token);
    if (!record) {
      return res.status(400).send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#1ecad3;">✅ Already Approved</h2><p>This account has already been approved. Your child can now open the app and sign in.</p><p style="color:#9ca3af;font-size:13px;margin-top:32px;">Barrel Pro — Built for barrel racers</p></div>`);
    }

    confirmedUsers.set(record.userId, { confirmedAt: new Date().toISOString(), minorEmail: record.minorEmail });
    pendingConfirmations.delete(token);
    persistGuardianStore();

    console.log("[GUARDIAN CONFIRMED] userId:", record.userId);
    return res.send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#1ecad3;">✅ Account Approved!</h2><p>Your child's Barrel Pro account has been approved. They can now open the app and log in.</p><p style="color:#9ca3af;font-size:13px;margin-top:32px;">Barrel Pro — Built for barrel racers</p></div>`);
  } catch (err) {
    console.error("[GUARDIAN CONFIRM ERROR]", err.message);
    return res.status(500).send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#b91c1c;">Something Went Wrong</h2><p>Please try clicking the link again or contact us at <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a></p></div>`);
  }
});

app.get("/reject-guardian", async (req, res) => {
  try {
    const token = String(req.query.token || "").trim();
    if (!token) {
      return res.status(400).send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#b91c1c;">Invalid Link</h2><p>This rejection link is invalid.</p></div>`);
    }

    const record = pendingConfirmations.get(token);
    if (!record) {
      return res.status(400).send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#b91c1c;">Link Expired or Already Used</h2><p>Contact us at <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a></p></div>`);
    }

    const { userId, minorEmail } = record;
    pendingConfirmations.delete(token);
    rejectedUsers.add(userId);
    persistGuardianStore();

    console.log("[GUARDIAN REJECTED] userId:", userId, "email:", minorEmail);

    if (adminAuth) {
      try { await adminAuth.deleteUser(userId); console.log("[REJECT] Deleted Firebase Auth user:", userId); }
      catch (err) { console.error("[REJECT] Could not delete Firebase Auth user:", err.message); }
    }

    if (adminDb) {
      try {
        const collections = ["runs", "profile", "account", "consent"];
        for (const col of collections) {
          try {
            const snapshot = await adminDb.collection(`users/${userId}/${col}`).get();
            const batch = adminDb.batch();
            snapshot.docs.forEach((d) => batch.delete(d.ref));
            if (!snapshot.empty) await batch.commit();
          } catch { /* silent */ }
        }
        await adminDb.doc(`users/${userId}`).delete();
      } catch (err) { console.error("[REJECT] Firestore cleanup error:", err.message); }
    }

    return res.send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#b91c1c;">Account Rejected</h2><p>The Barrel Pro account for <strong>${minorEmail}</strong> has been rejected and permanently deleted.</p><p style="color:#9ca3af;font-size:13px;margin-top:32px;">Barrel Pro — Built for barrel racers</p></div>`);
  } catch (err) {
    console.error("[GUARDIAN REJECT ERROR]", err.message);
    return res.status(500).send(`<div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;"><h2 style="color:#b91c1c;">Something Went Wrong</h2><p>Contact us at <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a></p></div>`);
  }
});

app.get("/guardian-status/:userId", (req, res) => {
  const userId = String(req.params.userId || "").trim();
  if (!userId) return res.status(400).json({ ok: false, error: "Missing userId." });

  const confirmed = confirmedUsers.has(userId);
  const rejected = rejectedUsers.has(userId);
  const pending = Array.from(pendingConfirmations.values()).some((r) => r.userId === userId);

  console.log("[GUARDIAN STATUS]", userId, "confirmed:", confirmed, "rejected:", rejected, "pending:", pending);
  return res.json({ ok: true, confirmed, rejected, pending });
});

// ─── Error Handler ────────────────────────────────────────────────────────────

app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") return res.status(400).json({ error: "Video file is too large (over 600MB). This almost always means the video was recorded in 4K or 60fps. Please switch your camera to 1080p at 30fps — a 30-second barrel run should be under 150MB." });
    return res.status(400).json({ error: err.message || "Upload failed." });
  }
  if (err) return res.status(400).json({ error: err.message || "Request failed." });
  return res.status(500).json({ error: "Unknown server error." });
});

// ─── Start Server ─────────────────────────────────────────────────────────────

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Barrel Pro AI Server running on port ${PORT}`);
});
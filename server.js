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
const PYTHON_TIMEOUT_MS = 1000 * 60 * 8;
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

// ─── Firebase Admin (for deleting rejected minor accounts) ───────────────────
// NOTE: Add firebase-admin to your package.json and set FIREBASE_SERVICE_ACCOUNT env var
// with the JSON string of your Firebase service account key.

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
  limits: { fileSize: 250 * 1024 * 1024 },
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

// Persistence files for guardian state (survives Render restarts)
const GUARDIAN_STORE_FILE = path.join(process.cwd(), "guardian-store.json");

function persistGuardianStore() {
  try {
    const data = {
      savedAt: new Date().toISOString(),
      pendingConfirmations: Array.from(pendingConfirmations.entries()),
      confirmedUsers: Array.from(confirmedUsers.entries()),
      rejectedUsers: Array.from(rejectedUsers),
    };
    fs.writeFileSync(GUARDIAN_STORE_FILE, JSON.stringify(data, null, 2), "utf8");
  } catch (err) {
    console.error("[GUARDIAN PERSIST] Failed:", err.message);
  }
}

function restoreGuardianStore() {
  try {
    if (!fs.existsSync(GUARDIAN_STORE_FILE)) return;
    const parsed = safeParseJson(fs.readFileSync(GUARDIAN_STORE_FILE, "utf8"));
    if (!parsed) return;
    // Restore pending confirmations (skip expired ones older than 7 days)
    const sevenDays = 1000 * 60 * 60 * 24 * 7;
    if (Array.isArray(parsed.pendingConfirmations)) {
      for (const [token, record] of parsed.pendingConfirmations) {
        if (Date.now() - record.createdAt < sevenDays) {
          pendingConfirmations.set(token, record);
        }
      }
    }
    if (Array.isArray(parsed.confirmedUsers)) {
      for (const [userId, data] of parsed.confirmedUsers) {
        confirmedUsers.set(userId, data);
      }
    }
    if (Array.isArray(parsed.rejectedUsers)) {
      for (const userId of parsed.rejectedUsers) {
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

function selectFramePaths(sampledFrames, maxFrames = 4) {
  const usable = (sampledFrames || [])
    .filter((f) => f?.read_success && (f?.overlay_image_path || f?.image_path))
    .map((f) => f.overlay_image_path || f.image_path)
    .filter(Boolean);

  if (usable.length <= maxFrames) return usable;

  const selected = [];
  for (let i = 0; i < maxFrames; i++) {
    selected.push(usable[Math.round((i * (usable.length - 1)) / (maxFrames - 1))]);
  }
  return [...new Set(selected)];
}

function buildImageInputs(framePaths) {
  return framePaths.map((p) => ({
    type: "image_url",
    image_url: {
      url: `data:image/jpeg;base64,${fs.readFileSync(p).toString("base64")}`,
      detail: "low",
    },
  }));
}

// ─── AI Prompts ───────────────────────────────────────────────────────────────

function buildCvSummary(run, pythonResult) {
  const barrels = pythonResult?.identified_barrels || {};
  const splits = pythonResult?.splits || {};
  const insights = Array.isArray(pythonResult?.insights) ? pythonResult.insights.slice(0, 4) : [];
  const splitsMethod = splits?.splits_method || "unknown";

  const barrelLines = ["barrel1", "barrel2", "barrel3"].map((name) => {
    const b = barrels[name];
    if (!b) return `${name}: not detected`;
    return `${name}: center=(${b.center_x}, ${b.center_y}), detections=${b.detection_count}`;
  });

  return `
Computer vision data:
- Run duration: ${pythonResult?.duration_seconds ?? "unknown"} seconds
- Horse detected in ${pythonResult?.horse_detected_frames ?? "unknown"} frames
- Pattern direction: ${pythonResult?.pattern_direction ?? "unknown"}
- Barrel ID method: ${pythonResult?.tracking_quality?.barrel_id_method ?? "unknown"}
- Dense pass used: ${pythonResult?.tracking_quality?.dense_pass_used ?? false}

Barrel positions:
- ${barrelLines.join("\n- ")}

Split times (method: ${splitsMethod}):
- Start to 1st barrel: ${splits?.start_to_barrel1_seconds ?? "n/a"} sec
- 1st to 2nd barrel: ${splits?.barrel1_to_barrel2_seconds ?? "n/a"} sec
- 2nd to 3rd barrel: ${splits?.barrel2_to_barrel3_seconds ?? "n/a"} sec
- 3rd barrel to home: ${splits?.barrel3_to_home_seconds ?? "n/a"} sec

CV insights:
${insights.length ? insights.map((i) => `- ${i}`).join("\n") : "- none"}

Run details:
- Horse: ${run?.horse || "not provided"}
- Time: ${run?.time || "not provided"} seconds
- Show: ${run?.showName || "not provided"}
- Location: ${run?.location || "not provided"}
- Arena condition: ${run?.arenaCondition || "not provided"}
- Placing: ${run?.placing || "not provided"}
- Earnings: ${run?.earnings || "not provided"}
- Rider feedback: ${run?.riderFeedback || "none"}
- Notes: ${run?.notes || "none"}
  `.trim();
}

function buildVideoPrompt(run, pythonResult) {
  const cvSummary = buildCvSummary(run, pythonResult);

  return `
You are a seasoned professional barrel racer and coach with decades of competitive experience — NFR qualifications, world titles, and coaching champions. You speak directly, practically, and with real expertise. You sound like a real barrel racer standing at the gate, not a generic sports coach.

You are reviewing video frames and computer vision data from a barrel racing run. Give this rider specific, no-nonsense coaching feedback.

Use proper barrel racing terminology naturally:
- Rate (slowing before a barrel), pocket (space around a barrel), drive (forward momentum exiting)
- Shoulder control, lead changes, collection, run-down
- First barrel, second barrel, third barrel (not "barrel 1" in coaching text)
- Home (finish line), alley (entry run)

Coaching priorities — address in order of impact on time:
1. The single most important thing this rider needs to fix RIGHT NOW
2. What they did well and should keep doing
3. Specific issues at each barrel where evidence exists
4. Drills that directly address what you see
5. Where time is being lost on this run

Rules:
- Be direct and specific. "Your horse dropped its shoulder entering the first barrel" beats "there were issues at barrel one."
- If you can see something in the frames, say exactly what you see.
- If data is limited, say so honestly but still give your best read.
- Do NOT invent details not supported by the data or frames.
- Summary: 2-3 punchy sentences like you're talking to a rider at the gate.
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- "focusNext" must be one short specific coaching cue — the single most important thing
- "speedInsight" — where is this run losing time? Be specific.
- "accuracyNotes" — honest note on what you could and could not see clearly
- 2-3 items each in strengths/issues/workOns/drills — quality over quantity
- Return ONLY valid JSON. No markdown. No backticks.

${cvSummary}

Return ONLY valid JSON:

{
  "summary": "Direct 2-3 sentence coaching read like you are talking at the gate.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "Rate earlier and give more pocket at the first barrel",
  "speedInsight": "This run is losing time in the run-down to the first barrel.",
  "accuracyNotes": "Split times are estimates from video tracking — use as coaching reference, not official data.",
  "strengths": ["Good forward drive out of the second barrel", "Horse showed nice collection at the third"],
  "issues": ["Coming in too straight to the first barrel", "Dropping the outside shoulder through the second"],
  "workOns": ["Work your approach angle to the first barrel", "Focus on outside shoulder through the turn"],
  "drills": ["Set a cone 6 feet past your first barrel and run to that cone", "Trot figure-8s focusing on outside shoulder elevation"]
}
  `.trim();
}

function buildTextOnlyPrompt(run) {
  return `
You are a seasoned professional barrel racer and coach with decades of competitive experience — NFR qualifications, world titles, and coaching champions. You speak directly, practically, and with real expertise. You sound like a real barrel racer, not a generic sports coach.

No video is available. Coach based on run data and rider notes only. Be upfront that you are working without video, but still give your most useful coaching read based on what the rider has shared.

Use proper barrel racing terminology naturally:
- Rate, pocket, drive, shoulder control, lead changes, collection, run-down
- First barrel, second barrel, third barrel
- Home, alley

Rules:
- Be direct and specific based on the rider's notes and run data
- If the rider mentioned something specific, address it by name
- Be honest that no video was available but don't dwell on it
- Summary: 2-3 punchy sentences like you're talking at the gate
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- "focusNext" — one specific coaching cue
- "speedInsight" — where is this run likely losing time based on data and notes?
- "accuracyNotes" — honest note about coaching without video
- 2-3 items each in strengths/issues/workOns/drills
- Return ONLY valid JSON. No markdown. No backticks.

Run data:
- Horse: ${run?.horse || "not provided"}
- Time: ${run?.time || "not provided"} seconds
- Show: ${run?.showName || "not provided"}
- Location: ${run?.location || "not provided"}
- Arena condition: ${run?.arenaCondition || "not provided"}
- Placing: ${run?.placing || "not provided"}
- Earnings: ${run?.earnings || "not provided"}
- Rider feedback: ${run?.riderFeedback || "none"}
- Notes: ${run?.notes || "none"}

Return ONLY valid JSON:

{
  "summary": "Direct 2-3 sentence coaching read based on run data and rider notes.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "Rate earlier and give more pocket at the first barrel",
  "speedInsight": "Based on your notes, the first barrel approach is where this run is likely losing the most time.",
  "accuracyNotes": "Coaching based on your notes and run data only — no video was available.",
  "strengths": ["Solid time suggests good overall pace", "Making the check shows competitive conditioning"],
  "issues": ["Rider noted feeling late to the first barrel", "Any shoulder drop at the second costs time in the crossover"],
  "workOns": ["Identify your rate point to the first barrel and be consistent hitting it", "Film practice runs so you can see what your approach angles actually look like"],
  "drills": ["Trot your pattern focusing on where you are rating", "Practice your run-down to the first barrel focusing on a consistent approach angle"]
}
  `.trim();
}

// ─── Analysis Output Sanitizer ────────────────────────────────────────────────

function sanitizeAnalysis(parsed) {
  return {
    summary: parsed.summary || "",
    bestBarrel: parsed.bestBarrel || null,
    bestTurn: parsed.bestTurn || null,
    focusNext: parsed.focusNext || null,
    speedInsight: parsed.speedInsight || null,
    accuracyNotes: parsed.accuracyNotes || null,
    strengths: Array.isArray(parsed.strengths) ? parsed.strengths : [],
    issues: Array.isArray(parsed.issues) ? parsed.issues : [],
    workOns: Array.isArray(parsed.workOns) ? parsed.workOns : [],
    drills: Array.isArray(parsed.drills) ? parsed.drills : [],
  };
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

    updateJob(jobId, { progress: 10, stage: "Running computer vision" });
    const pythonResult = await runPythonAnalysis(videoPath, job.run);
    pythonGeneratedPaths = getPythonGeneratedPaths(pythonResult);

    updateJob(jobId, { progress: 60, stage: "Selecting key frames" });
    const framePaths = selectFramePaths(pythonResult.sampled_frames || [], 4);
    if (!framePaths.length) throw new Error("Python did not return any usable frame images.");

    updateJob(jobId, { progress: 68, stage: "Preparing frames for AI" });
    const imageInputs = buildImageInputs(framePaths);

    updateJob(jobId, { progress: 75, stage: "Requesting AI coaching analysis" });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before AI analysis.");

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      max_tokens: 1200,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: buildVideoPrompt(latestJob.run, pythonResult) },
          ...imageInputs,
        ],
      }],
    });

    updateJob(jobId, { progress: 92, stage: "Finalizing analysis" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error("[VIDEO JOB] Invalid AI JSON:", preview(outputText, 500));
      throw new Error("AI returned invalid JSON.");
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

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      max_tokens: 1200,
      messages: [{ role: "user", content: buildTextOnlyPrompt(latestJob.run) }],
    });

    updateJob(jobId, { progress: 92, stage: "Finalizing analysis" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error("[TEXT JOB] Invalid AI JSON:", preview(outputText, 500));
      throw new Error("AI returned invalid JSON.");
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

// ─── Boot ─────────────────────────────────────────────────────────────────────

restoreJobs();
restoreGuardianStore();

// ─── Routes ───────────────────────────────────────────────────────────────────

// Health checks
app.get("/", (_req, res) => res.json({ ok: true, message: "Barrel Pro AI Server running", activeJobs: jobs.size }));
app.get("/health", (_req, res) => res.json({ ok: true, message: "Barrel Pro AI Server running", activeJobs: jobs.size }));

// Debug: view all jobs
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

// Start a video analysis job
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

// Start a text-only analysis job
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

// Poll job status
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

// ─── Guardian Email (Parental Consent + Confirmation) ─────────────────────────

app.post("/send-guardian-email", async (req, res) => {
  try {
    const { guardianEmail, guardianName, minorEmail, minorAge, userId } = req.body;

    if (!guardianEmail || !guardianName || !minorEmail || !userId) {
      return res.status(400).json({ error: "Missing required fields." });
    }

    const token = `${userId}-${Date.now()}-${Math.random().toString(36).slice(2)}`;

    pendingConfirmations.set(token, {
      userId,
      minorEmail,
      guardianEmail,
      guardianName,
      createdAt: Date.now(),
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
            <a href="${confirmUrl}"
              style="background: #1ecad3; color: #fff; padding: 16px 32px; border-radius: 8px;
                     text-decoration: none; font-weight: 700; font-size: 16px; display: inline-block; margin: 8px;">
              ✅ Approve Account
            </a>
            <a href="${rejectUrl}"
              style="background: #b91c1c; color: #fff; padding: 16px 32px; border-radius: 8px;
                     text-decoration: none; font-weight: 700; font-size: 16px; display: inline-block; margin: 8px;">
              ❌ Reject & Delete Account
            </a>
          </div>
          <table style="width: 100%; border-collapse: collapse; margin: 16px 0;">
            <tr>
              <td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">Account Email</td>
              <td style="padding: 8px; border: 1px solid #e5e7eb;">${minorEmail}</td>
            </tr>
            <tr>
              <td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">User Age</td>
              <td style="padding: 8px; border: 1px solid #e5e7eb;">${minorAge} years old</td>
            </tr>
            <tr>
              <td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">Guardian Name</td>
              <td style="padding: 8px; border: 1px solid #e5e7eb;">${guardianName}</td>
            </tr>
            <tr>
              <td style="padding: 8px; border: 1px solid #e5e7eb; font-weight: 600;">Date</td>
              <td style="padding: 8px; border: 1px solid #e5e7eb;">${new Date().toLocaleDateString()}</td>
            </tr>
          </table>
          <p style="color: #6b7280; font-size: 13px;">
            If you click <strong>Reject</strong>, the account and all associated data will be permanently deleted.
            If you did not authorize this account creation, please click Reject.
          </p>
          <p>Questions? Contact us at
            <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a>
          </p>
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

// ─── Guardian Confirm Endpoint ────────────────────────────────────────────────

app.get("/confirm-guardian", async (req, res) => {
  try {
    const token = String(req.query.token || "").trim();
    if (!token) {
      return res.status(400).send(`
        <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
          <h2 style="color:#b91c1c;">Invalid Link</h2>
          <p>This confirmation link is invalid. Please check your email and try again.</p>
        </div>
      `);
    }

    const record = pendingConfirmations.get(token);
    if (!record) {
      return res.status(400).send(`
        <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
          <h2 style="color:#b91c1c;">Link Expired or Already Used</h2>
          <p>This confirmation link has already been used or has expired.</p>
          <p>If your child still cannot log in, please contact us at
            <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a>
          </p>
        </div>
      `);
    }

    confirmedUsers.set(record.userId, {
      confirmedAt: new Date().toISOString(),
      minorEmail: record.minorEmail,
    });
    pendingConfirmations.delete(token);
    persistGuardianStore();

    console.log("[GUARDIAN CONFIRMED] userId:", record.userId);

    return res.send(`
      <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
        <h2 style="color:#1ecad3;">✅ Account Approved!</h2>
        <p>Your child's Barrel Pro account has been approved.</p>
        <p>They can now open the app and log in.</p>
        <p style="color:#9ca3af;font-size:13px;margin-top:32px;">Barrel Pro — Built for barrel racers</p>
      </div>
    `);
  } catch (err) {
    console.error("[GUARDIAN CONFIRM ERROR]", err.message);
    return res.status(500).send(`
      <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
        <h2 style="color:#b91c1c;">Something Went Wrong</h2>
        <p>Please try clicking the link again or contact us at
          <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a>
        </p>
      </div>
    `);
  }
});

// ─── Guardian Reject Endpoint (FIX 7 + 12) ───────────────────────────────────
// When parent clicks Reject: deletes Firebase Auth account + Firestore data

app.get("/reject-guardian", async (req, res) => {
  try {
    const token = String(req.query.token || "").trim();
    if (!token) {
      return res.status(400).send(`
        <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
          <h2 style="color:#b91c1c;">Invalid Link</h2>
          <p>This rejection link is invalid. Please check your email and try again.</p>
        </div>
      `);
    }

    const record = pendingConfirmations.get(token);
    if (!record) {
      return res.status(400).send(`
        <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
          <h2 style="color:#b91c1c;">Link Expired or Already Used</h2>
          <p>This link has already been used or has expired.</p>
          <p>If you need assistance, contact us at
            <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a>
          </p>
        </div>
      `);
    }

    const { userId, minorEmail } = record;
    pendingConfirmations.delete(token);
    rejectedUsers.add(userId);
    persistGuardianStore();

    console.log("[GUARDIAN REJECTED] userId:", userId, "email:", minorEmail);

    // Delete Firebase Auth account
    if (adminAuth) {
      try {
        await adminAuth.deleteUser(userId);
        console.log("[REJECT] Deleted Firebase Auth user:", userId);
      } catch (err) {
        console.error("[REJECT] Could not delete Firebase Auth user:", err.message);
      }
    } else {
      console.warn("[REJECT] Firebase Admin not initialized — Auth user not deleted");
    }

    // Delete Firestore data
    if (adminDb) {
      try {
        const collections = ["runs", "profile", "account", "consent"];
        for (const col of collections) {
          try {
            const snapshot = await adminDb.collection(`users/${userId}/${col}`).get();
            const batch = adminDb.batch();
            snapshot.docs.forEach((d) => batch.delete(d.ref));
            if (!snapshot.empty) await batch.commit();
            console.log(`[REJECT] Deleted Firestore collection users/${userId}/${col}`);
          } catch (colErr) {
            console.warn(`[REJECT] Could not delete collection ${col}:`, colErr.message);
          }
        }
        // Delete the user document itself
        await adminDb.doc(`users/${userId}`).delete();
        console.log("[REJECT] Deleted Firestore user document:", userId);
      } catch (err) {
        console.error("[REJECT] Firestore cleanup error:", err.message);
      }
    } else {
      console.warn("[REJECT] Firebase Admin not initialized — Firestore data not deleted");
    }

    return res.send(`
      <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
        <h2 style="color:#b91c1c;">Account Rejected</h2>
        <p>The Barrel Pro account for <strong>${minorEmail}</strong> has been rejected and permanently deleted.</p>
        <p>All associated data has been removed from our servers.</p>
        <p style="color:#9ca3af;font-size:13px;margin-top:32px;">Barrel Pro — Built for barrel racers</p>
      </div>
    `);
  } catch (err) {
    console.error("[GUARDIAN REJECT ERROR]", err.message);
    return res.status(500).send(`
      <div style="font-family:sans-serif;max-width:500px;margin:60px auto;text-align:center;padding:24px;">
        <h2 style="color:#b91c1c;">Something Went Wrong</h2>
        <p>Please try again or contact us at
          <a href="mailto:ben.dejonge34@gmail.com">ben.dejonge34@gmail.com</a>
        </p>
      </div>
    `);
  }
});

// ─── Check Guardian Status (app polls this at login) ─────────────────────────

app.get("/guardian-status/:userId", (req, res) => {
  const userId = String(req.params.userId || "").trim();
  if (!userId) return res.status(400).json({ ok: false, error: "Missing userId." });

  const confirmed = confirmedUsers.has(userId);
  const rejected = rejectedUsers.has(userId);
  console.log("[GUARDIAN STATUS]", userId, "confirmed:", confirmed, "rejected:", rejected);
  return res.json({ ok: true, confirmed, rejected });
});

// ─── Error Handler ────────────────────────────────────────────────────────────

app.use((err, _req, res, _next) => {
  if (err instanceof multer.MulterError) {
    if (err.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({ error: "Video is too large. Max size is 250MB." });
    }
    return res.status(400).json({ error: err.message || "Upload failed." });
  }
  if (err) return res.status(400).json({ error: err.message || "Request failed." });
  return res.status(500).json({ error: "Unknown server error." });
});

// ─── Start Server ─────────────────────────────────────────────────────────────

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Barrel Pro AI Server running on port ${PORT}`);
});
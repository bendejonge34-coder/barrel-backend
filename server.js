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

function buildBarrelCoachingData(run, pythonResult) {
  const barrelMetrics = pythonResult?.barrel_metrics || {};
  const speedSummary = pythonResult?.speed_summary || null;
  const splits = pythonResult?.splits || {};
  const turns = pythonResult?.turns || {};
  const insights = Array.isArray(pythonResult?.insights) ? pythonResult.insights : [];

  const barrelLabels = { barrel1: "First", barrel2: "Second", barrel3: "Third" };
  
  // Build detailed per-barrel coaching report
  const barrelReport = ["barrel1", "barrel2", "barrel3"].map(name => {
    const bm = barrelMetrics[name];
    const label = barrelLabels[name];
    if (!bm || !bm.detected) return `${label} barrel: not detected in video`;
    
    const tightness = bm.turn_tightness || {};
    const approach = bm.approach || {};
    const exitDrive = bm.exit_drive || null;
    const knocked = bm.potential_knockdown;
    
    const lines = [`${label} barrel:`];
    
    // Turn tightness with grade
    if (tightness.grade) {
      lines.push(`  - Turn tightness: Grade ${tightness.grade} (${tightness.label}) — ${tightness.coaching_note}`);
      if (tightness.min_distance_px !== null) {
        lines.push(`  - Closest approach: ${tightness.min_distance_px}px from barrel center`);
      }
    }
    
    // Approach angle
    if (approach.angle_degrees !== null && approach.angle_degrees !== undefined) {
      lines.push(`  - Approach angle: ${approach.angle_degrees}° (ideal: 20-40°) — ${approach.coaching_note}`);
    }
    
    // Exit drive
    if (exitDrive) {
      lines.push(`  - Exit drive: ${exitDrive.coaching_note}`);
      if (exitDrive.apex_speed_px_per_sec && exitDrive.exit_speed_px_per_sec) {
        const ratio = exitDrive.acceleration_ratio;
        lines.push(`  - Speed at turn: ${exitDrive.apex_speed_px_per_sec}px/s → exit: ${exitDrive.exit_speed_px_per_sec}px/s (${ratio >= 1.0 ? "+" : ""}${((ratio - 1) * 100).toFixed(0)}% change)`);
      }
    }
    
    // Knockdown flag
    if (knocked) {
      lines.push(`  - ⚠️ POSSIBLE KNOCKDOWN DETECTED (confidence: ${Math.round((bm.knockdown_confidence || 0) * 100)}%) — ${bm.knockdown_note || "barrel movement detected"}`);
    }
    
    // Summary tags
    if (bm.summary_tags && bm.summary_tags.length > 0) {
      lines.push(`  - Pattern tags: ${bm.summary_tags.join(", ")}`);
    }
    
    return lines.join("\n");
  }).join("\n\n");

  // Speed analysis
  let speedReport = "Speed data: not available";
  if (speedSummary) {
    const lines = ["Run speed profile:"];
    if (speedSummary.slowest_section_label) {
      lines.push(`  - SLOWEST section: ${speedSummary.slowest_section_label} — this is where the most time is being lost`);
    }
    if (speedSummary.fastest_section_label) {
      lines.push(`  - Fastest section: ${speedSummary.fastest_section_label}`);
    }
    if (speedSummary.section_speeds) {
      const s = speedSummary.section_speeds;
      lines.push(`  - Alley→1st: ${s.alley_to_barrel1 ?? "n/a"}px/s | 1st→2nd: ${s.barrel1_to_barrel2 ?? "n/a"}px/s | 2nd→3rd: ${s.barrel2_to_barrel3 ?? "n/a"}px/s | 3rd→Home: ${s.barrel3_to_home ?? "n/a"}px/s`);
    }
    speedReport = lines.join("\n");
  }

  // Split times
  const splitsMethod = splits?.splits_method || "unknown";
  const splitReport = `Split times (method: ${splitsMethod}):
  - Alley to first barrel: ${splits?.start_to_barrel1_seconds ?? "n/a"}s
  - First to second barrel: ${splits?.barrel1_to_barrel2_seconds ?? "n/a"}s
  - Second to third barrel: ${splits?.barrel2_to_barrel3_seconds ?? "n/a"}s
  - Third barrel to home: ${splits?.barrel3_to_home_seconds ?? "n/a"}s`;

  return `
=== COMPUTER VISION COACHING DATA ===

Run overview:
- Duration: ${pythonResult?.duration_seconds ?? "unknown"}s | Pattern: ${pythonResult?.pattern_direction ?? "unknown"}-first
- Horse tracked: ${pythonResult?.horse_detected_frames ?? "?"} frames | Frames analyzed: ${pythonResult?.tracking_quality?.sampled_frame_count ?? "?"}
- Video: ${pythonResult?.width ?? "?"}x${pythonResult?.height ?? "?"}

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
  `.trim();
}

function buildVideoPrompt(run, pythonResult) {
  const coachingData = buildBarrelCoachingData(run, pythonResult);
  const historicalContext = buildHistoricalContext(run);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";

  return `
You are an elite barrel racing coach — a seasoned NFR competitor, trainer, and analyst who has watched thousands of runs. You speak directly, use proper barrel racing terminology, and give advice that actually helps riders go faster.

You have detailed computer vision data from ${riderName}'s run on ${horseName}, including per-barrel turn grades, approach angles, speed profiles, exit drive analysis, and potential knockdown flags. Use ALL of this data. Do not give generic advice — every coaching note must connect directly to what the data shows.

COACHING RULES:
- Sound like a real barrel racing coach standing at the gate, not a sports AI
- Use proper terminology naturally: rate point, pocket, drive, collection, two-tracking, over-running, shoulder drop, lead change, run-down, alley, home
- Call each barrel by name: first barrel, second barrel, third barrel — never "barrel 1"
- If the data shows a Grade D or F turn, say it plainly: "You're running wide at the second barrel — that's costing you time"
- If a knockdown was detected, address it directly
- If approach angle is "too_straight", explain what that means: the horse will blow past the pocket
- If exit drive shows "drifted", explain: the horse isn't pushing forward out of the turn
- The slowest split section IS where the time is going — name it specifically
- Compare barrel grades to each other: "Your third barrel (Grade A) was your best turn — your second (Grade D) is costing you the most time"
- If rider left feedback, address it by name
- Every drill must connect to a specific observed problem
- NO generic advice. Every sentence earns its place.
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- "focusNext": ONE specific coaching cue — the single highest-priority fix
- Return ONLY valid JSON. No markdown. No extra text.

${coachingData}

${historicalContext}

Return ONLY this exact JSON structure:

{
  "summary": "2-3 punchy sentences at the gate. Lead with the most important finding from the data. Reference specific barrels and grades.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "One specific, actionable coaching cue tied directly to the data",
  "speedInsight": "Name the slowest section and explain why based on speed profile and turn grades",
  "splitAnalysis": "Read each split in context — which was strongest, which cost time, what it tells you about the pattern",
  "patternNotes": "What the approach angles, tightness grades, and exit drives tell you about how this horse runs the pattern",
  "accuracyNotes": "Honest note on data quality — what you could see clearly vs what was estimated",
  "strengths": ["Specific strength tied to data — mention grade or measurement", "Another specific strength"],
  "issues": ["Specific issue with barrel name and grade — e.g. Wide approach at the second barrel (Grade D, 87px) — running past the pocket", "Another specific issue"],
  "workOns": ["Specific work-on tied to observed problem", "Another specific work-on"],
  "drills": ["Specific drill that directly addresses an observed issue", "Another targeted drill"]
}
  `.trim();
}

function buildTextOnlyPrompt(run) {
  const historicalContext = buildHistoricalContext(run);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";

  return `
You are an elite barrel racing coach with NFR-level experience and thousands of runs coached. You speak directly, practically, and with real expertise.

No video is available for this run. Coach ${riderName} on ${horseName} using their run data, their own feedback, and their full history. A great coach delivers real value with or without video.

RULES:
- Acknowledge once that there's no video, then move on immediately — don't dwell on it
- Address the rider's own words directly. If they said "felt late to the first barrel" — coach that specific thing
- Use the history: if this was a personal best, say so. If they're trending slower, say so.
- Use proper terminology: rate, pocket, drive, collection, shoulder, run-down, alley, home
- Call barrels by name: first barrel, second barrel, third barrel
- Every sentence must be useful — no filler, no generic encouragement
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- "focusNext": ONE specific coaching cue
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

Return ONLY this JSON:

{
  "summary": "2-3 sentences at the gate. Reference rider's own feedback and history context.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "The single most important coaching cue for next time",
  "speedInsight": "Where time is likely being lost based on notes, feedback, and history",
  "splitAnalysis": "Read this time in context of their history — what does it suggest about the pattern",
  "patternNotes": "Based on rider feedback and history, what pattern tendencies are likely showing up",
  "accuracyNotes": "Honest note: coaching from rider feedback and history only, no video data available",
  "strengths": ["Specific strength from feedback or history", "Another specific strength"],
  "issues": ["Specific issue from rider's own words or history pattern", "Another specific issue"],
  "workOns": ["Specific work-on tied to their feedback", "Another targeted work-on"],
  "drills": ["Specific drill tied to their stated problem", "Another targeted drill"]
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
    splitAnalysis: parsed.splitAnalysis || null,
    patternNotes: parsed.patternNotes || null,
    accuracyNotes: parsed.accuracyNotes || null,
    strengths: Array.isArray(parsed.strengths) ? parsed.strengths : [],
    issues: Array.isArray(parsed.issues) ? parsed.issues : [],
    workOns: Array.isArray(parsed.workOns) ? parsed.workOns : [],
    drills: Array.isArray(parsed.drills) ? parsed.drills : [],
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

    // IMPROVEMENT #3: Upgraded from gpt-4o-mini to gpt-4o for significantly better
    // coaching quality, reasoning depth, and barrel racing expertise.
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 1800,
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

    // IMPROVEMENT #3: Upgraded from gpt-4o-mini to gpt-4o
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 1800,
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
    if (err.code === "LIMIT_FILE_SIZE") return res.status(400).json({ error: "Video is too large. Max size is 250MB." });
    return res.status(400).json({ error: err.message || "Upload failed." });
  }
  if (err) return res.status(400).json({ error: err.message || "Request failed." });
  return res.status(500).json({ error: "Unknown server error." });
});

// ─── Start Server ─────────────────────────────────────────────────────────────

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Barrel Pro AI Server running on port ${PORT}`);
});
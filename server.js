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
// Smart strategic 30-frame selection targeting the most important moments:
// alley approach, barrel approaches, apexes, exits, and home run.

function selectFramePaths(sampledFrames, maxFrames = 30) {
  const usable = (sampledFrames || [])
    .filter((f) => f?.read_success && (f?.overlay_image_path || f?.image_path))
    .map((f) => ({
      path: f.overlay_image_path || f.image_path,
      percent: f.percent ?? 0,
      dense: !!f.dense_pass,
    }))
    .filter((f) => f.path);

  if (usable.length === 0) return [];
  if (usable.length <= maxFrames) return usable.map((f) => f.path);

  // Strategic zones as % of run — each zone gets a frame allocation
  const zones = [
    { min: 0.00, max: 0.15, count: 2 },  // alley
    { min: 0.13, max: 0.28, count: 3 },  // barrel 1 approach
    { min: 0.26, max: 0.38, count: 4 },  // barrel 1 apex
    { min: 0.36, max: 0.44, count: 2 },  // barrel 1 exit
    { min: 0.40, max: 0.52, count: 3 },  // barrel 2 approach
    { min: 0.50, max: 0.60, count: 4 },  // barrel 2 apex
    { min: 0.58, max: 0.65, count: 2 },  // barrel 2 exit
    { min: 0.62, max: 0.74, count: 3 },  // barrel 3 approach
    { min: 0.72, max: 0.82, count: 4 },  // barrel 3 apex
    { min: 0.80, max: 0.88, count: 2 },  // barrel 3 exit
    { min: 0.86, max: 1.00, count: 1 },  // home run
  ];

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

  // Fill remaining slots with evenly spaced frames
  if (selected.size < maxFrames) {
    const remaining = maxFrames - selected.size;
    for (let i = 0; i < remaining; i++) {
      const idx = Math.round((i * (usable.length - 1)) / Math.max(remaining - 1, 1));
      selected.add(usable[idx].path);
    }
  }

  return [...selected].slice(0, maxFrames);
}

function buildImageInputs(framePaths) {
  // Read frames one at a time to avoid loading all 30 into memory simultaneously
  // This prevents OOM crashes on Render when processing large frame sets
  const inputs = [];
  for (const p of framePaths) {
    try {
      const b64 = fs.readFileSync(p).toString("base64");
      inputs.push({
        type: "image_url",
        image_url: {
          url: `data:image/jpeg;base64,${b64}`,
          detail: "low",
        },
      });
    } catch (err) {
      console.warn("[FRAMES] Could not read frame:", p, err.message);
    }
  }
  console.log(`[FRAMES] Loaded ${inputs.length} of ${framePaths.length} frames for GPT-4o`);
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

const BARREL_RACING_KNOWLEDGE_BASE = `
=== BARREL RACING EXPERT COACHING KNOWLEDGE ===
You are an elite barrel racing coach. Apply this knowledge to every analysis.

── CORE PHILOSOPHY ────────────────────────────────────────────────────────────
"Speed is a byproduct of correctness. If you fix the mechanics, the clock will follow."
"A great run is a conversation, not a fight. The best riders do 90% of their work in the training pen so that during the 15-second run, they can simply stay out of the horse's way."
The straightaways are for speed. The turns are for precision. If you don't set up the approach, you've lost the turn before you even get there.

── THE APPROACH & THE POCKET ──────────────────────────────────────────────────
The turn doesn't start AT the barrel — it starts 15 feet before it.
Use a J-shape approach: aim 5-8 feet to the side of the barrel, never straight at it. Running straight forces a hard check that kills momentum.
The Pocket: Maintain 4-6 feet of lateral clearance. This gives the horse room to move its ribcage without hitting the barrel. Too tight (under 2 feet) swings the hind end out and hits the backside. Too wide wastes time.
The First Barrel is the "money barrel" — highest speed entry. Sets up the entire run.
Second and Third Barrels: Cross-fire your vision. As you leave each barrel, look at the ENTRY POINT of the next barrel, not the barrel itself.

── THE RATE POINT ─────────────────────────────────────────────────────────────
Rate (shift weight from front end to hocks) when the horse's nose is even with the barrel.
Sit deep — "pockets in the dirt." Sink weight into seat bones. This is the physical cue for the horse to shift weight to its hocks and prepare to turn.
Stop driving with your legs. Use a slight command rein to ask the horse to break at the poll and gather its stride.
Rate too early = lose speed. Rate too late = "go long" (overshoot the turn, have to hook back).
"Crating" fix for going long: Stop the horse at the rate point every time in practice until they learn to wait for your seat.

── THE TURN ───────────────────────────────────────────────────────────────────
Keep the horse's inside shoulder UP. A dropped shoulder causes slicing, loss of arc, and knocked barrels.
The horse should bend around your inside leg — ribcage arc ensures hind feet follow front feet in a single track for maximum traction.
Wait until your leg (cinch) passes the barrel before committing to the turn.
Snap your eyes to the next barrel at the 3/4 mark of the turn — this leads the horse's momentum out of the turn.
Never look at the barrel itself. Look 10 feet past it on approach, then snap to the next destination mid-turn.
Finish the turn — leaving too early causes a wide exit and poor line to the next barrel.

── RIDER HAND POSITION ────────────────────────────────────────────────────────
Keep hands LOW. High hands = high-headed horse. A high-headed horse cannot see the ground or turn efficiently.
Inside hand: Guides the nose. Creates a soft arc — never a sharp pull. "Sawing" on the bit makes horses nervous and braced.
Outside hand: The "Wall." Keeps the horse from drifting out and keeps the shoulder upright. This is often more important than the inside hand.
One-handed guide: Barrel racing is more about legs and seat than steering. If the horse won't turn without heavy hands, return to lateral work.
Never hunt for the horn until AFTER you have sat deep. Grabbing the horn early pulls the chest forward, tips weight to the front end, pushes nose into the barrel.

── RIDER SEAT & BODY POSITION ─────────────────────────────────────────────────
Straightaways: Stand slightly in stirrups (two-point position) to take weight off horse's back, allowing full stride extension.
Rate point: Sit deep in pockets — this is a physical cue, not just balance.
Through the turn: Spine aligned with horse's spine. Do NOT lean (motorcycle lean) — shifting weight to inside shoulder causes diving and slicing.
After the turn: Forward and athletic to push the horse to the next spot. Getting left behind (leaning back during acceleration) hits horse in the mouth.

── COMMON FAULTS & EXACT FIXES ────────────────────────────────────────────────

1. DIVING INTO THE TURN (Pocket Killer #1)
Leaning body or pulling horse's nose in too early → horse shoulders in, hits barrel or loses momentum.
Fix drill: "Square the Barrel" — approach at a trot, ride a literal square around the barrel. Only turn when you can see the backside. Keep shoulders upright past the rate point.

2. LOOKING AT THE BARREL
Staring at the barrel shifts body weight, drops shoulder, horse follows eyes into collision.
Fix drill: "Horizon Focus" — focus on a spot 10 feet past the barrel on approach. Don't look at the next barrel until your horse's hip has cleared the current one.

3. OVER-HANDLING / SAWING ON THE BIT
Pulling inside rein throughout the turn pulls horse off-balance, prevents hindquarter power.
Fix drill: "One-Handed Guide" — work the pattern using only dominant hand. Use legs and seat to steer. If horse won't respond, return to lateral work (side-passing, leg-yielding).

4. SHOULDERING IN
Horse leans into barrel, knocks it with shoulder or rider's knee.
Fix: More inside leg, more outside rein. Keep horse "square" until turning point.
Fix drill: "Counter-Bending Circles" — circle barrel while bending horse's nose AWAY from barrel. Lifts inside shoulder, engages hindquarters.

5. GOING LONG (Overshooting)
Horse runs past barrel, has to hook back. Caused by failure to rate.
Fix drill: "Transition Points" — pick a point 15 feet before barrel, every time transition from lope to trot. Builds rate muscle memory.

6. FAILING TO FINISH THE TURN (Early Exit)
Leaving too early causes wide exit, poor line to next barrel.
Fix drill: "One-and-a-Half" — make a full circle around barrel PLUS another half-turn before heading to next. Teaches horse to keep turning until told to leave.

7. IMPROPER POCKET SIZING
Too tight or too wide ruins approach angle and momentum.
Fix drill: "Pinwheel" — set 4 cones around barrel at 5-foot intervals. Spiral in and out at a trot to master spatial awareness of the pocket.

8. GETTING AHEAD OF THE HORSE
Leaning forward over neck before turn is finished → horse stumbles, loses hind-end engagement.
Fix drill: "Deep Sit Stop" — lope the pattern and ask for complete stop at backside of every barrel. Must sit deep to cue the stop.

9. LACK OF RATE (Running Past)
Horse blows past barrel because it didn't gear down.
Fix: Consistent rate cue every practice run. Horse must learn to wait for the seat.

10. INCONSISTENT ALLEYWAY BEHAVIOR
Fighting in the alley causes stressed entry, poor alignment.
Fix drill: "Quiet Entry" — walk into alley, stop, back up, sit quietly until horse exhales, then walk out. Alley must represent calm, not chaos.

── SPLIT TIME INTERPRETATION ──────────────────────────────────────────────────
Slow alley-to-first: Late to rate, approach angle too straight, wrong rate point, or alley stress.
Slow first-to-second or second-to-third: Horse not driving between barrels. Rider sitting back (getting left behind), horse still in rate mode, not extending on the straightaway.
Slow third-to-home: Horse not rated out cleanly, tired, or rider not in two-point pushing forward.
One slow split vs others: Problem is specific to that barrel — check turn grade, approach angle, and exit drive data for that barrel specifically.
A run that feels fast but clocks slow = wide turns. A run that feels slow but clocks well = tight, efficient turns.

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

You are analyzing ${riderName}'s run on ${horseName}. You have 30 video frames covering the entire run — alley approach, all three barrel approaches, apexes, exits, and the home run. You also have detailed computer vision data.

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
- Is the horse running too wide (losing time) or dangerously tight?
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
  "speedInsight": "Name the slowest section and explain what the data tells you about why",
  "splitAnalysis": "Read each split time using the knowledge base — what does each split tell you about what happened at that barrel",
  "patternNotes": "What the approach angles, tightness grades, and exit drives tell you about how this horse runs the pattern — use proper terminology",
  "visualObservations": "What you specifically saw in the video frames — rider position, horse shoulder, hands, seat, head position. Reference what you actually saw.",
  "accuracyNotes": "Honest note on data quality and what you could see clearly vs what was estimated",
  "strengths": ["Specific strength tied to data or what you saw in frames", "Another specific strength with detail"],
  "issues": ["Specific issue with barrel name, grade, and visual observation — e.g. Motorcycle lean at the second barrel — rider's upper body clearly leaning in, dropping the horse's inside shoulder (Grade D turn, 87px wide)", "Another specific issue with detail"],
  "workOns": ["Specific work-on tied to an observed problem with proper terminology", "Another targeted work-on"],
  "drills": ["Specific drill that directly addresses an observed issue — e.g. Walk and trot the pattern focusing on sitting deep the moment your horse's nose hits the barrel", "Another targeted drill tied to what you saw"]
}
  `.trim();
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

Return ONLY this JSON:

{
  "summary": "2-3 sentences at the gate. Reference rider's own feedback with proper barrel racing terminology.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "The single most important coaching cue using proper terminology",
  "speedInsight": "Where time is likely being lost — use the knowledge base to explain why",
  "splitAnalysis": "Read this time in context of their history using knowledge base interpretation",
  "patternNotes": "Based on rider feedback and history — what pattern tendencies are showing up, explained with proper terminology",
  "visualObservations": "No video available — note this field is based on rider feedback only",
  "accuracyNotes": "Coaching from rider feedback and history only — no video data",
  "strengths": ["Specific strength from feedback or history", "Another specific strength"],
  "issues": ["Specific issue using proper barrel racing terminology", "Another specific issue"],
  "workOns": ["Specific work-on with proper terminology", "Another targeted work-on"],
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
    visualObservations: parsed.visualObservations || null,
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

    updateJob(jobId, { progress: 8, stage: "Running computer vision — tracking horse and barrels" });
    const pythonResult = await runPythonAnalysis(videoPath, job.run);
    pythonGeneratedPaths = getPythonGeneratedPaths(pythonResult);

    updateJob(jobId, { progress: 55, stage: "Computer vision complete — selecting key frames" });
    const framePaths = selectFramePaths(pythonResult.sampled_frames || [], 30);
    if (!framePaths.length) throw new Error("Python did not return any usable frame images.");

    updateJob(jobId, { progress: 62, stage: "Preparing frames for AI coach" });
    const imageInputs = buildImageInputs(framePaths);

    updateJob(jobId, { progress: 70, stage: "Sending to AI coach — analyzing your run" });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before AI analysis.");

    // IMPROVEMENT #3: Upgraded from gpt-4o-mini to gpt-4o for significantly better
    // coaching quality, reasoning depth, and barrel racing expertise.
    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 3000,
      messages: [{
        role: "user",
        content: [
          { type: "text", text: buildVideoPrompt(latestJob.run, pythonResult) },
          ...imageInputs,
        ],
      }],
    });

    updateJob(jobId, { progress: 90, stage: "Finalizing coaching feedback" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      // Try to recover partial JSON
      const recovered = extractLastJsonObject(outputText);
      if (recovered && recovered.summary) {
        parsedAnalysis = recovered;
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
      messages: [{ role: "user", content: buildTextOnlyPrompt(latestJob.run) }],
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
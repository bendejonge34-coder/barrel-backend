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
      const normalizedCwd = path.resolve(process.cwd());
      if (!normalized.startsWith(normalizedCwd)) {
        console.warn("[CLEANUP] Skipping path outside project:", normalized);
        continue;
      }
      const stats = fs.statSync(normalized);
      if (stats.isDirectory()) {
        fs.rmSync(normalized, { recursive: true, force: true });
      } else {
        fs.unlinkSync(normalized);
      }
    } catch (err) {
      console.warn("[CLEANUP] Warning:", err.message);
    }
  }
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

  let vsPersonalBest = "";
  if (best && !isNaN(currentTime)) {
    const diff = (currentTime - best).toFixed(3);
    if (diff <= 0) vsPersonalBest = `This is a NEW PERSONAL BEST by ${Math.abs(diff)}s!`;
    else vsPersonalBest = `This run is ${diff}s off the personal best.`;
  }

  const arenaHistory = horseRuns
    .filter(r => r.arenaCondition)
    .slice(0, 5)
    .map(r => `${r.arenaCondition}: ${r.time}s`)
    .join(", ");

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

const currentRunNotes = [];
      if (run.riderFeedback) currentRunNotes.push(`Rider felt: "${run.riderFeedback}"`);
      if (run.notes) currentRunNotes.push(`Notes: "${run.notes}"`);
      const currentRunNotesText = currentRunNotes.join(" | ");

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

Notes for THIS run only:
  ${currentRunNotesText || "No notes for this run"}
`.trim();
}

// ─── Arena Distance / Split Normalization ─────────────────────────────────────

const PATTERN_A_DISTANCES = {
  "Start to 1st Barrel":  60,
  "1st to 2nd Barrel":    90,
  "2nd to 3rd Barrel":   105,
  "3rd Barrel to Finish": 120,
};

function getArenaDistances(run) {
  const d = run?.arenaDistances;
  if (d && d.startToB1 && d.b1ToB2 && d.b2ToB3 && d.b3ToFinish) {
    return {
      "Start to 1st Barrel":  Number(d.startToB1),
      "1st to 2nd Barrel":    Number(d.b1ToB2),
      "2nd to 3rd Barrel":    Number(d.b2ToB3),
      "3rd Barrel to Finish": Number(d.b3ToFinish),
    };
  }
  return PATTERN_A_DISTANCES;
}

function normalizeSplitsToSpeed(splits, distances) {
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

// ─── Barrel Racing Knowledge Base ─────────────────────────────────────────────

const BARREL_RACING_KNOWLEDGE_BASE = `
=== BARREL RACING EXPERT COACHING KNOWLEDGE ===
You are an elite barrel racing coach. Apply this knowledge to every analysis.

── CORE PHILOSOPHY ────────────────────────────────────────────────────────────
"Speed is a byproduct of correctness. If you fix the mechanics, the clock will follow."
The straightaways are for speed. The turns are for precision. If you don't set up the approach, you've lost the turn before you even get there.
The turn doesn't start AT the barrel — it starts 15 feet before it. Every single time.

── THE APPROACH & THE POCKET ──────────────────────────────────────────────────
THE FIRST BARREL — "The Money Barrel"
Highest speed entry of the run. Sets up everything that follows.
Use a "J" approach — aim for a point 5 to 8 feet to the side of the barrel. Never run straight at it.
The Pocket: The horse needs enough room to arc its ribcage cleanly through the turn. Too tight = hind end clips the backside of the barrel.

THE SECOND AND THIRD BARRELS — Cross-Firing Your Vision
These require "cross-firing" your vision.
As you leave the first barrel, look at the ENTRY POINT of the second barrel — not the barrel itself.
"Don't look at the barrel. If you look at the barrel, you'll hit it. Look where you want the horse's feet to go."

── THE RATE POINT ─────────────────────────────────────────────────────────────
Rate = shifting weight from the front end to the hocks to prepare for the turn.
Cue: As the horse's nose reaches the barrel, "sit deep in your pockets." Sink your weight into your seat bones.
Rate too early = lose speed unnecessarily. Rate too late = "going long" — overshooting the turn, losing both time and line.

── THE TURN ───────────────────────────────────────────────────────────────────
Keep the horse's inside shoulder UP. A dropped shoulder causes slicing, loss of arc, and knocked barrels.
The horse should bend around your inside leg — ribcage arc ensures hind feet follow front feet in a single track for maximum traction.
Do NOT commit to the turn until your leg (cinch area) has passed the barrel.
Finish the turn completely. Leaving early causes a poor line to the next barrel.

── RIDER HAND POSITION ────────────────────────────────────────────────────────
Keep hands LOW at all times. High hands = high-headed horse.
Inside Hand: Guides the nose. Creates a soft arc into the turn — never a sharp pull.
Outside Hand: "The Wall." Keeps the horse from drifting out and keeps the inside shoulder upright.
Never "hunt" for the horn until AFTER you have sat deep in your seat.

── RIDER SEAT & BODY POSITION ─────────────────────────────────────────────────
Straightaways: Stand slightly in your stirrups (two-point position). Let the horse run.
The Rate: Sit DEEP in your pockets. This IS the physical cue for the horse to shift weight to its hocks.
Through the turn: Spine aligned with horse's spine. Do NOT lean (motorcycle lean).
After the turn: Immediately forward and athletic. Push the horse to the next spot.

── COMMON FAULTS & EXACT FIXES ────────────────────────────────────────────────

1. DIVING INTO THE TURN (Pocket Killer #1)
Riders lean their body or pull the horse's nose toward the barrel too early.
Fix drill: "Square the Barrel" — Approach at a TROT. Ride a literal square around the barrel instead of a circle. Forces the rider to wait before turning.

2. LOOKING AT THE BARREL
Your horse follows your eyes. If you stare at the barrel, your body weight shifts and the horse follows into a collision.
Fix drill: "The Look Ahead / Horizon Drill" — Focus on a spot 10 feet PAST each barrel as you approach it.

3. SHOULDERS DROPPING ("Washing Out")
Horse drops inside shoulder, losing leverage and power in the turn.
Fix drill: "Counter-Bending Circles" — Circle the barrel while bending the horse's nose AWAY from the barrel. Lifts inside shoulder and engages hindquarters.

4. OVER-HANDLING / SAWING ON THE BIT
Riders "saw" on the bit or pull the inside rein throughout the entire turn.
Fix drill A: "The One-Handed Guide" — Work the pattern at a trot or slow lope using ONLY your dominant riding hand.
Fix drill B: "Loose Rein Loping" — Lope a large circle around the barrel on a completely loose rein, using only weight and legs.

5. FAILING TO FINISH THE TURN (Early Exit)
Leaving the barrel too early, resulting in a compromised line to the next barrel.
Fix drill: "The One-and-a-Half" — Make a full circle around the barrel PLUS another half-turn before heading to the next.

6. IMPROPER POCKET — RUNNING TOO TIGHT
Horse clips the barrel with shoulder, hip, or hind end.
Fix drill: "The Pinwheel" — Set up 4 cones around a barrel at 5-foot intervals. Practice spiraling in and out at a trot.

7. GETTING AHEAD OF THE HORSE
Leaning forward over the neck before the horse has finished the turn.
Fix drill: "The Deep Sit Stop" — Lope the pattern and ask for a complete stop at the backside of every barrel.

8. LACK OF RATE (Running Past)
The horse "blows" past the barrel because they didn't gear down.
Fix drill: "Transition Points" — Pick a point 15 feet before the barrel. Every time you hit that point, transition from a lope to a trot. Builds rate muscle memory.

9. SHOULDERING IN
Horse leans into barrel, knocks it with shoulder or rider's knee.
Fix: More inside leg at the girth, more outside rein.
Fix drill: "Counter-Bending Circles" (same as fault #3).

10. INCONSISTENT ALLEYWAY BEHAVIOR
Fighting in the alley causes stressed entry, poor alignment, rushed approach.
Fix drill: "Quiet Entry" — Walk into alley, stop, back up, sit quietly until horse exhales, then walk out calmly. Repeat.

── FAULT DIAGNOSIS FROM SPLIT DATA ────────────────────────────────────────────
Slow alley→1st: Alley stress, wrong approach angle, failure to rate at the right point, or diving too early.
Slow 1st→2nd or 2nd→3rd: Horse not driving between barrels. Rider getting left behind. Horse still in rate mode — not extending on the straightaway.
Slow 3rd→home: Rider not in two-point pushing forward. Horse not rated out cleanly. Fatigue.
One split dramatically slower than others: Problem is SPECIFIC to that barrel.
All splits slow but consistent: Horse not extending on straightaways, OR rider restraining horse through entire run.
`;

// ─── CV Coaching Data Builder ─────────────────────────────────────────────────
// Converts Python's CV output into a clean text summary for the AI prompt.

function buildBarrelCoachingData(run, pythonResult) {
  const speedSummary = pythonResult?.speed_summary || null;
  const splits = pythonResult?.splits || {};
  const insights = Array.isArray(pythonResult?.insights) ? pythonResult.insights : [];

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

  // Prefer manual splits (ground truth) over CV splits (estimates)
  const manualSplits = run?.manualSplits || null;
  const activeSplits = manualSplits || splits || {};
  const splitsSource = manualSplits ? "user-marked and scaled to official time" : "CV-estimated";

  let s1 = activeSplits?.start_to_barrel1_seconds ?? null;
  let s2 = activeSplits?.barrel1_to_barrel2_seconds ?? null;
  let s3 = activeSplits?.barrel2_to_barrel3_seconds ?? null;
  let s4 = activeSplits?.barrel3_to_home_seconds ?? null;

  // Scale CV splits to official run time if no manual splits
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

  const splitMap = {
    "Start to 1st Barrel":  s1,
    "1st to 2nd Barrel":    s2,
    "2nd to 3rd Barrel":    s3,
    "3rd Barrel to Finish": s4,
  };
  const arenaDist = getArenaDistances(run);
  const normalized = normalizeSplitsToSpeed(
    Object.fromEntries(Object.entries(splitMap).filter(([, v]) => v != null && Number.isFinite(Number(v)))),
    arenaDist
  );
  const slowestBySpeed = normalized[0] || null;
  const fastestBySpeed = normalized[normalized.length - 1] || null;

  const splitReport = normalized.length > 0
    ? `Split times by speed — ${splitsSource} (Pattern A distances):
${normalized.map(s => `  - ${s.label}: ${s.seconds.toFixed(2)}s / ${s.distance}ft = ${s.speed.toFixed(1)} ft/s`).join("\n")}
${slowestBySpeed ? `  ► SLOWEST by speed: ${slowestBySpeed.label} at ${slowestBySpeed.speed.toFixed(1)} ft/s` : ""}
${fastestBySpeed ? `  ► FASTEST by speed: ${fastestBySpeed.label} at ${fastestBySpeed.speed.toFixed(1)} ft/s` : ""}`
    : "Split times: not available";

  // Barrel proximity summary from frame metrics
  const frameMetrics = Array.isArray(pythonResult?.frame_metrics) ? pythonResult.frame_metrics : [];
  const barrelProximity = ["barrel1", "barrel2", "barrel3"].map(name => {
    const distKey = `dist_to_${name}_px`;
    const dists = frameMetrics.filter(m => m[distKey] != null).map(m => m[distKey]);
    if (dists.length === 0) return `${name}: no proximity data`;
    const minDist = Math.min(...dists);
    const label = { barrel1: "1st", barrel2: "2nd", barrel3: "3rd" }[name];
    return `  - ${label} barrel: closest approach ${Math.round(minDist)}px from center`;
  }).join("\n");

  return `
=== COMPUTER VISION COACHING DATA ===

Run overview:
- Duration: ${pythonResult?.duration_seconds ?? "unknown"}s
- Pattern direction: ${pythonResult?.pattern_direction ?? "unknown"}-first
- Horse tracked: ${pythonResult?.horse_detected_frames ?? "?"} frames
- Total frames analyzed: ${pythonResult?.tracking_quality?.sampled_frame_count ?? "?"}

${splitReport}

${speedReport}

Barrel proximity (how close horse got to each barrel center):
${barrelProximity}

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
- Manual splits: ${run?.manualSplits ? "YES — highly accurate" : "No"}
  `.trim();
}

// ─── AI Prompt Builders ───────────────────────────────────────────────────────

function buildCoachingPrompt(run, pythonResult) {
  const coachingData = buildBarrelCoachingData(run, pythonResult);
  const historicalContext = buildHistoricalContext(run);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";

  // Penalty time calculation
  const baseTime = parseFloat(run?.time) || 0;
  const penaltySeconds = run?.knockedPenalty === "+5" && run?.knockedBarrels?.length > 0
    ? run.knockedBarrels.length * 5 : 0;
  const officialTime = baseTime + penaltySeconds;

  // Normalize splits for the prompt
  const manualSplits = run?.manualSplits || null;
  const cvSplits = pythonResult?.splits || {};
  const rawSplits = manualSplits || cvSplits;
  let sp1 = rawSplits?.start_to_barrel1_seconds ?? null;
  let sp2 = rawSplits?.barrel1_to_barrel2_seconds ?? null;
  let sp3 = rawSplits?.barrel2_to_barrel3_seconds ?? null;
  let sp4 = rawSplits?.barrel3_to_home_seconds ?? null;

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

  const splitRawMap = {
    "Start to 1st Barrel":  sp1,
    "1st to 2nd Barrel":    sp2,
    "2nd to 3rd Barrel":    sp3,
    "3rd Barrel to Finish": sp4,
  };
  const arenaDist = getArenaDistances(run);
  const normalizedSections = normalizeSplitsToSpeed(
    Object.fromEntries(Object.entries(splitRawMap).filter(([, v]) => v != null && Number.isFinite(Number(v)))),
    arenaDist
  );
  const slowestSplit = normalizedSections[0] || null;
  const fastestSplit = normalizedSections[normalizedSections.length - 1] || null;
  const hasSplits = normalizedSections.length > 0;
  const splitsLabel = manualSplits ? "user-marked, scaled to official time" : "CV-estimated, scaled to official time";

  return `${BARREL_RACING_KNOWLEDGE_BASE}

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
Arena distances — Start→B1: ${arenaDist["Start to 1st Barrel"]}ft | B1→B2: ${arenaDist["1st to 2nd Barrel"]}ft | B2→B3: ${arenaDist["2nd to 3rd Barrel"]}ft | B3→Finish: ${arenaDist["3rd Barrel to Finish"]}ft${run?.arenaDistances ? " (user-entered)" : " (WPRA Pattern A defaults)"}

${normalizedSections.map(s =>
  `  ${s.label}: ${s.seconds.toFixed(2)}s over ${s.distance}ft = ${s.speed.toFixed(1)} ft/s`
).join("\n")}

► SLOWEST BY SPEED: "${slowestSplit.label}" at ${slowestSplit.speed.toFixed(1)} ft/s — THIS IS WHERE THE HORSE WAS GENUINELY SLOWEST
► FASTEST BY SPEED: "${fastestSplit.label}" at ${fastestSplit.speed.toFixed(1)} ft/s

NOTE: Slowest is determined by ft/s, not raw time. Base all coaching on the ft/s ranking above.` : "No splits available — use rider feedback and run data only."}

${coachingData}

${historicalContext}

=== YOUR TASK ===
You are a barrel racing coach analyzing ${riderName}'s run on ${horseName}.
You have detailed computer vision data including split times, barrel proximity, and speed profile.
No video frames are available — coach entirely from the CV data and rider feedback above.

FAULT FRAMEWORK — use these terms when they apply:
- Rider: "The Tilter" (leaning in) | "The Reacher" (hands forward/pulling)
- Horse body: "Log Stiff" (no arc) | "Shoulder-Pop" (shoulder pushes out)
- Rate: "Downhill Run" (on forehand) | "Stiff-Legged" (choppy, no collection)
- Exit: "The Fishtail" (hind swings out) | "Wandering" (drifts off line)

RULES:
- Identify what went wrong at the SLOWEST split
- Explain why using the fault framework and knowledge base
- Give specific, executable fixes with named drills
- Reference the slowest split by name and ft/s value
- Use proper barrel racing terminology throughout
- Call barrels by name: first barrel, second barrel, third barrel
- If rider feedback describes a problem, connect it to the correct fault term
- NO generic advice — every point must be specific to this run's data
- Do NOT say a turn was "wide" or "narrow" — ever
- Do NOT mention degrees, angles, or pixel measurements
- Do NOT contradict the split data
- If only 1 or 2 genuine time losses exist, return 1 or 2 — do not pad
- Return ONLY valid JSON. No markdown. No extra text.

Return ONLY this JSON:
{
  "summary": "2-3 sentences. Reference the official time, the show or location if given, the slowest split by name and ft/s value, and the primary fault. Sound like a coach at the gate — direct and specific.",
  "timeLost": [
"State ONLY what the split data directly proves. Name the section, the time, the ft/s value, and what that number means for a horse running barrels. No speculation about cause.",
    "A second section only if the data shows a second genuine time loss. Must reference a specific split value and section name.",
    "Third entry only if a third distinct loss exists in the data — otherwise omit this entry entirely."  
    ],
"improvements": [
    "DIRECTLY fixes the first time loss above. Start with the barrel or section name. Name the exact drill from the knowledge base. Tell the rider precisely how to execute it at their next practice — what to do, where to do it, how many times.",
    "DIRECTLY fixes the second time loss above. Must be a different drill than improvement 1. Specific and executable. If no second time loss was identified, omit this entry.",
    "DIRECTLY fixes the third time loss above only if that entry exists — otherwise omit entirely."
  ]
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
- Every sentence must be useful. No filler.
- Do NOT say a turn was "wide" or "narrow" — ever
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
  "summary": "2-3 sentences. Reference the actual time, show/location, and what the rider felt. Sound like a real coach at the gate — specific to this run.",
  "timeLost": [
    "Time loss 1 — identify the specific section, what likely happened there based on rider feedback, explained with barrel racing terminology.",
    "Time loss 2 — a DIFFERENT section or fault from point 1. Must be supported by data or feedback.",
    "Third time loss only if a third genuine loss is identifiable — otherwise omit."
  ],
  "improvements": [
    "Directly fixes time loss 1 — name a specific drill and how to execute it.",
    "Directly fixes time loss 2 — specific and executable at their next training session.",
    "Directly fixes time loss 3 if applicable."
  ]
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
    insights: Array.isArray(pythonResult.insights) ? pythonResult.insights : [],
    highlights: pythonResult.highlights || null,
    speed_summary: pythonResult.speed_summary || null,
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

  try {
    updateJob(jobId, { status: "running", progress: 5, stage: "Starting analysis", startedAt: new Date().toISOString() });

    // ── Slow ticker while Python runs ─────────────────────────────────────
    // Python takes 30-60s. Without this the bar sits frozen at 5%.
    // Ticks from 5 → 48 over ~50 seconds so the user sees steady movement.
    let currentProgress = 5;
    const ticker = setInterval(() => {
      const j = jobs.get(jobId);
      if (!j || j.status !== "running") {
        clearInterval(ticker);
        return;
      }
      if (currentProgress < 48) {
        currentProgress += 1;
        updateJob(jobId, { progress: currentProgress, stage: "Running computer vision — tracking horse and barrels" });
      } else {
        clearInterval(ticker);
      }
    }, 1000); // ticks every 1 second

    let pythonResult;
    try {
      pythonResult = await runPythonAnalysis(videoPath, job.run);
    } finally {
      clearInterval(ticker); // always stop ticker when Python finishes
    }

    updateJob(jobId, { progress: 55, stage: "Computer vision complete — building coaching report" });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before AI analysis.");

    updateJob(jobId, { progress: 70, stage: "AI coach analyzing your run data" });

    const prompt = buildCoachingPrompt(latestJob.run, pythonResult);

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 4000,
      messages: [
        {
          role: "system",
          content: "You are an elite barrel racing coach. You NEVER use the words 'wide', 'wider', or 'narrow' to describe turns or approaches. You NEVER use labels like PRIMARY, SECONDARY, timeLost[0], or timeLost[1] in your responses. You always return valid JSON only. No markdown, no preamble."
        },
        {
          role: "user",
          content: prompt,
        }
      ],
    });

    updateJob(jobId, { progress: 92, stage: "Finalizing coaching feedback" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
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
        frameCount: 0,
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
    safeCleanup([videoPath]);
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
      model: "gpt-4o",
      max_tokens: 3000,
      messages: [
        {
          role: "system",
          content: "You are an elite barrel racing coach. You NEVER use the words 'wide', 'wider', or 'narrow' to describe turns or approaches. You NEVER use labels like PRIMARY, SECONDARY, timeLost[0], or timeLost[1] in your responses. You always return valid JSON only. No markdown, no preamble."
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

app.post("/compare-runs", async (req, res) => {
  try {
    const { runA, runB, splitsA, splitsB } = req.body;
    if (!runA || !runB) return res.status(400).json({ error: "Two runs required." });

    const buildDesc = (run, splits, label) => {
      let desc = `${label}: ${run.horse || "Unknown"}, Time: ${run.time}s`;
      if (run.showName) desc += `, Show: ${run.showName}`;
      if (run.arenaCondition) desc += `, Arena: ${run.arenaCondition}`;
      if (run.placing) desc += `, Placing: ${run.placing}`;
      if (splits) desc += `. Splits — Start to B1: ${splits.s1}s, B1 to B2: ${splits.s2}s, B2 to B3: ${splits.s3}s, B3 to Finish: ${splits.s4}s`;
      return desc;
    };

    const prompt = `You are analyzing two barrel racing runs. State only facts from the data provided. Do not guess at causes, do not give coaching advice, do not speculate. Only describe what the numbers show.\n\n${buildDesc(runA, splitsA, "Run A")}\n${buildDesc(runB, splitsB, "Run B")}\n\nWrite 3-5 short factual sentences comparing these two runs. Only state what the numbers directly show.`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      messages: [{ role: "user", content: prompt }],
      max_tokens: 300,
    });

    const summary = response.choices?.[0]?.message?.content || "Could not generate summary.";
    res.json({ summary });
  } catch (err) {
    console.error("[COMPARE] Error:", err.message);
    res.status(500).json({ error: "Failed to generate comparison summary." });
  }
});
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
    if (err.code === "LIMIT_FILE_SIZE") return res.status(400).json({ error: "Video file is too large (over 600MB). Please switch your camera to 1080p at 30fps — a 30-second barrel run should be under 150MB." });
    return res.status(400).json({ error: err.message || "Upload failed." });
  }
  if (err) return res.status(400).json({ error: err.message || "Request failed." });
  return res.status(500).json({ error: "Unknown server error." });
});

// ─── Start Server ─────────────────────────────────────────────────────────────

app.listen(PORT, "0.0.0.0", () => {
  console.log(`Barrel Pro AI Server running on port ${PORT}`);
});
import cors from "cors";
import dotenv from "dotenv";
import express from "express";
import fs from "fs";
import multer from "multer";
import OpenAI from "openai";
import path from "path";
import { Resend } from "resend";

dotenv.config();

// ─── Config ───────────────────────────────────────────────────────────────────

const PORT = Number(process.env.PORT || 3001);
const JOB_TTL_MS = 1000 * 60 * 60;

const uploadsDir = path.join(process.cwd(), "uploads");
const JOB_STORE_FILE = path.join(process.cwd(), "job-store.json");

// ─── Boot Log ─────────────────────────────────────────────────────────────────

console.log("===== SERVER START =====");
console.log("[BOOTED]", new Date().toISOString());
console.log("Port:", PORT);
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

// ─── Historical Context Builder ───────────────────────────────────────────────

function buildHistoricalContext(run) {
  const history = run?.runHistory || [];
  const horseName = run?.horse || "this horse";

  if (!history || history.length === 0) {
    return `No previous run history available for ${horseName}.`;
  }

  const horseRuns = history.filter(r => r.horse === horseName && r.time && !isNaN(parseFloat(r.time)));

  if (horseRuns.length === 0) {
    return `No previous timed runs found for ${horseName}.`;
  }

  const times = horseRuns.map(r => parseFloat(r.time)).filter(t => !isNaN(t));
  const currentTime = parseFloat(run?.time);

  const best = times.length > 0 ? Math.min(...times) : null;
  const avg = times.length > 0 ? (times.reduce((a, b) => a + b, 0) / times.length).toFixed(3) : null;

  const recent3 = times.slice(0, 3);
  const prior3 = times.slice(3, 6);
  let trend = "not enough data yet";
  if (recent3.length >= 2 && prior3.length >= 2) {
    const recentAvg = recent3.reduce((a, b) => a + b, 0) / recent3.length;
    const priorAvg = prior3.reduce((a, b) => a + b, 0) / prior3.length;
    const diff = (recentAvg - priorAvg).toFixed(3);
    if (recentAvg < priorAvg) trend = `getting faster — down ${Math.abs(diff)}s on average recently`;
    else if (recentAvg > priorAvg) trend = `running a bit slower lately — up ${diff}s on average`;
    else trend = "holding steady";
  }

  let vsPersonalBest = "";
  if (best && !isNaN(currentTime)) {
    const diff = (currentTime - best).toFixed(3);
    if (diff <= 0) vsPersonalBest = `New personal best by ${Math.abs(diff)}s.`;
    else vsPersonalBest = `${diff}s off the personal best of ${best}s.`;
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
HISTORY FOR ${horseName.toUpperCase()} (${horseRuns.length} logged runs):
- Personal best: ${best ? best + "s" : "unknown"}
- Average: ${avg ? avg + "s" : "unknown"}
- Trend: ${trend}
- ${vsPersonalBest}
- Arena conditions: ${arenaHistory || "none logged"}
- Recent shows: ${recentShows || "none logged"}
- This run notes: ${currentRunNotesText || "none"}
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
=== BARREL RACING COACHING KNOWLEDGE ===
You are a seasoned barrel racing coach. You talk like you're standing at the gate — direct, plain, and specific to what just happened. No textbook language. No invented labels. Just real talk that a competitive barrel racer would respect and actually use.

── HOW THE PATTERN WORKS ──────────────────────────────────────────────────────
The straightaways are where you make up time. The turns are where you lose it if you're sloppy.
The turn doesn't start at the barrel — it starts about 15 feet out. If you're not set up by then, the barrel already beat you.
First barrel sets the whole run. Get that one right and the rest follows.

── THE APPROACH ───────────────────────────────────────────────────────────────
Don't run straight at the barrel. Angle off to the side — give yourself room to arc around it cleanly.
The pocket is the area just before the barrel where you rate down and set up the turn. Get to the pocket right and the turn takes care of itself.
After the first barrel, stop looking at the next barrel. Look at where you want the horse's feet to go — the entry point, not the barrel itself.

── RATING ─────────────────────────────────────────────────────────────────────
Rating is when the horse shifts its weight back onto its hocks and gears down for the turn.
The cue is sitting deep — sink into your seat bones as the horse's nose reaches the barrel.
Rate too early and you give up speed for nothing. Rate too late and the horse blows past the pocket and you're scrambling to recover.

── THE TURN ───────────────────────────────────────────────────────────────────
Keep that inside shoulder up. A dropped shoulder kills your arc and puts you right into the barrel.
The horse needs to bend through its whole body — ribs, not just neck. If the neck bends but the ribs stay stiff, you're losing traction through the turn.
Don't leave the barrel until your leg — at the cinch — has passed it. Leaving early messes up your line to the next one.
Finish the turn. Riders lose more time leaving early than almost anything else.

── RIDER POSITION ─────────────────────────────────────────────────────────────
On the straightaways, get up off their back a little. Let them run.
Through the turn, sit straight. Don't lean into it like a motorcycle — that shifts weight onto the inside shoulder and slows the horse down.
Hands low. Always. High hands bring the head up and kill collection.
Inside hand guides the nose softly. Outside hand keeps the shoulder from drifting out.
Don't grab the horn until you've already sat deep. Reaching for it early tips your weight forward at the worst time.

── WHAT CAUSES SLOW SPLITS ────────────────────────────────────────────────────
Slow into the first barrel: Usually the approach — came in too fast, wrong angle, or didn't get to the pocket in time. Sometimes alley nerves.
Slow first to second, or second to third: Horse isn't extending on the straightaway. Either the rider is holding them back, or the horse is still in rate mode and hasn't opened back up.
Slow third barrel to home: Rider didn't get forward and push. Horse ran out of gas or wasn't asked.
One section way slower than the rest: Something specific happened at that barrel — worth drilling that turn specifically.
Everything slow and consistent: Horse isn't extending anywhere, or the rider is restraining them throughout.

── DRILLS THAT ACTUALLY WORK ──────────────────────────────────────────────────
Not rating — pick a spot 15 feet out from the barrel. Every time you hit that spot at a lope, transition to a trot. Do it until it's automatic, then speed it up.
Dropping a shoulder or drifting — lope circles around the barrel with the horse's nose tipped slightly away from the barrel. Keeps the shoulder up and gets them bending through the ribs.
Leaving the turn too early — lope the pattern and stop completely at the backside of every barrel. Gets the horse waiting for the cue instead of anticipating.
Diving in or pulling too hard — work the pattern one-handed at a trot. Forces you to guide with your body instead of your hands.
Rushing the approach — walk the pattern. All three barrels. Do it five times before you ever pick up speed. It builds patience and precision.
Drifting off the barrel entry line — set up 4 cones around the barrel in a cross pattern about 5 feet out. Spiral in and out at a trot until the approach is automatic.
`;

// ─── AI Prompt Builders ───────────────────────────────────────────────────────

function buildCoachingPrompt(run) {
  const historicalContext = buildHistoricalContext(run);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";

  const baseTime = parseFloat(run?.time) || 0;
  const penaltySeconds = run?.knockedPenalty === "+5" && run?.knockedBarrels?.length > 0
    ? run.knockedBarrels.length * 5 : 0;
  const officialTime = baseTime + penaltySeconds;

  const manualSplits = run?.manualSplits || null;
  let sp1 = manualSplits?.start_to_barrel1_seconds ?? null;
  let sp2 = manualSplits?.barrel1_to_barrel2_seconds ?? null;
  let sp3 = manualSplits?.barrel2_to_barrel3_seconds ?? null;
  let sp4 = manualSplits?.barrel3_to_home_seconds ?? null;

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

  return `${BARREL_RACING_KNOWLEDGE_BASE}

=== THIS RUN ===
Horse: ${horseName} | Rider: ${riderName}
Time: ${run?.knockedPenalty === "nt" ? "N-T (knocked barrel, no time)" : officialTime > 0 ? officialTime.toFixed(3) + "s" + (penaltySeconds > 0 ? ` (base ${baseTime.toFixed(3)}s + ${penaltySeconds}s penalty)` : "") : "not recorded"}
Show: ${run?.showName || "—"} | Location: ${run?.location || "—"} | Date: ${run?.runDate || "—"}
Arena condition: ${run?.arenaCondition || "not provided"} | Placing: ${run?.placing || "—"}
Knocked barrels: ${run?.knockedBarrels?.length > 0 ? `Barrel ${run.knockedBarrels.join(", ")} — ${run.knockedPenalty === "nt" ? "N-T" : `+${run.knockedBarrels.length * 5}s`}` : "none"}
Rider felt: "${run?.riderFeedback || "no feedback"}"
Notes: "${run?.notes || "none"}"

${hasSplits ? `SPLIT TIMES (user-marked, scaled to official time):
Arena: Start→B1: ${arenaDist["Start to 1st Barrel"]}ft | B1→B2: ${arenaDist["1st to 2nd Barrel"]}ft | B2→B3: ${arenaDist["2nd to 3rd Barrel"]}ft | B3→Finish: ${arenaDist["3rd Barrel to Finish"]}ft${run?.arenaDistances ? " (rider entered)" : " (WPRA Pattern A)"}

${normalizedSections.map(s => `  ${s.label}: ${s.seconds.toFixed(3)}s — ${s.speed.toFixed(1)} ft/s`).join("\n")}

Slowest section by speed: ${slowestSplit ? `"${slowestSplit.label}" at ${slowestSplit.speed.toFixed(1)} ft/s` : "unknown"}
Fastest section by speed: ${fastestSplit ? `"${fastestSplit.label}" at ${fastestSplit.speed.toFixed(1)} ft/s` : "unknown"}

Speed is calculated from feet per second — not raw time. The slowest ft/s is where the horse was genuinely losing ground regardless of section length.` : "No splits available — coach from rider feedback and run data only."}

${historicalContext}

=== YOUR JOB ===
You're the coach at the gate after this run. Talk like it. Short, specific, no fluff.

WHAT THIS MEANS IN PRACTICE:
- Don't say "it appears" or "it seems" — state what the data shows
- Don't say "wide" or "narrow" to describe turns — ever
- Don't use invented labels or categories — just describe what happened in plain language
- Don't repeat the same drill across multiple improvements — each one should be different
- Don't pad — if there's only one real problem, say one thing
- Reference the actual split times and ft/s values when you have them
- If the rider gave feedback, connect it to what the splits show
- Call barrels by name: first barrel, second barrel, third barrel
- Vary your language run to run — don't use the same phrases every time
- Sound like a real person who knows barrel racing, not a textbook

Return ONLY this JSON — no markdown, no extra text:
{
  "summary": "2-3 sentences max. Lead with the time and where this run stood — new PB, off pace, solid run. Then name the one thing that cost the most time, using the split data if available. Gate-side voice — direct and specific.",
  "timeLost": [
    "Name the section, the ft/s value if available, and what that actually means for this horse on this run. No speculation — only what the data directly shows.",
    "Second section only if the data shows a real second loss. Must be a different section from the first.",
    "Third only if genuinely supported by the data — otherwise leave it out."
  ],
  "improvements": [
    "Fix for the first time loss. Name the drill, explain exactly how to run it at their next practice. Be specific — where, how many times, what to feel for.",
    "Fix for the second time loss. Must be a different drill from improvement 1.",
    "Fix for the third only if there was a third time loss."
  ]
}`.trim();
}

function buildTextOnlyPrompt(run) {
  const historicalContext = buildHistoricalContext(run);
  const horseName = run?.horse || "this horse";
  const riderName = run?.rider || "the rider";

  const baseTime = parseFloat(run?.time) || 0;
  const penaltySeconds = run?.knockedPenalty === "+5" && run?.knockedBarrels?.length > 0
    ? run.knockedBarrels.length * 5 : 0;
  const officialTime = baseTime + penaltySeconds;

  return `${BARREL_RACING_KNOWLEDGE_BASE}

=== THIS RUN ===
Horse: ${horseName} | Rider: ${riderName}
Time: ${run?.knockedPenalty === "nt" ? "N-T (knocked barrel, no time)" : officialTime > 0 ? officialTime.toFixed(3) + "s" + (penaltySeconds > 0 ? ` (base ${baseTime.toFixed(3)}s + ${penaltySeconds}s penalty)` : "") : "not recorded"}
Show: ${run?.showName || "—"} | Location: ${run?.location || "—"}
Arena condition: ${run?.arenaCondition || "not provided"} | Placing: ${run?.placing || "—"}
Rider felt: "${run?.riderFeedback || "no feedback provided"}"
Notes: "${run?.notes || "none"}"

${historicalContext}

=== YOUR JOB ===
No video, no splits. Coach ${riderName} on ${horseName} from their own feedback and run history.
Talk like you're at the gate — plain, direct, no fluff. One sentence to acknowledge there's no video data, then get into it.

RULES:
- Connect the rider's own words to what likely happened on the pattern
- Use the history — is this better or worse than usual? Is this a pattern?
- Don't say "wide" or "narrow" to describe turns
- Don't invent faults you can't support from the feedback
- Vary your language — don't use the same phrases every time
- Return ONLY valid JSON, no markdown

Return ONLY this JSON:
{
  "summary": "2-3 sentences. Reference the time, what the rider felt, and how it compares to their history. Direct and specific.",
  "timeLost": [
    "What likely cost time based on rider feedback — connect their words to the pattern using barrel racing terminology.",
    "A second issue only if the feedback or data supports it. Must be different from the first.",
    "Third only if genuinely supported."
  ],
  "improvements": [
    "Specific drill to fix the first issue. Name it, explain how to run it.",
    "Specific drill to fix the second issue — different from improvement 1.",
    "Third fix only if there was a third issue."
  ]
}`.trim();
}

// ─── Analysis Output Sanitizer ────────────────────────────────────────────────

function sanitizeAnalysis(parsed) {
  return {
    summary: parsed.summary || "",
    timeLost: Array.isArray(parsed.timeLost) ? parsed.timeLost : [],
    improvements: Array.isArray(parsed.improvements) ? parsed.improvements : [],
    drills: Array.isArray(parsed.drills) ? parsed.drills : [],
    observations: Array.isArray(parsed.observations) ? parsed.observations : [],
    strengths: Array.isArray(parsed.strengths) ? parsed.strengths : [],
    issues: Array.isArray(parsed.issues) ? parsed.issues : [],
  };
}

// ─── Job Processors ───────────────────────────────────────────────────────────

async function processAnalysisJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) {
    console.warn("[ANALYSIS JOB] Job missing at start:", jobId);
    return;
  }

  try {
    updateJob(jobId, { status: "running", progress: 10, stage: "Building coaching report", startedAt: new Date().toISOString() });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before AI analysis.");

    updateJob(jobId, { progress: 30, stage: "Analyzing your splits and run data" });

    const hasSplits = !!latestJob.run?.manualSplits;
    const prompt = hasSplits ? buildCoachingPrompt(latestJob.run) : buildTextOnlyPrompt(latestJob.run);

    updateJob(jobId, { progress: 50, stage: "Getting coaching feedback" });

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      max_tokens: 2000,
      temperature: 0.8,
      messages: [
        {
          role: "system",
          content: "You are a real barrel racing coach talking at the gate. You never use the words 'wide', 'wider', or 'narrow'. You never use invented fault labels. You speak plainly and specifically. You return valid JSON only — no markdown, no preamble, no extra text.",
        },
        {
          role: "user",
          content: prompt,
        }
      ],
    });

    updateJob(jobId, { progress: 85, stage: "Finishing up" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      const recovered = extractLastJsonObject(outputText);
      if (recovered && recovered.summary) {
        parsedAnalysis = recovered;
        console.warn("[ANALYSIS JOB] Recovered partial JSON from AI response");
      } else {
        console.error("[ANALYSIS JOB] Invalid AI JSON:", preview(outputText, 500));
        throw new Error("AI returned invalid JSON.");
      }
    }

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      stage: "Done",
      completedAt: new Date().toISOString(),
      result: {
        success: true,
        analysis: sanitizeAnalysis(parsedAnalysis),
        python: null,
        frameCount: 0,
      },
    });

  } catch (err) {
    console.error("[ANALYSIS JOB ERROR]", err.message);
    updateJob(jobId, {
      status: "failed", progress: 100, stage: "Failed",
      completedAt: new Date().toISOString(),
      error: err.message || "Analysis failed.",
    });
  } finally {
    if (job.videoPath) safeCleanup([job.videoPath]);
  }
}

function startJobProcessing(job) {
  console.log("[JOB START]", job.id, "kind:", job.kind);
  void processAnalysisJob(job.id);
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
      model: "gpt-4o",
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

// Both video and text analysis routes now use the same fast AI-only pipeline
app.post("/analyze-run-video/start", upload.single("video"), async (req, res) => {
  try {
    const run = safeParseJson(req.body?.runData ?? req.body?.run ?? "{}");
    if (!run || typeof run !== "object") {
      if (req.file) safeCleanup([req.file.path]);
      return res.status(400).json({ error: "Run data was missing or invalid." });
    }

    const job = createJob({ kind: "analysis", run, videoPath: req.file?.path || null });
    updateJob(job.id, { progress: 5, stage: "Starting analysis" });
    startJobProcessing(job);

    console.log("[START VIDEO ANALYSIS]", job.id);
    return res.json({ ok: true, jobId: job.id });
  } catch (err) {
    console.error("[START VIDEO ERROR]", err.message);
    return res.status(500).json({ error: "Could not start analysis.", details: err.message });
  }
});

app.post("/analyze-run/start", async (req, res) => {
  try {
    const run = req.body || {};
    const job = createJob({ kind: "analysis", run });
    updateJob(job.id, { progress: 5, stage: "Starting analysis" });
    startJobProcessing(job);

    console.log("[START TEXT ANALYSIS]", job.id);
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

// ─── Generate Insights ────────────────────────────────────────────────────────

app.post("/generate-insights", async (req, res) => {
  try {
    const { horseName, insights } = req.body;
    if (!insights || !Array.isArray(insights) || insights.length === 0) {
      return res.status(400).json({ error: "No insights data provided." });
    }

    const insightDescriptions = insights.map(i => {
      return `Insight: ${i.title}\nKey: ${i.key}\nStats: ${JSON.stringify(i.stats)}`;
    }).join("\n\n---\n\n");

    const prompt = `You are a barrel racing performance analyst looking at a rider's run log data for a horse named "${horseName}". For each insight below, write 2-3 sentences that are specific to the actual numbers. Sound like a trainer who just pulled up the stats — plain language, no fluff, name actual times and gaps.

Format your response as JSON only: { "timeTrend": "...", "personalBest": "...", ... }
Use the exact "key" field from each insight as the JSON key.

RULES:
- Name specific numbers — never be vague
- If one arena condition is faster, name both and the difference
- Compare directly: "On firm ground the average is 15.2s — that's 0.4s faster than on deep ground at 15.6s"
- No training advice — data only
- No apologies for limited data — just state what's there
- 2-3 sentences max, no padding

${insightDescriptions}`;

    const response = await openai.chat.completions.create({
      model: "gpt-4o",
      messages: [{ role: "user", content: prompt }],
      temperature: 0.7,
      max_tokens: 1500,
    });

    const content = response.choices?.[0]?.message?.content || "";

    let summaries = {};
    try {
      const cleaned = content.replace(/```json\n?/g, "").replace(/```\n?/g, "").trim();
      summaries = JSON.parse(cleaned);
    } catch {
      console.error("Failed to parse AI insights response:", content);
      summaries = {};
    }

    res.json({ summaries });
  } catch (err) {
    console.error("Generate insights error:", err);
    res.status(500).json({ error: "Failed to generate insights." });
  }
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
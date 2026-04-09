import express from "express";
import multer from "multer";
import cors from "cors";
import fs from "fs";
import path from "path";
import { execFile } from "child_process";
import { promisify } from "util";
import OpenAI from "openai";
import dotenv from "dotenv";

dotenv.config();

const execFileAsync = promisify(execFile);

const app = express();
const PORT = Number(process.env.PORT || 3001);
const EXEC_MAX_BUFFER = 1024 * 1024 * 50;
const PYTHON_TIMEOUT_MS = 1000 * 60 * 8;
const JOB_TTL_MS = 1000 * 60 * 60;
const JOB_STORE_FILE = path.join(process.cwd(), "job-store.json");

const pythonExePath =
  process.env.PYTHON_PATH ||
  (process.platform === "win32" ? "python" : "python3");

const pythonScriptPath = path.join(process.cwd(), "python", "analyze_run.py");
const uploadsDir = path.join(process.cwd(), "uploads");

console.log("===== SERVER START =====");
console.log("[SERVER BOOTED]", new Date().toISOString());
console.log("Node ENV:", process.env.NODE_ENV || "development");
console.log("Port:", PORT);
console.log("Python Executable:", pythonExePath);
console.log("Python Script Path:", pythonScriptPath);
console.log("Uploads Directory:", uploadsDir);
console.log("Job Store File:", JOB_STORE_FILE);
console.log("========================");

const openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });

const jobs = new Map();
const cleanupTimers = new Map();

app.use(cors());
app.use(express.json({ limit: "2mb" }));

try {
  if (!fs.existsSync(uploadsDir)) {
    fs.mkdirSync(uploadsDir, { recursive: true });
  }
} catch (error) {
  console.error("Failed to create uploads directory:", error);
}

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
    const ext = String(path.extname(file.originalname || "") || "").toLowerCase();

    const allowedMimeTypes = ["video/mp4", "video/quicktime", "video/x-m4v", "video/mpeg", "video/webm", "video/3gpp", "application/octet-stream"];
    const allowedExtensions = [".mp4", ".mov", ".m4v", ".mpeg", ".mpg", ".webm", ".3gp"];

    if (allowedMimeTypes.includes(mime) || allowedExtensions.includes(ext)) {
      cb(null, true);
      return;
    }
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
    const candidate = raw.slice(start, end + 1);
    try { return JSON.parse(candidate); } catch { /* continue */ }
  }
  return null;
}

function previewText(text, max = 1000) {
  return String(text || "").slice(0, max);
}

function parseModelJson(outputText) {
  const cleanedOutputText = String(outputText || "")
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();
  return JSON.parse(cleanedOutputText);
}

// ─── Cleanup Utilities ────────────────────────────────────────────────────────

function safeCleanup(paths) {
  for (const targetPath of paths) {
    try {
      if (!targetPath) continue;
      if (!fs.existsSync(targetPath)) continue;
      const normalizedPath = path.resolve(targetPath);
      const normalizedUploadsDir = path.resolve(uploadsDir);
      const normalizedCwd = path.resolve(process.cwd());
      const isInsideUploads = normalizedPath.startsWith(normalizedUploadsDir);
      const isInsideProject = normalizedPath.startsWith(normalizedCwd);
      if (!isInsideProject) { console.warn("Skipping cleanup outside project directory:", normalizedPath); continue; }
      const stats = fs.statSync(normalizedPath);
      if (stats.isDirectory()) {
        fs.rmSync(normalizedPath, { recursive: true, force: true });
      } else if (isInsideUploads || normalizedPath.includes(`${path.sep}python${path.sep}`)) {
        fs.unlinkSync(normalizedPath);
      } else {
        console.warn("Skipping cleanup for unexpected file path:", normalizedPath);
      }
    } catch (cleanupError) {
      console.warn("Cleanup warning:", cleanupError.message);
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

// ─── Prompt Builders ──────────────────────────────────────────────────────────

function buildLeanCvSummary(run, pythonResult) {
  const identifiedBarrels = pythonResult?.identified_barrels || {};
  const splits = pythonResult?.splits || {};
  const insights = Array.isArray(pythonResult?.insights) ? pythonResult.insights.slice(0, 4) : [];

  const barrelLines = ["barrel1", "barrel2", "barrel3"].map((name) => {
    const barrel = identifiedBarrels?.[name];
    if (!barrel) return `${name}: not confidently identified`;
    return `${name}: center=(${barrel.center_x}, ${barrel.center_y}), detections=${barrel.detection_count}`;
  });

  return `
Computer vision data:
- Run duration: ${pythonResult?.duration_seconds ?? "unknown"} seconds
- Horse detected in ${pythonResult?.horse_detected_frames ?? "unknown"} frames
- Pattern direction: ${pythonResult?.pattern_direction ?? "unknown"}

Barrel positions detected:
- ${barrelLines.join("\n- ")}

Estimated split times:
- Start to 1st barrel: ${splits?.start_to_barrel1_seconds ?? "n/a"} sec
- 1st to 2nd barrel: ${splits?.barrel1_to_barrel2_seconds ?? "n/a"} sec
- 2nd to 3rd barrel: ${splits?.barrel2_to_barrel3_seconds ?? "n/a"} sec
- 3rd barrel to home: ${splits?.barrel3_to_home_seconds ?? "n/a"} sec

CV insights:
${insights.length ? insights.map((item) => `- ${item}`).join("\n") : "- none"}

Run details:
- Horse: ${run?.horse || "not provided"}
- Time: ${run?.time || "not provided"} seconds
- Show: ${run?.showName || "not provided"}
- Location: ${run?.location || "not provided"}
- Arena condition: ${run?.arenaCondition || "not provided"}
- Placing: ${run?.placing || "not provided"}
- Earnings: ${run?.earnings || "not provided"}
- Rider notes: ${run?.notes || "none"}
- Rider feedback: ${run?.riderFeedback || "none"}
  `.trim();
}

function buildLeanVisionPrompt(run, pythonResult) {
  const cvSummary = buildLeanCvSummary(run, pythonResult);

  return `
You are a seasoned professional barrel racer and coach with decades of competitive experience at the highest levels — NFR qualifications, world titles, and coaching champions. You speak directly, practically, and with genuine expertise. You sound like a real barrel racer, not a generic sports coach.

You are reviewing video frames and computer vision data from a barrel racing run. Give this rider the kind of specific, no-nonsense coaching feedback you would give a serious competitor.

Use proper barrel racing terminology naturally:
- Rate (slowing before a barrel), pocket (the space around a barrel), drive (forward momentum exiting)
- Shoulder control, lead changes, collection, run-down
- First barrel, second barrel, third barrel — not "barrel 1" in the coaching text
- Home (the finish line), alley (the entry run)

Coaching priorities — address in order of impact on time:
1. The most important thing this rider needs to fix RIGHT NOW
2. What they did well that they should keep doing
3. Specific issues observed at each barrel where evidence exists
4. Drills that directly address the problems you see
5. Where time is being lost on this run

Rules:
- Be direct and specific. "Your horse dropped its shoulder entering the first barrel" beats "there were issues at barrel one."
- If you can see something specific in the frames, say exactly what you see.
- If the data is limited, say so honestly but still give your best read.
- Do NOT invent details that are not supported by the data or frames.
- Keep the summary punchy — 2-3 sentences max, like you are talking to a rider at the gate.
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- "focusNext" must be one short specific coaching cue — the single most important thing to work on
- "speedInsight" — where is this run losing time? Be specific.
- "accuracyNotes" — honest note on what you could and could not see clearly
- Each item in "strengths", "issues", "workOns", "drills" must be specific and actionable — no vague generalities
- 2-3 items each — quality over quantity
- Return ONLY valid JSON. No markdown. No backticks.

${cvSummary}

Return ONLY valid JSON in this exact format:

{
  "summary": "Direct 2-3 sentence coaching read like you are talking to this rider at the gate.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "Rate earlier and give more pocket at the first barrel",
  "speedInsight": "This run is losing time in the run-down to the first barrel — the approach angle is forcing the horse to set up wide.",
  "accuracyNotes": "Split times and barrel positions are estimates from video tracking — use as coaching reference, not official data.",
  "strengths": [
    "Good forward drive coming out of the second barrel",
    "Horse showed nice collection approaching the third"
  ],
  "issues": [
    "Coming in too straight to the first barrel — not enough arc on the approach",
    "Dropping the outside shoulder through the second barrel turn"
  ],
  "workOns": [
    "Work your approach angle to the first barrel — you want to be running to a spot about 6 feet past the barrel, not straight at it",
    "Focus on keeping your horse's outside shoulder up through the turn using your outside leg and rein"
  ],
  "drills": [
    "Set a cone 6 feet past your first barrel and practice running to that cone, not the barrel itself",
    "Trot figure-8s focusing on keeping your horse's outside shoulder elevated through the turns"
  ]
}
  `.trim();
}

function buildLeanTextOnlyPrompt(run) {
  return `
You are a seasoned professional barrel racer and coach with decades of competitive experience at the highest levels — NFR qualifications, world titles, and coaching champions. You speak directly, practically, and with genuine expertise. You sound like a real barrel racer, not a generic sports coach.

No video is available for this run. You are coaching based on run data and rider notes only. Be upfront that you are working without video, but still give your best, most useful coaching read based on what the rider has shared.

Use proper barrel racing terminology naturally:
- Rate, pocket, drive, shoulder control, lead changes, collection, run-down
- First barrel, second barrel, third barrel
- Home, alley

Rules:
- Be direct and specific based on the rider's notes and run data
- If the rider mentioned something specific in their notes, address it directly by name
- Be honest that no video was available but do not dwell on it
- Keep the summary punchy — 2-3 sentences like you are talking at the gate
- "bestBarrel" and "bestTurn" must be exactly: "1st", "2nd", or "3rd"
- "focusNext" — one specific coaching cue to work on next
- "speedInsight" — where is this run likely losing time based on the data and notes?
- "accuracyNotes" — honest note about coaching without video
- Each item in "strengths", "issues", "workOns", "drills" must be specific and actionable
- 2-3 items each
- Return ONLY valid JSON. No markdown. No backticks.

Run data:
- Horse: ${run?.horse || "not provided"}
- Time: ${run?.time || "not provided"} seconds
- Show: ${run?.showName || "not provided"}
- Location: ${run?.location || "not provided"}
- Arena condition: ${run?.arenaCondition || "not provided"}
- Placing: ${run?.placing || "not provided"}
- Earnings: ${run?.earnings || "not provided"}
- Rider notes: ${run?.notes || "none"}
- Rider feedback: ${run?.riderFeedback || "none"}

Return ONLY valid JSON in this exact format:

{
  "summary": "Direct 2-3 sentence coaching read based on the run data and rider notes.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "Rate earlier and give more pocket at the first barrel",
  "speedInsight": "Based on your notes, the first barrel approach is where this run is likely losing the most time.",
  "accuracyNotes": "Coaching based on your notes and run data only — no video was available for this analysis.",
  "strengths": [
    "Solid time suggests good overall pace and horse fitness",
    "Making the check shows competitive conditioning"
  ],
  "issues": [
    "Rider noted feeling late to the first barrel — worth examining your approach angle and rate point",
    "Any shoulder drop through the second barrel will cost you time in the crossover"
  ],
  "workOns": [
    "Identify your rate point to the first barrel and be consistent hitting it every single run",
    "Film your practice runs so you can see your approach angles — what you feel and what is happening are often very different"
  ],
  "drills": [
    "Trot your pattern and focus on where you are rating — your body should be back and your horse collecting before you reach your rate point",
    "Practice your run-down to the first barrel at speed, focusing on a consistent approach angle every time"
  ]
}
  `.trim();
}

// ─── Python Result Sanitizer ──────────────────────────────────────────────────

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
    },
    pattern_direction: pythonResult.pattern_direction || null,
    normalized_actual_template_path: pythonResult.normalized_actual_template_path || [],
    ideal_template_path: pythonResult.ideal_template_path || [],
    insights: Array.isArray(pythonResult.insights) ? pythonResult.insights : [],
    highlights: pythonResult.highlights || null,
    sampled_frames: Array.isArray(pythonResult.sampled_frames)
      ? pythonResult.sampled_frames.map((frame) => ({
          percent: frame?.percent ?? null,
          frame_index: frame?.frame_index ?? null,
          timestamp_seconds: frame?.timestamp_seconds ?? null,
          read_success: !!frame?.read_success,
          horse_detection: frame?.horse_detection || null,
          barrel_detection_count: frame?.barrel_detection_count ?? 0,
          rejection_reason: frame?.rejection_reason || null,
        }))
      : [],
    frame_metrics: Array.isArray(pythonResult.frame_metrics)
      ? pythonResult.frame_metrics.map((metric) => ({
          frame_index: metric?.frame_index ?? null,
          timestamp_seconds: metric?.timestamp_seconds ?? null,
          horse_detected: !!metric?.horse_detected,
          horse_center: metric?.horse_center || null,
          nearest_barrel: metric?.nearest_barrel || null,
          nearest_barrel_distance_px: metric?.nearest_barrel_distance_px ?? null,
          dist_to_barrel1_px: metric?.dist_to_barrel1_px ?? null,
          dist_to_barrel2_px: metric?.dist_to_barrel2_px ?? null,
          dist_to_barrel3_px: metric?.dist_to_barrel3_px ?? null,
        }))
      : [],
  };
}

// ─── Job Store ────────────────────────────────────────────────────────────────

function serializeJobsForDisk() {
  return Array.from(jobs.values()).map((job) => ({ ...job }));
}

function persistJobsToDisk() {
  try {
    fs.writeFileSync(
      JOB_STORE_FILE,
      JSON.stringify({ savedAt: new Date().toISOString(), jobs: serializeJobsForDisk() }, null, 2),
      "utf8"
    );
  } catch (error) {
    console.error("Failed to persist jobs to disk:", error);
  }
}

function clearCleanupTimer(jobId) {
  const existing = cleanupTimers.get(jobId);
  if (existing) { clearTimeout(existing); cleanupTimers.delete(jobId); }
}

function deleteJob(jobId) {
  clearCleanupTimer(jobId);
  const existed = jobs.delete(jobId);
  if (existed) { console.log("[JOB DELETE]", jobId); persistJobsToDisk(); }
}

function scheduleJobCleanup(jobId) {
  clearCleanupTimer(jobId);
  const job = jobs.get(jobId);
  if (!job) return;
  const createdMs = new Date(job.createdAt).getTime();
  const expiresAtMs = createdMs + JOB_TTL_MS;
  const delay = Math.max(0, expiresAtMs - Date.now());
  const timer = setTimeout(() => { deleteJob(jobId); }, delay);
  cleanupTimers.set(jobId, timer);
}

function restoreJobsFromDisk() {
  try {
    if (!fs.existsSync(JOB_STORE_FILE)) return;
    const raw = fs.readFileSync(JOB_STORE_FILE, "utf8");
    const parsed = safeParseJson(raw);
    if (!parsed || !Array.isArray(parsed.jobs)) { console.warn("Job store file exists but could not be parsed."); return; }

    let restoredCount = 0;
    for (const job of parsed.jobs) {
      if (!job?.id || !job?.createdAt) continue;
      const ageMs = Date.now() - new Date(job.createdAt).getTime();
      if (ageMs >= JOB_TTL_MS) continue;
      const restoredJob = { ...job };
      if (restoredJob.status === "queued" || restoredJob.status === "running") {
        restoredJob.status = "failed";
        restoredJob.stage = "Failed";
        restoredJob.completedAt = new Date().toISOString();
        restoredJob.error = "Server restarted while analysis was running. Please re-run the analysis.";
      }
      jobs.set(restoredJob.id, restoredJob);
      scheduleJobCleanup(restoredJob.id);
      restoredCount += 1;
    }

    console.log("[JOB RESTORE] restored jobs:", restoredCount);
    persistJobsToDisk();
  } catch (error) {
    console.error("Failed to restore jobs from disk:", error);
  }
}

function createJob({ kind, run, videoPath = null }) {
  const jobId = `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
  const job = {
    id: jobId, kind, run, videoPath,
    status: "queued", progress: 0, stage: "Queued",
    createdAt: new Date().toISOString(),
    startedAt: null, completedAt: null, error: null, result: null,
  };
  jobs.set(jobId, job);
  scheduleJobCleanup(jobId);
  persistJobsToDisk();
  console.log("[JOB CREATED]", jobId, "kind:", kind);
  console.log("[JOB STORED]", jobId, "exists:", jobs.has(jobId), "jobCount:", jobs.size);
  return job;
}

function updateJob(jobId, updates) {
  const job = jobs.get(jobId);
  if (!job) { console.warn("[JOB UPDATE MISSED]", jobId, "updates:", updates); return null; }
  Object.assign(job, updates);
  persistJobsToDisk();
  console.log("[JOB UPDATE]", jobId, "status:", job.status, "progress:", job.progress, "stage:", job.stage);
  return job;
}

// ─── Frame Selection ──────────────────────────────────────────────────────────

function selectStrategicFramePaths(sampledFrames, maxFrames = 4) {
  const usable = (sampledFrames || [])
    .filter((frame) => frame?.read_success && (frame?.overlay_image_path || frame?.image_path))
    .map((frame) => frame.overlay_image_path || frame.image_path)
    .filter(Boolean);

  if (usable.length <= maxFrames) return usable;

  const selected = [];
  for (let i = 0; i < maxFrames; i += 1) {
    const index = Math.round((i * (usable.length - 1)) / (maxFrames - 1));
    selected.push(usable[index]);
  }
  return [...new Set(selected)];
}

function buildImageInputs(framePaths) {
  return framePaths.map((framePath) => {
    const base64 = fs.readFileSync(framePath).toString("base64");
    return {
      type: "image_url",
      image_url: { url: `data:image/jpeg;base64,${base64}`, detail: "low" },
    };
  });
}

// ─── Python Runner ────────────────────────────────────────────────────────────

async function runPythonAnalysis(videoPath) {
  console.log("Running Python analysis...", videoPath);

  try {
    const { stdout, stderr } = await execFileAsync(
      pythonExePath,
      [pythonScriptPath, videoPath],
      {
        maxBuffer: EXEC_MAX_BUFFER,
        timeout: PYTHON_TIMEOUT_MS,
        env: { ...process.env, PYTHONUNBUFFERED: "1", YOLO_CONFIG_DIR: "/tmp/Ultralytics" },
      }
    );

    if (stderr && String(stderr).trim()) console.warn("Python stderr preview:", previewText(stderr));
    console.log("Python stdout length:", String(stdout || "").length);

    const pythonResult = safeParseJson(stdout) || extractLastJsonObject(stdout);

    if (!pythonResult) { console.error("Invalid Python output preview:", previewText(stdout)); throw new Error("Python returned invalid JSON."); }
    if (!pythonResult.ok) throw new Error(pythonResult.error || "Python analysis failed.");

    return pythonResult;
  } catch (error) {
    if (error?.stdout) console.error("Python stdout on failure:", previewText(error.stdout));
    if (error?.stderr) console.error("Python stderr on failure:", previewText(error.stderr));

    const recovered = safeParseJson(error?.stdout) || extractLastJsonObject(error?.stdout);
    if (recovered) {
      if (!recovered.ok) throw new Error(recovered.error || "Python analysis failed.");
      return recovered;
    }

    if (error?.killed || error?.signal === "SIGTERM") throw new Error("Python analysis timed out on the server.");
    if (error?.code === "ETIMEDOUT") throw new Error("Python analysis timed out on the server.");
    throw error;
  }
}

// ─── Job Processors ───────────────────────────────────────────────────────────

async function processVideoJob(jobId) {
  const initialJob = jobs.get(jobId);
  if (!initialJob) { console.warn("[VIDEO JOB START FAILED] job missing:", jobId); return; }

  const videoPath = initialJob.videoPath;
  let pythonGeneratedPaths = [];

  try {
    updateJob(jobId, { status: "running", progress: 10, stage: "Starting computer vision", startedAt: new Date().toISOString() });

    const currentJob = jobs.get(jobId);
    if (!currentJob) throw new Error("Job disappeared before Python analysis started.");

    const pythonResult = await runPythonAnalysis(videoPath);
    updateJob(jobId, { progress: 55, stage: "Computer vision finished" });

    pythonGeneratedPaths = getPythonGeneratedPaths(pythonResult);
    updateJob(jobId, { progress: 65, stage: "Selecting key frames" });

    const framePaths = selectStrategicFramePaths(pythonResult.sampled_frames || [], 4);
    if (!framePaths.length) throw new Error("Python did not return any usable frame images.");

    updateJob(jobId, { progress: 72, stage: "Preparing images for AI" });
    const imageInputs = buildImageInputs(framePaths);
    updateJob(jobId, { progress: 80, stage: "Requesting AI coaching analysis" });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before OpenAI analysis started.");

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      max_tokens: 1200,
      messages: [
        {
          role: "user",
          content: [
            { type: "text", text: buildLeanVisionPrompt(latestJob.run, pythonResult) },
            ...imageInputs,
          ],
        },
      ],
    });

    updateJob(jobId, { progress: 95, stage: "Finalizing analysis" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error("Invalid OpenAI JSON output preview:", String(outputText || "").slice(0, 1000));
      throw new Error("OpenAI returned invalid JSON.");
    }

    const safeAnalysis = {
      summary: parsedAnalysis.summary || "",
      bestBarrel: parsedAnalysis.bestBarrel || null,
      bestTurn: parsedAnalysis.bestTurn || null,
      focusNext: parsedAnalysis.focusNext || null,
      speedInsight: parsedAnalysis.speedInsight || null,
      accuracyNotes: parsedAnalysis.accuracyNotes || null,
      strengths: Array.isArray(parsedAnalysis.strengths) ? parsedAnalysis.strengths : [],
      issues: Array.isArray(parsedAnalysis.issues) ? parsedAnalysis.issues : [],
      workOns: Array.isArray(parsedAnalysis.workOns) ? parsedAnalysis.workOns : [],
      drills: Array.isArray(parsedAnalysis.drills) ? parsedAnalysis.drills : [],
    };

    updateJob(jobId, {
      status: "completed", progress: 100, stage: "Completed",
      completedAt: new Date().toISOString(),
      result: { success: true, analysis: safeAnalysis, python: sanitizePythonForClient(pythonResult), frameCount: framePaths.length },
    });
  } catch (error) {
    console.error("VIDEO JOB ERROR:", error);
    updateJob(jobId, { status: "failed", progress: 100, stage: "Failed", completedAt: new Date().toISOString(), error: error.message || "Video analysis failed." });
  } finally {
    safeCleanup([videoPath, ...pythonGeneratedPaths]);
  }
}

async function processTextJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) { console.warn("[TEXT JOB START FAILED] job missing:", jobId); return; }

  try {
    updateJob(jobId, { status: "running", progress: 15, stage: "Preparing metadata-only analysis", startedAt: new Date().toISOString() });
    updateJob(jobId, { progress: 75, stage: "Requesting AI coaching analysis" });

    const latestJob = jobs.get(jobId);
    if (!latestJob) throw new Error("Job disappeared before text analysis started.");

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      max_tokens: 1200,
      messages: [{ role: "user", content: buildLeanTextOnlyPrompt(latestJob.run) }],
    });

    updateJob(jobId, { progress: 95, stage: "Finalizing analysis" });

    const outputText = response.choices?.[0]?.message?.content || "";
    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error("Invalid OpenAI JSON output preview:", String(outputText || "").slice(0, 1000));
      throw new Error("OpenAI returned invalid JSON.");
    }

    const safeAnalysis = {
      summary: parsedAnalysis.summary || "",
      bestBarrel: parsedAnalysis.bestBarrel || null,
      bestTurn: parsedAnalysis.bestTurn || null,
      focusNext: parsedAnalysis.focusNext || null,
      speedInsight: parsedAnalysis.speedInsight || null,
      accuracyNotes: parsedAnalysis.accuracyNotes || null,
      strengths: Array.isArray(parsedAnalysis.strengths) ? parsedAnalysis.strengths : [],
      issues: Array.isArray(parsedAnalysis.issues) ? parsedAnalysis.issues : [],
      workOns: Array.isArray(parsedAnalysis.workOns) ? parsedAnalysis.workOns : [],
      drills: Array.isArray(parsedAnalysis.drills) ? parsedAnalysis.drills : [],
    };

    updateJob(jobId, {
      status: "completed", progress: 100, stage: "Completed",
      completedAt: new Date().toISOString(),
      result: { success: true, analysis: safeAnalysis, python: null, frameCount: 0 },
    });
  } catch (error) {
    console.error("TEXT JOB ERROR:", error);
    updateJob(jobId, { status: "failed", progress: 100, stage: "Failed", completedAt: new Date().toISOString(), error: error.message || "Run analysis failed." });
  }
}

function startJobProcessing(job) {
  console.log("[JOB PROCESS START]", job.id, "kind:", job.kind);
  if (job.kind === "video") { void processVideoJob(job.id); return; }
  if (job.kind === "text") { void processTextJob(job.id); }
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

restoreJobsFromDisk();

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/", (_req, res) => res.json({ ok: true, message: "AI Coaching Server running", activeJobs: jobs.size }));
app.get("/health", (_req, res) => res.json({ ok: true, message: "AI Coaching Server running", activeJobs: jobs.size }));

app.get("/debug/jobs", (_req, res) => {
  res.json({
    ok: true, count: jobs.size,
    jobs: Array.from(jobs.values()).map((job) => ({
      id: job.id, kind: job.kind, status: job.status, progress: job.progress,
      stage: job.stage, createdAt: job.createdAt, startedAt: job.startedAt,
      completedAt: job.completedAt, error: job.error, hasResult: !!job.result,
    })),
  });
});

app.post("/analyze-run-video/start", upload.single("video"), async (req, res) => {
  try {
    if (!req.file) return res.status(400).json({ error: "No video file was uploaded." });

    const runDataRaw = req.body?.runData ?? req.body?.run ?? "{}";
    const run = safeParseJson(runDataRaw);

    if (!run || typeof run !== "object") {
      safeCleanup([req.file.path]);
      return res.status(400).json({ error: "Run data was missing or invalid JSON." });
    }

    const job = createJob({ kind: "video", run, videoPath: req.file.path });
    updateJob(job.id, { progress: 5, stage: "Upload received" });
    startJobProcessing(job);
    console.log("[JOB START RESPONSE]", { jobId: job.id });
    return res.json({ ok: true, jobId: job.id });
  } catch (error) {
    console.error("START VIDEO JOB ERROR:", error);
    return res.status(500).json({ error: "Could not start video analysis.", details: error.message });
  }
});

app.post("/analyze-run/start", async (req, res) => {
  try {
    const run = req.body || {};
    const job = createJob({ kind: "text", run });
    updateJob(job.id, { progress: 5, stage: "Analysis request received" });
    startJobProcessing(job);
    console.log("[JOB START RESPONSE]", { jobId: job.id });
    return res.json({ ok: true, jobId: job.id });
  } catch (error) {
    console.error("START TEXT JOB ERROR:", error);
    return res.status(500).json({ error: "Could not start run analysis.", details: error.message });
  }
});

app.get("/analysis-status/:jobId", (req, res) => {
  const requestedJobId = String(req.params.jobId || "").trim();
  console.log("[JOB POLL]", requestedJobId, "exists:", jobs.has(requestedJobId), "jobCount:", jobs.size);

  const job = jobs.get(requestedJobId);
  if (!job) {
    return res.status(404).json({ ok: false, error: "Analysis job not found.", requestedJobId, activeJobCount: jobs.size });
  }

  return res.json({
    ok: true, jobId: job.id, status: job.status, progress: job.progress, stage: job.stage,
    createdAt: job.createdAt, startedAt: job.startedAt, completedAt: job.completedAt,
    error: job.error, result: job.status === "completed" ? job.result : null,
  });
});

// ─── Error Handler ────────────────────────────────────────────────────────────

app.use((error, _req, res, _next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === "LIMIT_FILE_SIZE") return res.status(400).json({ error: "Uploaded video is too large. Max size is 250 MB." });
    return res.status(400).json({ error: error.message || "Upload failed." });
  }
  if (error) return res.status(400).json({ error: error.message || "Request failed." });
  return res.status(500).json({ error: "Unknown server error." });
});

// ─── Start Server ─────────────────────────────────────────────────────────────

app.listen(PORT, "0.0.0.0", () => {
  console.log(`AI Coaching Server running on port ${PORT}`);
});
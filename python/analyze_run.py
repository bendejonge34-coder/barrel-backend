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
const EXEC_MAX_BUFFER = 1024 * 1024 * 50; // 50 MB
const PYTHON_TIMEOUT_MS = 1000 * 60 * 8; // 8 minutes
const JOB_TTL_MS = 1000 * 60 * 60; // 1 hour
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

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

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
  limits: {
    fileSize: 250 * 1024 * 1024,
  },
  fileFilter: (_req, file, cb) => {
    const mime = String(file.mimetype || "").toLowerCase();
    const ext = String(path.extname(file.originalname || "") || "").toLowerCase();

    const allowedMimeTypes = [
      "video/mp4",
      "video/quicktime",
      "video/x-m4v",
      "video/mpeg",
      "video/webm",
      "video/3gpp",
      "application/octet-stream",
    ];

    const allowedExtensions = [
      ".mp4",
      ".mov",
      ".m4v",
      ".mpeg",
      ".mpg",
      ".webm",
      ".3gp",
    ];

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
    if (typeof input === "object" && input !== null) {
      return input;
    }
    if (typeof input !== "string" || !input.trim()) {
      return fallback;
    }
    return JSON.parse(input);
  } catch {
    return fallback;
  }
}

function extractLastJsonObject(text) {
  const raw = String(text || "").trim();
  if (!raw) return null;

  try {
    return JSON.parse(raw);
  } catch {
    // continue
  }

  const end = raw.lastIndexOf("}");
  if (end === -1) return null;

  for (
    let start = raw.lastIndexOf("{", end);
    start !== -1;
    start = raw.lastIndexOf("{", start - 1)
  ) {
    const candidate = raw.slice(start, end + 1);
    try {
      return JSON.parse(candidate);
    } catch {
      // continue
    }
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

      if (!isInsideProject) {
        console.warn("Skipping cleanup outside project directory:", normalizedPath);
        continue;
      }

      const stats = fs.statSync(normalizedPath);

      if (stats.isDirectory()) {
        fs.rmSync(normalizedPath, { recursive: true, force: true });
      } else if (
        isInsideUploads ||
        normalizedPath.includes(`${path.sep}python${path.sep}`)
      ) {
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

  if (!pythonResult || typeof pythonResult !== "object") {
    return paths;
  }

  if (pythonResult.path_map_path) {
    paths.push(pythonResult.path_map_path);
  }

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
  const insights = Array.isArray(pythonResult?.insights)
    ? pythonResult.insights.slice(0, 4)
    : [];

  const barrelLines = ["barrel1", "barrel2", "barrel3"].map((name) => {
    const barrel = identifiedBarrels?.[name];
    if (!barrel) return `${name}: not confidently identified`;
    return `${name}: center=(${barrel.center_x}, ${barrel.center_y}), detections=${barrel.detection_count}`;
  });

  return `
Computer vision summary:
- Duration: ${pythonResult?.duration_seconds ?? "unknown"} seconds
- FPS: ${pythonResult?.fps ?? "unknown"}
- Resolution: ${pythonResult?.width ?? "unknown"} x ${pythonResult?.height ?? "unknown"}
- Horse detected frames: ${pythonResult?.horse_detected_frames ?? "unknown"}
- Raw trajectory points: ${pythonResult?.raw_trajectory_point_count ?? 0}
- Accepted trajectory points: ${pythonResult?.accepted_trajectory_point_count ?? 0}
- Smoothed trajectory points: ${pythonResult?.smoothed_trajectory_point_count ?? 0}

Barrels:
- ${barrelLines.join("\n- ")}

Estimated split timing:
- start to barrel 1: ${splits?.start_to_barrel1_seconds ?? "n/a"}
- barrel 1 to barrel 2: ${splits?.barrel1_to_barrel2_seconds ?? "n/a"}
- barrel 2 to barrel 3: ${splits?.barrel2_to_barrel3_seconds ?? "n/a"}
- barrel 3 to home: ${splits?.barrel3_to_home_seconds ?? "n/a"}

Key CV insights:
${insights.length ? insights.map((item) => `- ${item}`).join("\n") : "- none"}

Run metadata:
- Horse: ${run?.horse || ""}
- Time: ${run?.time || ""}
- Show Name: ${run?.showName || ""}
- Location: ${run?.location || ""}
- Arena Condition: ${run?.arenaCondition || ""}
- Placing: ${run?.placing || ""}
- Earnings: ${run?.earnings || ""}
- Notes: ${run?.notes || ""}
- Rider Feedback: ${run?.riderFeedback || ""}

Important limitations:
- This is image-space analysis, not true calibrated arena measurement.
- Be conservative. If something is unclear from the images, say so.
  `.trim();
}

function buildLeanVisionPrompt(run, pythonResult) {
  const cvSummary = buildLeanCvSummary(run, pythonResult);

  return `
You are an experienced barrel racing coach.

Your job is to give a FAST, practical, believable coaching read on this run using:
- run metadata
- computer vision summary
- a small set of representative frame images

Focus only on the highest-value coaching points.

Rules:
- Be honest and conservative.
- Do not invent detail that is not visible or supported.
- Keep the response practical and short.
- "bestBarrel" and "bestTurn" must be only: "1st", "2nd", or "3rd".
- "focusNext" must be a short coaching priority phrase.
- "speedInsight" should be 1 short sentence.
- "accuracyNotes" should be 1 short sentence about confidence/limitations.
- Each item in "strengths", "issues", "workOns", and "drills" must be a short plain string.
- "strengths" should have 2-3 items of what looked good in the run.
- "issues" should have 2-3 items of the most likely problems observed.
- "workOns" should have 2-3 short actionable things to practice.
- "drills" should have 2-3 specific drill suggestions.
- Return ONLY valid JSON. No markdown. No backticks.

${cvSummary}

Return ONLY valid JSON in this exact format:

{
  "summary": "2-4 sentence practical coaching summary.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "Cleaner entry to 1st barrel",
  "speedInsight": "The run carried decent speed between the 2nd and 3rd barrels but gave away momentum leaving the 1st.",
  "accuracyNotes": "Barrel identity and turn detail are moderately confident but limited by image-space analysis.",
  "strengths": [
    "Good rate into 2nd barrel",
    "Strong exit drive after 3rd barrel"
  ],
  "issues": [
    "Wide entry to 1st barrel",
    "Lost momentum coming out of the turn"
  ],
  "workOns": [
    "Tighten the approach line to 1st barrel",
    "Maintain pace and forward drive through each turn"
  ],
  "drills": [
    "Trot the pattern slowly focusing on entry angles at each barrel",
    "Practice rate and release at 1st barrel using a cone marker"
  ]
}
  `.trim();
}

function buildLeanTextOnlyPrompt(run) {
  return `
You are an experienced barrel racing coach.

No video was provided. Use only the metadata and rider notes.

Be practical and conservative.

Rules:
- Be honest that no video was available.
- Keep the response short and useful.
- "bestBarrel" and "bestTurn" must be only: "1st", "2nd", or "3rd".
- "focusNext" must be a short coaching priority phrase.
- "speedInsight" should be 1 short sentence.
- "accuracyNotes" should be 1 short sentence about confidence/limitations.
- Each item in "strengths", "issues", "workOns", and "drills" must be a short plain string.
- "strengths" should have 2-3 items of what likely went well based on the notes.
- "issues" should have 2-3 items of likely problems based on the notes.
- "workOns" should have 2-3 short actionable things to practice.
- "drills" should have 2-3 specific drill suggestions.
- Return ONLY valid JSON. No markdown. No backticks.

Run data:
- Horse: ${run?.horse || ""}
- Time: ${run?.time || ""}
- Show Name: ${run?.showName || ""}
- Location: ${run?.location || ""}
- Arena Condition: ${run?.arenaCondition || ""}
- Placing: ${run?.placing || ""}
- Earnings: ${run?.earnings || ""}
- Notes: ${run?.notes || ""}
- Rider Feedback: ${run?.riderFeedback || ""}

Return ONLY valid JSON in this exact format:

{
  "summary": "2-4 sentence practical coaching summary.",
  "bestBarrel": "1st",
  "bestTurn": "2nd",
  "focusNext": "Cleaner entry to 1st barrel",
  "speedInsight": "Based on the notes, the run may have lost momentum through the first turn.",
  "accuracyNotes": "Confidence is limited because no video was provided.",
  "strengths": [
    "Consistent pace reported through the middle of the run",
    "Good placing suggests overall solid effort"
  ],
  "issues": [
    "Rider notes suggest issues at the first barrel",
    "Momentum loss likely on one or more turns"
  ],
  "workOns": [
    "Focus on entry angles at the first barrel",
    "Work on maintaining forward drive through turns"
  ],
  "drills": [
    "Trot the pattern to reinforce correct entry lines",
    "Practice the first barrel approach at speed with a ground pole guide"
  ]
}
  `.trim();
}

// ─── Python Result Sanitizer ──────────────────────────────────────────────────

function sanitizePythonForClient(pythonResult) {
  if (!pythonResult || typeof pythonResult !== "object") {
    return null;
  }

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
    accepted_trajectory_point_count:
      pythonResult.accepted_trajectory_point_count ?? 0,
    smoothed_trajectory_point_count:
      pythonResult.smoothed_trajectory_point_count ?? 0,

    tracking_quality: pythonResult.tracking_quality || null,
    barrel_detection_summary: pythonResult.barrel_detection_summary || null,
    identified_barrels: pythonResult.identified_barrels || {
      barrel1: null,
      barrel2: null,
      barrel3: null,
    },
    turns: pythonResult.turns || {
      barrel1: null,
      barrel2: null,
      barrel3: null,
    },
    splits: pythonResult.splits || {
      start_to_barrel1_seconds: null,
      barrel1_to_barrel2_seconds: null,
      barrel2_to_barrel3_seconds: null,
      barrel3_to_home_seconds: null,
    },
    pattern_direction: pythonResult.pattern_direction || null,
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
          nearest_barrel_distance_px:
            metric?.nearest_barrel_distance_px ?? null,
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
      JSON.stringify(
        {
          savedAt: new Date().toISOString(),
          jobs: serializeJobsForDisk(),
        },
        null,
        2
      ),
      "utf8"
    );
  } catch (error) {
    console.error("Failed to persist jobs to disk:", error);
  }
}

function clearCleanupTimer(jobId) {
  const existing = cleanupTimers.get(jobId);
  if (existing) {
    clearTimeout(existing);
    cleanupTimers.delete(jobId);
  }
}

function deleteJob(jobId) {
  clearCleanupTimer(jobId);
  const existed = jobs.delete(jobId);
  if (existed) {
    console.log("[JOB DELETE]", jobId);
    persistJobsToDisk();
  }
}

function scheduleJobCleanup(jobId) {
  clearCleanupTimer(jobId);

  const job = jobs.get(jobId);
  if (!job) return;

  const createdMs = new Date(job.createdAt).getTime();
  const expiresAtMs = createdMs + JOB_TTL_MS;
  const delay = Math.max(0, expiresAtMs - Date.now());

  const timer = setTimeout(() => {
    deleteJob(jobId);
  }, delay);

  cleanupTimers.set(jobId, timer);
}

function restoreJobsFromDisk() {
  try {
    if (!fs.existsSync(JOB_STORE_FILE)) {
      return;
    }

    const raw = fs.readFileSync(JOB_STORE_FILE, "utf8");
    const parsed = safeParseJson(raw);

    if (!parsed || !Array.isArray(parsed.jobs)) {
      console.warn("Job store file exists but could not be parsed.");
      return;
    }

    let restoredCount = 0;

    for (const job of parsed.jobs) {
      if (!job?.id || !job?.createdAt) {
        continue;
      }

      const ageMs = Date.now() - new Date(job.createdAt).getTime();
      if (ageMs >= JOB_TTL_MS) {
        continue;
      }

      const restoredJob = { ...job };

      if (restoredJob.status === "queued" || restoredJob.status === "running") {
        restoredJob.status = "failed";
        restoredJob.stage = "Failed";
        restoredJob.completedAt = new Date().toISOString();
        restoredJob.error =
          "Server restarted while analysis was running. Please re-run the analysis.";
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
    id: jobId,
    kind,
    run,
    videoPath,
    status: "queued",
    progress: 0,
    stage: "Queued",
    createdAt: new Date().toISOString(),
    startedAt: null,
    completedAt: null,
    error: null,
    result: null,
  };

  jobs.set(jobId, job);
  scheduleJobCleanup(jobId);
  persistJobsToDisk();

  console.log("[JOB CREATED]", jobId, "kind:", kind);
  console.log(
    "[JOB STORED]",
    jobId,
    "exists:",
    jobs.has(jobId),
    "jobCount:",
    jobs.size
  );

  return job;
}

function updateJob(jobId, updates) {
  const job = jobs.get(jobId);
  if (!job) {
    console.warn("[JOB UPDATE MISSED]", jobId, "updates:", updates);
    return null;
  }

  Object.assign(job, updates);
  persistJobsToDisk();

  console.log(
    "[JOB UPDATE]",
    jobId,
    "status:",
    job.status,
    "progress:",
    job.progress,
    "stage:",
    job.stage
  );

  return job;
}

// ─── Frame Selection ──────────────────────────────────────────────────────────

function selectStrategicFramePaths(sampledFrames, maxFrames = 4) {
  const usable = (sampledFrames || [])
    .filter(
      (frame) =>
        frame?.read_success &&
        (frame?.overlay_image_path || frame?.image_path)
    )
    .map((frame) => frame.overlay_image_path || frame.image_path)
    .filter(Boolean);

  if (usable.length <= maxFrames) {
    return usable;
  }

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
      image_url: {
        url: `data:image/jpeg;base64,${base64}`,
        detail: "low",
      },
    };
  });
}

// ─── Python Runner ────────────────────────────────────────────────────────────

async function runPythonAnalysis(videoPath, runTimeSeconds = null) {
  console.log("Running Python analysis...");
  console.log("Video Path:", videoPath);
  console.log("Run Time (seconds):", runTimeSeconds);

  try {
    const args = [pythonScriptPath, videoPath];

    if (runTimeSeconds !== null && Number.isFinite(runTimeSeconds)) {
      args.push(String(runTimeSeconds));
    }

    const { stdout, stderr } = await execFileAsync(
      pythonExePath,
      args,
      {
        maxBuffer: EXEC_MAX_BUFFER,
        timeout: PYTHON_TIMEOUT_MS,
        env: {
          ...process.env,
          PYTHONUNBUFFERED: "1",
          YOLO_CONFIG_DIR: "/tmp/Ultralytics",
        },
      }
    );

    if (stderr && String(stderr).trim()) {
      console.warn("Python stderr preview:", previewText(stderr));
    }

    console.log("Python stdout length:", String(stdout || "").length);

    const pythonResult =
      safeParseJson(stdout) || extractLastJsonObject(stdout);

    if (!pythonResult) {
      console.error("Invalid Python output preview:", previewText(stdout));
      throw new Error("Python returned invalid JSON.");
    }

    if (!pythonResult.ok) {
      throw new Error(pythonResult.error || "Python analysis failed.");
    }

    return pythonResult;
  } catch (error) {
    if (error?.stdout) {
      console.error("Python stdout on failure:", previewText(error.stdout));
    }

    if (error?.stderr) {
      console.error("Python stderr on failure:", previewText(error.stderr));
    }

    const recovered =
      safeParseJson(error?.stdout) || extractLastJsonObject(error?.stdout);

    if (recovered) {
      if (!recovered.ok) {
        throw new Error(recovered.error || "Python analysis failed.");
      }
      return recovered;
    }

    if (error?.killed || error?.signal === "SIGTERM") {
      throw new Error("Python analysis timed out on the server.");
    }

    if (error?.code === "ETIMEDOUT") {
      throw new Error("Python analysis timed out on the server.");
    }

    throw error;
  }
}

// ─── Job Processors ───────────────────────────────────────────────────────────

async function processVideoJob(jobId) {
  const initialJob = jobs.get(jobId);
  if (!initialJob) {
    console.warn("[VIDEO JOB START FAILED] job missing:", jobId);
    return;
  }

  const videoPath = initialJob.videoPath;
  let pythonGeneratedPaths = [];

  try {
    updateJob(jobId, {
      status: "running",
      progress: 10,
      stage: "Starting computer vision",
      startedAt: new Date().toISOString(),
    });

    const currentJob = jobs.get(jobId);
    if (!currentJob) {
      throw new Error("Job disappeared before Python analysis started.");
    }

    const parsedRunTime = parseFloat(initialJob.run?.time || "");
    const runTimeSeconds = Number.isFinite(parsedRunTime)
      ? parsedRunTime
      : null;

    const pythonResult = await runPythonAnalysis(videoPath, runTimeSeconds);

    updateJob(jobId, {
      progress: 55,
      stage: "Computer vision finished",
    });

    pythonGeneratedPaths = getPythonGeneratedPaths(pythonResult);

    updateJob(jobId, {
      progress: 65,
      stage: "Selecting key frames",
    });

    const framePaths = selectStrategicFramePaths(
      pythonResult.sampled_frames || [],
      4
    );

    if (!framePaths.length) {
      throw new Error("Python did not return any usable frame images.");
    }

    updateJob(jobId, {
      progress: 72,
      stage: "Preparing images for AI",
    });

    const imageInputs = buildImageInputs(framePaths);

    updateJob(jobId, {
      progress: 80,
      stage: "Requesting AI coaching analysis",
    });

    const latestJob = jobs.get(jobId);
    if (!latestJob) {
      throw new Error("Job disappeared before OpenAI analysis started.");
    }

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      max_tokens: 1000,
      messages: [
        {
          role: "user",
          content: [
            {
              type: "text",
              text: buildLeanVisionPrompt(latestJob.run, pythonResult),
            },
            ...imageInputs,
          ],
        },
      ],
    });

    updateJob(jobId, {
      progress: 95,
      stage: "Finalizing analysis",
    });

    const outputText = response.choices?.[0]?.message?.content || "";

    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error(
        "Invalid OpenAI JSON output preview:",
        String(outputText || "").slice(0, 1000)
      );
      throw new Error("OpenAI returned invalid JSON.");
    }

    const safeAnalysis = {
      summary: parsedAnalysis.summary || "",
      bestBarrel: parsedAnalysis.bestBarrel || null,
      bestTurn: parsedAnalysis.bestTurn || null,
      focusNext: parsedAnalysis.focusNext || null,
      speedInsight: parsedAnalysis.speedInsight || null,
      accuracyNotes: parsedAnalysis.accuracyNotes || null,
      strengths: Array.isArray(parsedAnalysis.strengths)
        ? parsedAnalysis.strengths
        : [],
      issues: Array.isArray(parsedAnalysis.issues)
        ? parsedAnalysis.issues
        : [],
      workOns: Array.isArray(parsedAnalysis.workOns)
        ? parsedAnalysis.workOns
        : [],
      drills: Array.isArray(parsedAnalysis.drills)
        ? parsedAnalysis.drills
        : [],
    };

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      stage: "Completed",
      completedAt: new Date().toISOString(),
      result: {
        success: true,
        analysis: safeAnalysis,
        python: sanitizePythonForClient(pythonResult),
        frameCount: framePaths.length,
      },
    });
  } catch (error) {
    console.error("VIDEO JOB ERROR:", error);

    updateJob(jobId, {
      status: "failed",
      progress: 100,
      stage: "Failed",
      completedAt: new Date().toISOString(),
      error: error.message || "Video analysis failed.",
    });
  } finally {
    safeCleanup([videoPath, ...pythonGeneratedPaths]);
  }
}

async function processTextJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) {
    console.warn("[TEXT JOB START FAILED] job missing:", jobId);
    return;
  }

  try {
    updateJob(jobId, {
      status: "running",
      progress: 15,
      stage: "Preparing metadata-only analysis",
      startedAt: new Date().toISOString(),
    });

    updateJob(jobId, {
      progress: 75,
      stage: "Requesting AI coaching analysis",
    });

    const latestJob = jobs.get(jobId);
    if (!latestJob) {
      throw new Error("Job disappeared before text analysis started.");
    }

    const response = await openai.chat.completions.create({
      model: "gpt-4o-mini",
      max_tokens: 1000,
      messages: [
        {
          role: "user",
          content: buildLeanTextOnlyPrompt(latestJob.run),
        },
      ],
    });

    updateJob(jobId, {
      progress: 95,
      stage: "Finalizing analysis",
    });

    const outputText = response.choices?.[0]?.message?.content || "";

    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error(
        "Invalid OpenAI JSON output preview:",
        String(outputText || "").slice(0, 1000)
      );
      throw new Error("OpenAI returned invalid JSON.");
    }

    const safeAnalysis = {
      summary: parsedAnalysis.summary || "",
      bestBarrel: parsedAnalysis.bestBarrel || null,
      bestTurn: parsedAnalysis.bestTurn || null,
      focusNext: parsedAnalysis.focusNext || null,
      speedInsight: parsedAnalysis.speedInsight || null,
      accuracyNotes: parsedAnalysis.accuracyNotes || null,
      strengths: Array.isArray(parsedAnalysis.strengths)
        ? parsedAnalysis.strengths
        : [],
      issues: Array.isArray(parsedAnalysis.issues)
        ? parsedAnalysis.issues
        : [],
      workOns: Array.isArray(parsedAnalysis.workOns)
        ? parsedAnalysis.workOns
        : [],
      drills: Array.isArray(parsedAnalysis.drills)
        ? parsedAnalysis.drills
        : [],
    };

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      stage: "Completed",
      completedAt: new Date().toISOString(),
      result: {
        success: true,
        analysis: safeAnalysis,
        python: null,
        frameCount: 0,
      },
    });
  } catch (error) {
    console.error("TEXT JOB ERROR:", error);

    updateJob(jobId, {
      status: "failed",
      progress: 100,
      stage: "Failed",
      completedAt: new Date().toISOString(),
      error: error.message || "Run analysis failed.",
    });
  }
}

function startJobProcessing(job) {
  console.log("[JOB PROCESS START]", job.id, "kind:", job.kind);

  if (job.kind === "video") {
    void processVideoJob(job.id);
    return;
  }

  if (job.kind === "text") {
    void processTextJob(job.id);
  }
}

// ─── Boot ─────────────────────────────────────────────────────────────────────

restoreJobsFromDisk();

// ─── Routes ───────────────────────────────────────────────────────────────────

app.get("/", (_req, res) => {
  res.json({
    ok: true,
    message: "AI Coaching Server running",
    activeJobs: jobs.size,
  });
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    message: "AI Coaching Server running",
    activeJobs: jobs.size,
  });
});

app.get("/debug/jobs", (_req, res) => {
  res.json({
    ok: true,
    count: jobs.size,
    jobs: Array.from(jobs.values()).map((job) => ({
      id: job.id,
      kind: job.kind,
      status: job.status,
      progress: job.progress,
      stage: job.stage,
      createdAt: job.createdAt,
      startedAt: job.startedAt,
      completedAt: job.completedAt,
      error: job.error,
      hasResult: !!job.result,
    })),
  });
});

app.post(
  "/analyze-run-video/start",
  upload.single("video"),
  async (req, res) => {
    try {
      if (!req.file) {
        return res.status(400).json({
          error: "No video file was uploaded.",
        });
      }

      const runDataRaw = req.body?.runData ?? req.body?.run ?? "{}";
      const run = safeParseJson(runDataRaw);

      if (!run || typeof run !== "object") {
        safeCleanup([req.file.path]);
        return res.status(400).json({
          error: "Run data was missing or invalid JSON.",
        });
      }

      const job = createJob({
        kind: "video",
        run,
        videoPath: req.file.path,
      });

      updateJob(job.id, {
        progress: 5,
        stage: "Upload received",
      });

      startJobProcessing(job);

      console.log("[JOB START RESPONSE]", { jobId: job.id });

      return res.json({
        ok: true,
        jobId: job.id,
      });
    } catch (error) {
      console.error("START VIDEO JOB ERROR:", error);
      return res.status(500).json({
        error: "Could not start video analysis.",
        details: error.message,
      });
    }
  }
);

app.post("/analyze-run/start", async (req, res) => {
  try {
    const run = req.body || {};

    const job = createJob({
      kind: "text",
      run,
    });

    updateJob(job.id, {
      progress: 5,
      stage: "Analysis request received",
    });

    startJobProcessing(job);

    console.log("[JOB START RESPONSE]", { jobId: job.id });

    return res.json({
      ok: true,
      jobId: job.id,
    });
  } catch (error) {
    console.error("START TEXT JOB ERROR:", error);
    return res.status(500).json({
      error: "Could not start run analysis.",
      details: error.message,
    });
  }
});

app.get("/analysis-status/:jobId", (req, res) => {
  const requestedJobId = String(req.params.jobId || "").trim();

  console.log(
    "[JOB POLL]",
    requestedJobId,
    "exists:",
    jobs.has(requestedJobId),
    "jobCount:",
    jobs.size
  );

  const job = jobs.get(requestedJobId);

  if (!job) {
    return res.status(404).json({
      ok: false,
      error: "Analysis job not found.",
      requestedJobId,
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

// ─── Error Handler ────────────────────────────────────────────────────────────

app.use((error, _req, res, _next) => {
  if (error instanceof multer.MulterError) {
    if (error.code === "LIMIT_FILE_SIZE") {
      return res.status(400).json({
        error: "Uploaded video is too large. Max size is 250 MB.",
      });
    }
    return res.status(400).json({
      error: error.message || "Upload failed.",
    });
  }

  if (error) {
    return res.status(400).json({
      error: error.message || "Request failed.",
    });
  }

  return res.status(500).json({
    error: "Unknown server error.",
  });
});

// ─── Start Server ─────────────────────────────────────────────────────────────

app.listen(PORT, "0.0.0.0", () => {
  console.log(`AI Coaching Server running on port ${PORT}`);
});
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

const pythonExePath =
  process.env.PYTHON_PATH ||
  (process.platform === "win32" ? "python" : "python3");

const pythonScriptPath = path.join(
  process.cwd(),
  "python",
  "analyze_run.py"
);

const uploadsDir = path.join(process.cwd(), "uploads");

console.log("===== SERVER START =====");
console.log("Node ENV:", process.env.NODE_ENV || "development");
console.log("Port:", PORT);
console.log("Python Executable:", pythonExePath);
console.log("Python Script Path:", pythonScriptPath);
console.log("Uploads Directory:", uploadsDir);
console.log("========================");

const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY,
});

const jobs = new Map();
const JOB_TTL_MS = 1000 * 60 * 60;

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

function parseModelJson(outputText) {
  const cleanedOutputText = String(outputText || "")
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/\s*```$/, "")
    .trim();

  return JSON.parse(cleanedOutputText);
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

function roundMaybe(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return null;
  }

  return Number(Number(value).toFixed(digits));
}

function buildCvSummary(run, pythonResult) {
  const sampledFrames = Array.isArray(pythonResult?.sampled_frames)
    ? pythonResult.sampled_frames
    : [];

  const successfulFrames = sampledFrames.filter((f) => f?.read_success);

  const sampledTimestamps = successfulFrames
    .map((f) => f?.timestamp_seconds)
    .filter((value) => value !== null && value !== undefined);

  const previewTimestamps = sampledTimestamps.slice(0, 12).join(", ");

  const trackingQuality = pythonResult?.tracking_quality || {};
  const identifiedBarrels = pythonResult?.identified_barrels || {};
  const turns = pythonResult?.turns || {};
  const splits = pythonResult?.splits || {};
  const barrelMetrics = pythonResult?.barrel_metrics || {};
  const pythonInsights = Array.isArray(pythonResult?.insights)
    ? pythonResult.insights
    : [];

  const barrelSummaryLines = ["barrel1", "barrel2", "barrel3"].map((name) => {
    const barrel = identifiedBarrels?.[name];
    if (!barrel) {
      return `- ${name}: not confidently identified`;
    }

    return `- ${name}: center=(${barrel.center_x}, ${barrel.center_y}), detections=${barrel.detection_count}, avg_conf=${barrel.average_confidence}`;
  });

  const turnSummaryLines = ["barrel1", "barrel2", "barrel3"].map((name) => {
    const turn = turns?.[name];
    if (!turn) {
      return `- ${name}: no turn window confidently identified`;
    }

    return `- ${name}: start=${turn.start_frame}, apex=${turn.apex_frame}, end=${turn.end_frame}, min_distance_px=${turn.min_distance_px}`;
  });

  const metricSummaryLines = ["barrel1", "barrel2", "barrel3"].map((name) => {
    const metric = barrelMetrics?.[name];
    if (!metric) {
      return `- ${name}: no barrel metrics available`;
    }

    return `- ${name}: entry_speed=${metric.entry_speed_px_per_sec ?? "n/a"}, exit_speed=${metric.exit_speed_px_per_sec ?? "n/a"}, path_length=${metric.path_length_px ?? "n/a"}, heading_change=${metric.heading_change_deg ?? "n/a"}`;
  });

  const splitSummary = [
    `- start to barrel 1: ${splits?.start_to_barrel1_seconds ?? "n/a"}`,
    `- barrel 1 to barrel 2: ${splits?.barrel1_to_barrel2_seconds ?? "n/a"}`,
    `- barrel 2 to barrel 3: ${splits?.barrel2_to_barrel3_seconds ?? "n/a"}`,
    `- barrel 3 to home: ${splits?.barrel3_to_home_seconds ?? "n/a"}`,
  ].join("\n");

  const trimmedPythonInsights = pythonInsights
    .slice(0, 8)
    .map((line) => `- ${line}`)
    .join("\n");

  return `
Computer vision summary:
- Video duration (seconds): ${pythonResult?.duration_seconds ?? "unknown"}
- FPS: ${pythonResult?.fps ?? "unknown"}
- Resolution: ${pythonResult?.width ?? "unknown"} x ${pythonResult?.height ?? "unknown"}
- Total sampled frames: ${sampledFrames.length}
- Successfully read sampled frames: ${successfulFrames.length}
- Frames with horse detected: ${pythonResult?.horse_detected_frames ?? "unknown"}
- Raw trajectory points: ${pythonResult?.raw_trajectory_point_count ?? 0}
- Accepted trajectory points: ${pythonResult?.accepted_trajectory_point_count ?? 0}
- Smoothed trajectory points: ${pythonResult?.smoothed_trajectory_point_count ?? 0}
- Sampled frame timestamps (seconds): ${previewTimestamps || "none"}

Tracking quality:
- Read success rate: ${trackingQuality?.read_success_rate ?? "unknown"}
- Horse detection rate: ${trackingQuality?.horse_detection_rate ?? "unknown"}
- Accepted point rate: ${trackingQuality?.accepted_point_rate ?? "unknown"}
- Rejected jump count: ${trackingQuality?.rejected_jump_count ?? "unknown"}

Identified barrels:
${barrelSummaryLines.join("\n")}

Turn windows:
${turnSummaryLines.join("\n")}

Barrel metrics:
${metricSummaryLines.join("\n")}

Estimated split timing:
${splitSummary}

Python-generated insights:
${trimmedPythonInsights || "- none"}

Important limitations:
- All geometry is image-space geometry, not true calibrated arena-space measurement.
- Estimated split timing and speed values are derived from sampled video frames and should be treated as coaching estimates.
- Barrel identity is inferred from clustered detections and may be less reliable when detections are sparse, distant, or partially occluded.
- Use the frames and overlays to confirm posture, pocket, line, and exit quality before making strong claims.

Run data:
Horse: ${run?.horse || ""}
Time: ${run?.time || ""}
Show Name: ${run?.showName || ""}
Location: ${run?.location || ""}
Arena Condition: ${run?.arenaCondition || ""}
Placing: ${run?.placing || ""}
Earnings: ${run?.earnings || ""}
Notes: ${run?.notes || ""}
Rider Feedback: ${run?.riderFeedback || ""}
  `.trim();
}

function buildVisionPrompt(run, pythonResult) {
  const cvSummary = buildCvSummary(run, pythonResult);

  return `
You are an experienced barrel racing coach analyzing a rider's run.

Use the provided run metadata, computer vision summary, and video frame images to evaluate the run.

Evaluate these performance factors as carefully as the evidence supports:
- Entry angle to each barrel
- Pocket size entering each barrel
- Horse shoulder control through each turn
- Rider seat position and balance
- Exit drive leaving each barrel
- Line efficiency between barrels
- Estimated time lost due to mistakes
- Speed/pace quality from:
  - start to barrel 1
  - barrel 1 to barrel 2
  - barrel 2 to barrel 3
  - barrel 3 to home

Important rules:
- Do not claim certainty beyond what the metadata, CV summary, and images support.
- Use the Python barrel identities and turn windows as guidance, but do not blindly trust them if the frames visibly contradict them.
- If something is unclear from the images, say so briefly in the relevant note or explanation.
- Be specific and practical.
- Treat all numeric scores as estimates from 1 to 10, where 10 is strongest.
- For "bestBarrel" and "bestTurn", use only: "1st", "2nd", or "3rd".
- For "focusNext", give one short coaching priority phrase.
- For "estimatedTimeLost", give estimated values as strings like "0.08s".
- Any mention of shoulder control or shoulder drop must refer to the horse, not the rider.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not wrap the JSON in backticks.

${cvSummary}

Return ONLY valid JSON in this exact format:

{
  "summary": "Overall run analysis",
  "strengths": ["", "", ""],
  "issues": ["", "", ""],
  "workOns": ["", "", ""],
  "drills": ["", "", ""],
  "bestBarrel": "1st",
  "bestBarrelReason": "Short explanation of why this was the strongest barrel overall.",
  "bestTurn": "2nd",
  "bestTurnReason": "Short explanation of why this was the strongest turn overall.",
  "focusNext": "Cleaner entry to 1st barrel",
  "pathEfficiency": {
    "score": 7,
    "note": "Short explanation"
  },
  "barrelPocketScore": {
    "barrel1": 6,
    "barrel2": 8,
    "barrel3": 7,
    "note": "Short explanation"
  },
  "entryAngleScore": {
    "barrel1": 6,
    "barrel2": 7,
    "barrel3": 8
  },
  "turnShapeScore": {
    "barrel1": 6,
    "barrel2": 7,
    "barrel3": 8
  },
  "exitDriveScore": {
    "barrel1": 6,
    "barrel2": 7,
    "barrel3": 9
  },
  "riderPositionScore": {
    "score": 7,
    "note": "Short explanation"
  },
  "lineEfficiencyScore": {
    "score": 7,
    "note": "Short explanation"
  },
  "speedScore": {
    "startToBarrel1": 7,
    "barrel1ToBarrel2": 6,
    "barrel2ToBarrel3": 8,
    "barrel3ToHome": 9,
    "note": "Short explanation"
  },
  "estimatedTimeLost": {
    "barrel1": "0.08s",
    "barrel2": "0.03s",
    "barrel3": "0.01s",
    "totalEstimatedTimeLost": "0.12s"
  }
}
  `.trim();
}

function buildTextOnlyPrompt(run) {
  return `
You are an experienced barrel racing coach.

No video was provided for this run. Analyze only the run metadata below.

Important rules:
- Be honest that no video was provided.
- Use the rider feedback, notes, time, and arena condition to make practical coaching inferences.
- If you cannot confidently score a field without video, still provide a reasonable estimate but keep the note cautious.
- Treat all numeric scores as estimates from 1 to 10, where 10 is strongest.
- For "bestBarrel" and "bestTurn", use only: "1st", "2nd", or "3rd".
- For "focusNext", give one short coaching priority phrase.
- For "estimatedTimeLost", give estimated values as strings like "0.08s".
- Any mention of shoulder control or shoulder drop must refer to the horse, not the rider.
- Return ONLY valid JSON.
- Do not use markdown.
- Do not wrap the JSON in backticks.

Return ONLY valid JSON in this exact format:

{
  "summary": "Overall run analysis",
  "strengths": ["", "", ""],
  "issues": ["", "", ""],
  "workOns": ["", "", ""],
  "drills": ["", "", ""],
  "bestBarrel": "1st",
  "bestBarrelReason": "Short explanation of why this was the strongest barrel overall.",
  "bestTurn": "2nd",
  "bestTurnReason": "Short explanation of why this was the strongest turn overall.",
  "focusNext": "Cleaner entry to 1st barrel",
  "pathEfficiency": {
    "score": 7,
    "note": "Short explanation"
  },
  "barrelPocketScore": {
    "barrel1": 6,
    "barrel2": 8,
    "barrel3": 7,
    "note": "Short explanation"
  },
  "entryAngleScore": {
    "barrel1": 6,
    "barrel2": 7,
    "barrel3": 8
  },
  "turnShapeScore": {
    "barrel1": 6,
    "barrel2": 7,
    "barrel3": 8
  },
  "exitDriveScore": {
    "barrel1": 6,
    "barrel2": 7,
    "barrel3": 9
  },
  "riderPositionScore": {
    "score": 7,
    "note": "Short explanation"
  },
  "lineEfficiencyScore": {
    "score": 7,
    "note": "Short explanation"
  },
  "speedScore": {
    "startToBarrel1": 7,
    "barrel1ToBarrel2": 6,
    "barrel2ToBarrel3": 8,
    "barrel3ToHome": 9,
    "note": "Short explanation"
  },
  "estimatedTimeLost": {
    "barrel1": "0.08s",
    "barrel2": "0.03s",
    "barrel3": "0.01s",
    "totalEstimatedTimeLost": "0.12s"
  }
}

Run data:
Horse: ${run?.horse || ""}
Time: ${run?.time || ""}
Show Name: ${run?.showName || ""}
Location: ${run?.location || ""}
Arena Condition: ${run?.arenaCondition || ""}
Placing: ${run?.placing || ""}
Earnings: ${run?.earnings || ""}
Notes: ${run?.notes || ""}
Rider Feedback: ${run?.riderFeedback || ""}
  `.trim();
}

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

    normalized_smoothed_path_points:
      pythonResult.normalized_smoothed_path_points || [],
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
    barrel_metrics: pythonResult.barrel_metrics || {
      barrel1: null,
      barrel2: null,
      barrel3: null,
    },
    insights: Array.isArray(pythonResult.insights) ? pythonResult.insights : [],
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
  return job;
}

function scheduleJobCleanup(jobId) {
  setTimeout(() => {
    jobs.delete(jobId);
  }, JOB_TTL_MS);
}

function updateJob(jobId, updates) {
  const job = jobs.get(jobId);
  if (!job) return null;

  Object.assign(job, updates);
  return job;
}

function selectStrategicFramePaths(sampledFrames, maxFrames = 10) {
  const usable = (sampledFrames || [])
    .filter(
      (frame) =>
        frame?.read_success && (frame?.overlay_image_path || frame?.image_path)
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
      type: "input_image",
      image_url: `data:image/jpeg;base64,${base64}`,
      detail: "auto",
    };
  });
}

async function runPythonAnalysis(videoPath) {
  console.log("Running Python analysis...");
  console.log("Video Path:", videoPath);

  const { stdout, stderr } = await execFileAsync(
    pythonExePath,
    [pythonScriptPath, videoPath],
    { maxBuffer: EXEC_MAX_BUFFER }
  );

  if (stderr) {
    console.warn("Python stderr:", stderr);
  }

  console.log("Python stdout length:", String(stdout || "").length);

  const pythonResult = safeParseJson(stdout);

  if (!pythonResult) {
    console.error("Invalid Python output preview:", String(stdout || "").slice(0, 1000));
    throw new Error("Python returned invalid JSON.");
  }

  if (!pythonResult.ok) {
    throw new Error(pythonResult.error || "Python analysis failed.");
  }

  return pythonResult;
}

async function processVideoJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;

  let pythonGeneratedPaths = [];

  try {
    updateJob(jobId, {
      status: "running",
      progress: 10,
      stage: "Starting computer vision",
      startedAt: new Date().toISOString(),
    });

    const pythonResult = await runPythonAnalysis(job.videoPath);

    updateJob(jobId, {
      progress: 50,
      stage: "Computer vision finished",
    });

    pythonGeneratedPaths = getPythonGeneratedPaths(pythonResult);

    updateJob(jobId, {
      progress: 60,
      stage: "Selecting strategic frames",
    });

    const framePaths = selectStrategicFramePaths(
      pythonResult.sampled_frames || [],
      10
    );

    if (!framePaths.length) {
      throw new Error("Python did not return any usable frame images.");
    }

    updateJob(jobId, {
      progress: 72,
      stage: "Preparing frame images for AI",
    });

    const imageInputs = buildImageInputs(framePaths);

    updateJob(jobId, {
      progress: 80,
      stage: "Requesting AI coaching analysis",
    });

    const response = await openai.responses.create({
      model: "gpt-4o-mini",
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: buildVisionPrompt(job.run, pythonResult),
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

    const outputText = response.output_text;

    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error("Invalid OpenAI JSON output preview:", String(outputText || "").slice(0, 1000));
      throw new Error("OpenAI returned invalid JSON.");
    }

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      stage: "Completed",
      completedAt: new Date().toISOString(),
      result: {
        success: true,
        analysis: parsedAnalysis,
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
    safeCleanup([job.videoPath, ...pythonGeneratedPaths]);
  }
}

async function processTextJob(jobId) {
  const job = jobs.get(jobId);
  if (!job) return;

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

    const response = await openai.responses.create({
      model: "gpt-4o-mini",
      input: [
        {
          role: "user",
          content: [
            {
              type: "input_text",
              text: buildTextOnlyPrompt(job.run),
            },
          ],
        },
      ],
    });

    updateJob(jobId, {
      progress: 95,
      stage: "Finalizing analysis",
    });

    const outputText = response.output_text;

    let parsedAnalysis;
    try {
      parsedAnalysis = parseModelJson(outputText);
    } catch {
      console.error("Invalid OpenAI JSON output preview:", String(outputText || "").slice(0, 1000));
      throw new Error("OpenAI returned invalid JSON.");
    }

    updateJob(jobId, {
      status: "completed",
      progress: 100,
      stage: "Completed",
      completedAt: new Date().toISOString(),
      result: {
        success: true,
        analysis: parsedAnalysis,
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
  if (job.kind === "video") {
    void processVideoJob(job.id);
    return;
  }

  if (job.kind === "text") {
    void processTextJob(job.id);
  }
}

app.get("/", (_req, res) => {
  res.json({
    ok: true,
    message: "AI Coaching Server running",
  });
});

app.get("/health", (_req, res) => {
  res.json({
    ok: true,
    message: "AI Coaching Server running",
  });
});

app.get("/test-python", async (_req, res) => {
  try {
    const testVideoPath = "test-video.mp4";
    const pythonResult = await runPythonAnalysis(testVideoPath);

    return res.json({
      ok: true,
      python: pythonResult,
    });
  } catch (error) {
    console.error("Python execution error:", error);
    return res.status(500).json({
      ok: false,
      error: error.message,
    });
  }
});

app.post("/analyze-run-video/start", upload.single("video"), async (req, res) => {
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
});

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
  const job = jobs.get(req.params.jobId);

  if (!job) {
    return res.status(404).json({
      error: "Analysis job not found.",
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

app.listen(PORT, "0.0.0.0", () => {
  console.log(`AI Coaching Server running on port ${PORT}`);
});
// One-off diagnostic script — NOT a server route, run manually.
// Builds the re-engagement campaign list: every user who has never added a
// horse and is not currently a paying/entitled RevenueCat subscriber.
// Writes outputs/reengagement-list.csv (email,userId). No emails are printed
// to the console — the CSV is the only place addresses appear.
//
// Usage:
//   node scripts/reengagement-list.js
//
// Required env vars (same values already used by server.js / set in Render):
//   FIREBASE_SERVICE_ACCOUNT   - JSON service account string
//   REVENUECAT_API_KEY         - RevenueCat secret key (Bearer token)
//
// Optional:
//   REVENUECAT_PROJECT_ID      - pin the V2 project id instead of auto-discovering it

import dotenv from "dotenv";
dotenv.config();

import { mkdirSync, writeFileSync } from "fs";
import { dirname, resolve } from "path";

const OUTPUT_PATH = resolve(process.cwd(), "outputs/reengagement-list.csv");
const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

const REVENUECAT_API_KEY = process.env.REVENUECAT_API_KEY;
if (!REVENUECAT_API_KEY) {
  console.error("Missing REVENUECAT_API_KEY environment variable.");
  process.exit(1);
}

// ─── Firebase Admin (same init pattern as server.js) ──────────────────────

const { initializeApp, cert, getApps } = await import("firebase-admin/app");
const { getFirestore } = await import("firebase-admin/firestore");

if (!getApps().length) {
  const serviceAccount = process.env.FIREBASE_SERVICE_ACCOUNT
    ? JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT)
    : null;
  if (!serviceAccount) {
    console.error("Missing FIREBASE_SERVICE_ACCOUNT environment variable.");
    process.exit(1);
  }
  initializeApp({ credential: cert(serviceAccount) });
}
const db = getFirestore();

// ─── Step 1: build the candidate cohort from Firestore ────────────────────
// "Has horses" is the UNION of two sources that co-exist in this data set:
//   a) users/{uid}/horses          - subcollection, one doc per horse
//   b) users/{uid}/profile/horses  - single doc holding a `horses` array
// Roughly a fifth of horse owners only appear in (a), so checking (b) alone
// (as funnel-check.js does) undercounts owners and would leak them into this
// campaign list.
//
// Minor/guardian accounts are dropped entirely: account/profile.userEmail is
// the MINOR's own address, not the guardian's, so it is not a valid marketing
// destination.

function isMinorAccount(profile) {
  if (!profile) return false;
  if (profile.guardianEmail || profile.guardianName || profile.parentalConsentAt) return true;
  return typeof profile.age === "number" && profile.age < 18;
}

async function buildCohort() {
  const userRefs = await db.collection("users").listDocuments();

  // One collection-group read covers the subcollection side for every user.
  const hasHorseSubcollection = new Set();
  const horseDocs = await db.collectionGroup("horses").select().get();
  horseDocs.forEach((doc) => {
    const userDoc = doc.ref.parent.parent; // users/{uid}
    if (userDoc) hasHorseSubcollection.add(userDoc.id);
  });

  const stats = {
    totalUsers: userRefs.length,
    hasHorses: 0,
    minorsExcluded: 0,
    missingProfile: 0,
    unusableEmail: 0,
  };
  const cohort = [];

  for (const ref of userRefs) {
    const uid = ref.id;

    const horsesArrayDoc = await ref.collection("profile").doc("horses").get().catch(() => null);
    const horsesArray =
      horsesArrayDoc && horsesArrayDoc.exists && Array.isArray(horsesArrayDoc.data()?.horses)
        ? horsesArrayDoc.data().horses
        : [];

    if (hasHorseSubcollection.has(uid) || horsesArray.length > 0) {
      stats.hasHorses++;
      continue;
    }

    const profileDoc = await ref.collection("account").doc("profile").get().catch(() => null);
    const profile = profileDoc && profileDoc.exists ? profileDoc.data() : null;

    if (!profile) {
      stats.missingProfile++;
      continue;
    }
    if (isMinorAccount(profile)) {
      stats.minorsExcluded++;
      continue;
    }

    const email = typeof profile.userEmail === "string" ? profile.userEmail.trim() : "";
    if (!EMAIL_RE.test(email)) {
      stats.unusableEmail++;
      continue;
    }

    cohort.push({ uid, email });
  }

  return { cohort, stats };
}

// ─── Step 2: RevenueCat lookups ────────────────────────────────────────────
// V2 is tried first (per project, via /customers/{id}/subscriptions, using
// the `gives_access` field). Promotional/comped grants report gives_access
// true just like real purchases, so they are excluded from the campaign
// automatically — no hardcoded email list needed.

let resolvedProjectId = process.env.REVENUECAT_PROJECT_ID || null;
let projectResolutionFailed = false;
const apiVersionUsedCounts = { v2: 0, v1: 0 };

async function resolveProjectId() {
  if (resolvedProjectId || projectResolutionFailed) return resolvedProjectId;
  try {
    const res = await fetch("https://api.revenuecat.com/v2/projects", {
      headers: { Authorization: `Bearer ${REVENUECAT_API_KEY}` },
    });
    if (!res.ok) {
      throw new Error(`status ${res.status}: ${await res.text()}`);
    }
    const data = await res.json();
    const projects = data.items || [];
    if (projects.length === 0) throw new Error("key has access to zero V2 projects");
    if (projects.length > 1) {
      console.warn(
        "[V2] Multiple RevenueCat projects visible to this key — using the first:",
        projects.map((p) => `${p.id} (${p.name || "unnamed"})`).join(", ")
      );
      console.warn("[V2] Set REVENUECAT_PROJECT_ID env var to pin a specific project if this is wrong.");
    }
    resolvedProjectId = projects[0].id;
  } catch (err) {
    console.warn("[V2] Could not resolve project id, will fall back to V1 for all lookups:", err.message);
    projectResolutionFailed = true;
  }
  return resolvedProjectId;
}

async function checkActiveV2(appUserId) {
  const projectId = await resolveProjectId();
  if (!projectId) return { ok: false, found: false };

  const url = `https://api.revenuecat.com/v2/projects/${projectId}/customers/${encodeURIComponent(appUserId)}/subscriptions`;
  const res = await fetch(url, { headers: { Authorization: `Bearer ${REVENUECAT_API_KEY}` } });

  if (res.status === 404) return { ok: true, found: false, active: false };
  if (!res.ok) return { ok: false, found: false, error: `V2 status ${res.status}: ${await res.text()}` };

  const data = await res.json();
  const subs = data.items || [];
  if (subs.length === 0) return { ok: true, found: false, active: false };

  const active = subs.some((s) => s.gives_access === true);
  return { ok: true, found: true, active };
}

async function checkActiveV1(appUserId) {
  const url = `https://api.revenuecat.com/v1/subscribers/${encodeURIComponent(appUserId)}`;
  const res = await fetch(url, { headers: { Authorization: `Bearer ${REVENUECAT_API_KEY}` } });

  if (res.status === 404) return { ok: true, found: false, active: false };
  if (!res.ok) return { ok: false, found: false, error: `V1 status ${res.status}: ${await res.text()}` };

  const data = await res.json();
  const entitlements = data.subscriber?.entitlements || {};
  const subscriptions = data.subscriber?.subscriptions || {};
  const now = Date.now();

  const active = Object.values(entitlements).some((e) => {
    if (!e.expires_date) return true; // null = non-expiring / lifetime access
    return new Date(e.expires_date).getTime() > now;
  });

  const found = Object.keys(entitlements).length > 0 || Object.keys(subscriptions).length > 0;
  return { ok: true, found, active };
}

async function checkActive(appUserId) {
  const v2 = await checkActiveV2(appUserId);
  if (!v2.ok && v2.error) {
    console.warn(`[V2] Lookup error for ${appUserId}, falling back to V1: ${v2.error}`);
  }
  if (v2.ok && v2.found) {
    apiVersionUsedCounts.v2++;
    return v2.active;
  }
  // V2 answered successfully and this customer simply has no subscriptions.
  // That is a definitive "not a subscriber" — do NOT fall through to V1.
  // V2-format secret keys are rejected outright by V1 (code 7723), which would
  // turn every no-record user into a thrown lookup failure.
  if (v2.ok) {
    apiVersionUsedCounts.v2++;
    return false;
  }

  const v1 = await checkActiveV1(appUserId);
  if (!v1.ok) {
    throw new Error(v1.error || `V1 lookup failed for ${appUserId}`);
  }
  if (v1.found) apiVersionUsedCounts.v1++;
  return v1.active;
}

// ─── CSV ───────────────────────────────────────────────────────────────────

function csvCell(value) {
  const s = String(value ?? "");
  return /[",\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

function writeCsv(rows) {
  const lines = ["email,userId", ...rows.map((r) => `${csvCell(r.email)},${csvCell(r.uid)}`)];
  mkdirSync(dirname(OUTPUT_PATH), { recursive: true });
  writeFileSync(OUTPUT_PATH, lines.join("\n") + "\n", "utf8");
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  console.log("\nBuilding re-engagement list (users with no horses, not subscribed)...\n");

  const { cohort, stats } = await buildCohort();
  console.log(`Scanned ${stats.totalUsers} user(s); ${cohort.length} candidate(s) before RevenueCat check.\n`);

  const notSubscribed = [];
  let activeCount = 0;
  const failures = [];

  for (const entry of cohort) {
    try {
      const active = await checkActive(entry.uid);
      if (active) activeCount++;
      else notSubscribed.push(entry);
    } catch (err) {
      failures.push({ uid: entry.uid, error: err.message });
      console.warn(`ERROR     ${entry.uid} — ${err.message}`);
    }
  }

  // One address can be attached to more than one account; the campaign should
  // only ever mail it once, so the CSV is deduplicated on email.
  const seen = new Set();
  const rows = [];
  for (const entry of notSubscribed) {
    const key = entry.email.toLowerCase();
    if (seen.has(key)) continue;
    seen.add(key);
    rows.push(entry);
  }
  const duplicatesDropped = notSubscribed.length - rows.length;

  writeCsv(rows);

  const versionSummary =
    apiVersionUsedCounts.v2 === 0 && apiVersionUsedCounts.v1 === 0
      ? "unknown — no successful lookups"
      : `V2 for ${apiVersionUsedCounts.v2} user(s), V1 fallback for ${apiVersionUsedCounts.v1} user(s)`;

  console.log("\n──────────────────────────────────────────");
  console.log(`Total accounts scanned:      ${stats.totalUsers}`);
  console.log(`Excluded — has horses:       ${stats.hasHorses}`);
  console.log(`Excluded — minor/guardian:   ${stats.minorsExcluded}`);
  console.log(`Excluded — no profile doc:   ${stats.missingProfile}`);
  console.log(`Excluded — unusable email:   ${stats.unusableEmail}`);
  console.log(`Excluded — active RevenueCat:${String(activeCount).padStart(4)}`);
  console.log(`Failed lookups:              ${failures.length}`);
  console.log(`RevenueCat API used:         ${versionSummary}`);
  console.log(`Duplicate addresses dropped: ${duplicatesDropped}`);
  console.log(`FINAL LIST (unique emails):  ${rows.length}`);
  console.log(`Written to:                  ${OUTPUT_PATH}`);
  console.log("──────────────────────────────────────────\n");

  if (failures.length > 0) {
    console.log("Users that failed lookup (NOT included in the CSV — verify manually):");
    for (const f of failures) console.log(`  - ${f.uid}: ${f.error}`);
  }
}

main().catch((err) => {
  console.error("[FATAL]", err);
  process.exit(1);
});

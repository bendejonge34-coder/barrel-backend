// One-off diagnostic script — NOT a server route, run manually.
// Calculates ambassador payout for a referral code: finds every user who
// redeemed the code, checks each one's CURRENT RevenueCat subscription
// status, and prints a payout total ($5/subscriber, capped at $200/40 subs).
//
// Usage:
//   node scripts/referral-payout.js MICHIGAN2026
//
// Required env vars (same values already used by server.js / set in Render):
//   FIREBASE_SERVICE_ACCOUNT   - JSON service account string
//   REVENUECAT_API_KEY         - RevenueCat secret key (Bearer token)
//
// Optional:
//   REVENUECAT_PROJECT_ID      - pin the V2 project id instead of auto-discovering it

import dotenv from "dotenv";
dotenv.config();

const PAYOUT_PER_SUBSCRIBER = 5;
const PAYOUT_CAP = 200;
const MAX_PAYABLE_SUBS = PAYOUT_CAP / PAYOUT_PER_SUBSCRIBER; // 40

const REFERRAL_CODE = process.argv[2];
if (!REFERRAL_CODE) {
  console.error("Usage: node scripts/referral-payout.js <REFERRAL_CODE>");
  process.exit(1);
}

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

// ─── Step 1: find every user who redeemed this code ───────────────────────
// Schema: users/{uid}/account/profile, field "referralCodeUsed".
// Collection-group query on "account", then keep only the "profile" doc
// (the subcollection may hold other doc types).

async function findRedeemers(code) {
  let snap;
  try {
    snap = await db
      .collectionGroup("account")
      .where("referralCodeUsed", "==", code)
      .get();
  } catch (err) {
    if (err.code === 9 || /index/i.test(err.message || "")) {
      console.error("\n[FIRESTORE INDEX REQUIRED]");
      console.error("This collection-group query needs a composite index that doesn't exist yet.");
      console.error("Firestore's error below includes a direct console link to create it:\n");
      console.error(err.message);
      console.error("\nCreate the index, wait for it to finish building (a few minutes), then re-run this script.");
      process.exit(1);
    }
    throw err;
  }

  const uids = [];
  for (const doc of snap.docs) {
    if (doc.id !== "profile") continue;
    const userDoc = doc.ref.parent.parent; // users/{uid}
    if (userDoc) uids.push(userDoc.id);
  }
  return uids;
}

// ─── Step 2: RevenueCat lookups ────────────────────────────────────────────
// V2 is tried first (per project, via /customers/{id}/subscriptions, using
// the `gives_access` field). If V2 returns no data for a user (e.g. the key
// in use doesn't have V2 access, or the project id can't be resolved), each
// user falls back to the V1 subscriber endpoint independently.

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

  const v1 = await checkActiveV1(appUserId);
  if (!v1.ok) {
    throw new Error(v1.error || `V1 lookup failed for ${appUserId}`);
  }
  if (v1.found) apiVersionUsedCounts.v1++;
  return v1.active;
}

// ─── Main ───────────────────────────────────────────────────────────────────

async function main() {
  console.log(`\nLooking up redeemers of referral code: ${REFERRAL_CODE}\n`);

  const uids = await findRedeemers(REFERRAL_CODE);
  console.log(`Found ${uids.length} user(s) who redeemed this code.\n`);

  if (uids.length === 0) {
    console.log("Nothing to pay out.");
    return;
  }

  let activeCount = 0;
  const failures = [];

  for (const uid of uids) {
    try {
      const active = await checkActive(uid);
      if (active) activeCount++;
      console.log(`${active ? "ACTIVE  " : "inactive"}  ${uid}`);
    } catch (err) {
      failures.push({ uid, error: err.message });
      console.warn(`ERROR     ${uid} — ${err.message}`);
    }
  }

  const payableSubs = Math.min(activeCount, MAX_PAYABLE_SUBS);
  const payout = payableSubs * PAYOUT_PER_SUBSCRIBER;

  const versionSummary =
    apiVersionUsedCounts.v2 === 0 && apiVersionUsedCounts.v1 === 0
      ? "unknown — no successful lookups"
      : `V2 for ${apiVersionUsedCounts.v2} user(s), V1 fallback for ${apiVersionUsedCounts.v1} user(s)`;

  console.log("\n──────────────────────────────────────────");
  console.log(`Referral code:            ${REFERRAL_CODE}`);
  console.log(`Total redeemers found:    ${uids.length}`);
  console.log(`Currently active subs:    ${activeCount}`);
  console.log(`Failed lookups:           ${failures.length}`);
  console.log(`RevenueCat API used:      ${versionSummary}`);
  console.log(
    `Payable subscribers:      ${payableSubs}${activeCount > MAX_PAYABLE_SUBS ? ` (capped from ${activeCount})` : ""}`
  );
  console.log(`Payout owed:              $${payout}`);
  console.log("──────────────────────────────────────────\n");

  if (failures.length > 0) {
    console.log("Users that failed lookup (verify manually before finalizing payout):");
    for (const f of failures) console.log(`  - ${f.uid}: ${f.error}`);
  }
}

main().catch((err) => {
  console.error("[FATAL]", err);
  process.exit(1);
});

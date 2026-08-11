// One-off diagnostic script — NOT a server route, run manually.
// READ-ONLY. This script never writes to Firestore.
//
// Hunts for the horse-data-loss pattern already confirmed on one paying
// account: account/profile says horsesMigrated: true, but the users/{uid}/horses
// subcollection is empty, while that account's runs still reference horse names
// as embedded text (run.horse). That combination means the horse records the
// runs were logged against are gone from the cloud.
//
// Usage:
//   node scripts/horse-data-integrity-check.js
//
// Required env vars (same values already used by server.js / set in Render):
//   FIREBASE_SERVICE_ACCOUNT   - JSON service account string
//
// No RevenueCat calls are made. Subscription status is reported only if it is
// visible in Firestore already; otherwise it prints "unknown".
//
// Privacy: horse names and emails are never printed — only counts.

import dotenv from "dotenv";
dotenv.config();

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

// ─── Helpers ───────────────────────────────────────────────────────────────

// Collection-group queries match same-named collections at ANY depth. Every
// collection we care about hangs directly off users/{uid}, so anything nested
// deeper (or parked under a different root) is ignored rather than being
// mis-attributed to a user.
function ownerUidOf(doc) {
  const userDoc = doc.ref.parent.parent; // e.g. users/{uid}
  if (!userDoc) return null;
  if (userDoc.parent.id !== "users") return null;
  return userDoc.id;
}

// Horse names in runs are free text typed by the rider, so "Dash", "dash " and
// "DASH" are the same horse. Normalize before counting distinct values.
function normalizeHorseName(value) {
  if (typeof value !== "string") return null;
  const trimmed = value.trim();
  if (!trimmed) return null;
  return trimmed.toLowerCase();
}

// The backend never writes horsesMigrated (the mobile client does), so treat it
// as an opaque field: report exactly what is there, and don't assume the only
// possible values are true/false.
function readMigratedFlag(profile) {
  if (!profile || !("horsesMigrated" in profile)) return { value: "missing", isTrue: false };
  const raw = profile.horsesMigrated;
  return { value: JSON.stringify(raw), isTrue: raw === true };
}

// There is no known subscription mirror in Firestore — entitlements live in
// RevenueCat, and we are explicitly not calling their API here. Rather than
// hardcode a field name that may not exist, scan the profile for any key that
// looks subscription-related and report the scalar values found. If nothing
// matches, the honest answer is "unknown".
const SUBSCRIPTION_KEY_RE =
  /subscri|entitle|revenue_?cat|premium|purchase|billing|paywall|trial|\bplan\b|\btier\b|\bpaid\b|\bpro\b/i;

function readSubscriptionSignal(profile, extraAccountDocIds) {
  const parts = [];

  for (const [key, value] of Object.entries(profile || {})) {
    if (!SUBSCRIPTION_KEY_RE.test(key)) continue;
    // Only scalars — objects/arrays could carry PII or receipt payloads.
    if (value === null || ["string", "number", "boolean"].includes(typeof value)) {
      parts.push(`${key}=${JSON.stringify(value)}`);
    } else {
      parts.push(`${key}=<${Array.isArray(value) ? "array" : typeof value}>`);
    }
  }

  for (const docId of extraAccountDocIds) {
    if (SUBSCRIPTION_KEY_RE.test(docId)) parts.push(`account/${docId} doc exists`);
  }

  return parts.length ? parts.join(", ") : "unknown";
}

// ─── Step 1: bulk-load everything with collection-group reads ─────────────
// One query per collection instead of four reads per user — the per-user loop
// below then does zero further I/O.

async function loadSnapshot() {
  const userRefs = await db.collection("users").listDocuments();

  console.log(`Found ${userRefs.length} account(s). Loading subcollections...`);

  const [accountSnap, horsesSnap, runsSnap, profileSnap] = await Promise.all([
    db.collectionGroup("account").get(),
    db.collectionGroup("horses").select().get(),
    db.collectionGroup("runs").select("horse").get(),
    db.collectionGroup("profile").get(),
  ]);

  // users/{uid}/account/profile — the doc carrying horsesMigrated.
  const profiles = new Map();
  // Any other users/{uid}/account/* doc ids, in case subscription state is
  // stored as a sibling doc rather than a profile field.
  const otherAccountDocs = new Map();
  for (const doc of accountSnap.docs) {
    const uid = ownerUidOf(doc);
    if (!uid) continue;
    if (doc.id === "profile") profiles.set(uid, doc.data() || {});
    else {
      if (!otherAccountDocs.has(uid)) otherAccountDocs.set(uid, []);
      otherAccountDocs.get(uid).push(doc.id);
    }
  }

  // users/{uid}/horses — one doc per horse.
  const horseCounts = new Map();
  for (const doc of horsesSnap.docs) {
    const uid = ownerUidOf(doc);
    if (!uid) continue;
    horseCounts.set(uid, (horseCounts.get(uid) || 0) + 1);
  }

  // users/{uid}/profile/horses — legacy single doc holding a `horses` array.
  // Both storage shapes co-exist in this data set (see reengagement-list.js),
  // so an empty subcollection alone does not prove the data is gone.
  const legacyHorseCounts = new Map();
  for (const doc of profileSnap.docs) {
    if (doc.id !== "horses") continue;
    const uid = ownerUidOf(doc);
    if (!uid) continue;
    const arr = doc.data()?.horses;
    legacyHorseCounts.set(uid, Array.isArray(arr) ? arr.length : 0);
  }

  // users/{uid}/runs — total count plus the distinct horse names referenced.
  const runCounts = new Map();
  const runHorseNames = new Map();
  for (const doc of runsSnap.docs) {
    const uid = ownerUidOf(doc);
    if (!uid) continue;
    runCounts.set(uid, (runCounts.get(uid) || 0) + 1);

    const name = normalizeHorseName(doc.data()?.horse);
    if (!name) continue;
    if (!runHorseNames.has(uid)) runHorseNames.set(uid, new Set());
    runHorseNames.get(uid).add(name);
  }

  return { userRefs, profiles, otherAccountDocs, horseCounts, legacyHorseCounts, runCounts, runHorseNames };
}

// ─── Step 2: classify each account ────────────────────────────────────────

async function main() {
  console.log("\nHorse data integrity check (READ-ONLY — nothing is written)\n");

  const snapshot = await loadSnapshot();
  const { userRefs, profiles, otherAccountDocs, horseCounts, legacyHorseCounts, runCounts, runHorseNames } = snapshot;

  const suspects = [];
  const weakSignals = [];
  let noProfile = 0;

  for (const ref of userRefs) {
    const uid = ref.id;

    const profile = profiles.get(uid) || null;
    if (!profile) noProfile++;

    const migrated = readMigratedFlag(profile);
    const horses = horseCounts.get(uid) || 0;
    const legacyHorses = legacyHorseCounts.get(uid) || 0;
    const runs = runCounts.get(uid) || 0;
    const distinctRunHorses = (runHorseNames.get(uid) || new Set()).size;

    const row = {
      uid,
      horsesMigrated: migrated.value,
      horses,
      legacyHorses,
      runs,
      distinctRunHorses,
      subscription: readSubscriptionSignal(profile, otherAccountDocs.get(uid) || []),
    };

    // SUSPECT: migration was expected, the subcollection is empty (or holds far
    // fewer horses than the runs reference), and there are runs to prove horses
    // once existed.
    const migrationExpected = migrated.isTrue;
    const subcollectionShort = horses === 0 || (distinctRunHorses >= 2 && horses < distinctRunHorses);

    if (migrationExpected && subcollectionShort && runs >= 1 && distinctRunHorses >= 1) {
      suspects.push(row);
      continue;
    }

    // WEAKER SIGNAL: fewer horse docs than distinct horse names in runs, but the
    // account didn't meet the full SUSPECT bar — e.g. one horse never synced
    // while the rest did, or horsesMigrated was never set.
    if (runs >= 1 && distinctRunHorses > horses) {
      weakSignals.push(row);
    }
  }

  // ─── Report ──────────────────────────────────────────────────────────────
  // Counts only. No horse names, no emails, no run data.

  const line = (r) =>
    `  ${r.uid}\n` +
    `      horsesMigrated: ${r.horsesMigrated}   horses subcollection: ${r.horses}   ` +
    `legacy profile/horses array: ${r.legacyHorses}\n` +
    `      runs: ${r.runs}   distinct horse names in runs: ${r.distinctRunHorses}\n` +
    `      subscription (Firestore only): ${r.subscription}`;

  console.log("\n===== SUSPECT ACCOUNTS =====\n");
  if (suspects.length === 0) {
    console.log("  none");
  } else {
    for (const r of suspects) console.log(line(r) + "\n");
  }

  console.log("\n===== WEAKER SIGNAL (partial mismatch, not called suspect) =====\n");
  if (weakSignals.length === 0) {
    console.log("  none");
  } else {
    for (const r of weakSignals) console.log(line(r) + "\n");
  }

  // A suspect whose legacy array still holds horses has NOT lost data — the
  // records are sitting in the old storage shape. Split the total so the truly
  // empty accounts are obvious.
  const suspectsWithNoHorseDataAnywhere = suspects.filter((r) => r.legacyHorses === 0).length;
  const suspectsRecoverableFromLegacy = suspects.length - suspectsWithNoHorseDataAnywhere;

  console.log("\n──────────────────────────────────────────");
  console.log(`Total accounts scanned:          ${userRefs.length}`);
  console.log(`Accounts with no account/profile:${String(noProfile).padStart(5)}`);
  console.log(`Flagged SUSPECT:                 ${suspects.length}`);
  console.log(`  — no horse data anywhere:      ${suspectsWithNoHorseDataAnywhere}`);
  console.log(`  — legacy array still populated:${String(suspectsRecoverableFromLegacy).padStart(4)}`);
  console.log(`Flagged WEAKER SIGNAL:           ${weakSignals.length}`);
  console.log("──────────────────────────────────────────");
  console.log("This script performed reads only. Nothing in Firestore was modified.\n");
}

main().catch((err) => {
  console.error("[FATAL]", err);
  process.exit(1);
});

import 'dotenv/config';
import admin from 'firebase-admin';

const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });
const db = admin.firestore();

// Finds the first array field in a doc and returns its length.
// Defensive because I'm not 100% sure of your horses/riders field name.
function arrayLen(data) {
  if (!data) return 0;
  for (const k of Object.keys(data)) {
    if (Array.isArray(data[k])) return data[k].length;
  }
  return 0;
}

async function main() {
  const userRefs = await db.collection('users').listDocuments();
  const total = userRefs.length;
  console.log('Total accounts: ' + total);
  console.log('Scanning (this takes a minute)...\n');

  const runsSnap = await db.collectionGroup('runs').select().get();
  const runsPerUser = new Map();
  runsSnap.forEach(d => {
    const uid = d.ref.parent.parent.id;
    runsPerUser.set(uid, (runsPerUser.get(uid) || 0) + 1);
  });

  let didConsent = 0, hasHorse = 0, hasRider = 0, hasRun = 0, emptyShell = 0;
  let probed = false;

  for (const ref of userRefs) {
    const uid = ref.id;

    const [consentSnap, horsesSnap, ridersSnap] = await Promise.all([
      db.doc(`users/${uid}/consent/privacy`).get().catch(() => null),
      db.doc(`users/${uid}/profile/horses`).get().catch(() => null),
      db.doc(`users/${uid}/profile/riders`).get().catch(() => null),
    ]);

    // One-time schema probe so we can confirm the paths are right
    if (!probed && horsesSnap && horsesSnap.exists) {
      console.log('--- schema probe (field names only, no values) ---');
      console.log('horses doc fields:', Object.keys(horsesSnap.data() || {}));
      console.log('riders doc fields:', Object.keys((ridersSnap && ridersSnap.data()) || {}));
      console.log('--------------------------------------------------\n');
      probed = true;
    }

    const consented = !!(consentSnap && consentSnap.exists);
    const horses = horsesSnap && horsesSnap.exists ? arrayLen(horsesSnap.data()) : 0;
    const riders = ridersSnap && ridersSnap.exists ? arrayLen(ridersSnap.data()) : 0;
    const runs = runsPerUser.get(uid) || 0;

    if (consented) didConsent++;
    if (horses > 0) hasHorse++;
    if (riders > 0) hasRider++;
    if (runs > 0) hasRun++;
    if (!consented && horses === 0 && riders === 0 && runs === 0) emptyShell++;
  }

  const pct = n => total ? ((n / total) * 100).toFixed(1) + '%' : '0%';

  console.log('\n===== BARREL PRO ACTIVATION FUNNEL =====\n');
  console.log('1. Account created:      ' + total + '   (100%)');
  console.log('2. Completed consent:    ' + didConsent + '   (' + pct(didConsent) + ')');
  console.log('3. Added a horse:        ' + hasHorse + '   (' + pct(hasHorse) + ')');
  console.log('4. Added a rider:        ' + hasRider + '   (' + pct(hasRider) + ')');
  console.log('5. Logged a run:         ' + hasRun + '   (' + pct(hasRun) + ')');
  console.log('');
  console.log('Accounts with NOTHING at all: ' + emptyShell + '   (' + pct(emptyShell) + ')');
  console.log('\n========================================\n');
}

main().then(() => process.exit(0)).catch(err => {
  console.error('ERROR:', err.message);
  process.exit(1);
});
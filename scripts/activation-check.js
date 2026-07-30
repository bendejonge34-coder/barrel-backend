import 'dotenv/config';
import admin from 'firebase-admin';

const serviceAccount = JSON.parse(process.env.FIREBASE_SERVICE_ACCOUNT);
admin.initializeApp({ credential: admin.credential.cert(serviceAccount) });

const db = admin.firestore();

async function main() {
  console.log('Counting users...');

  const userRefs = await db.collection('users').listDocuments();
  const totalUsers = userRefs.length;

  console.log('Counting runs...');

  const runsSnap = await db.collectionGroup('runs').select().get();

  const runsPerUser = new Map();
  runsSnap.forEach(doc => {
    const uid = doc.ref.parent.parent.id;
    runsPerUser.set(uid, (runsPerUser.get(uid) || 0) + 1);
  });

  const totalRuns = runsSnap.size;
  const usersWithRuns = runsPerUser.size;
  const usersWithNoRuns = totalUsers - usersWithRuns;

  let b1to4 = 0, b5to19 = 0, b20plus = 0;
  for (const count of runsPerUser.values()) {
    if (count >= 20) b20plus++;
    else if (count >= 5) b5to19++;
    else b1to4++;
  }

  const counts = [...runsPerUser.values()].sort((a, b) => a - b);
  const median = counts.length ? counts[Math.floor(counts.length / 2)] : 0;
  const max = counts.length ? counts[counts.length - 1] : 0;

  const pct = n => totalUsers ? ((n / totalUsers) * 100).toFixed(1) + '%' : '0%';

  console.log('\n===== BARREL PRO ACTIVATION =====\n');
  console.log('Total accounts:        ' + totalUsers);
  console.log('Total runs logged:     ' + totalRuns);
  console.log('');
  console.log('0 runs (never used):   ' + usersWithNoRuns + '   (' + pct(usersWithNoRuns) + ')');
  console.log('1-4 runs (tried it):   ' + b1to4 + '   (' + pct(b1to4) + ')');
  console.log('5-19 runs (engaged):   ' + b5to19 + '   (' + pct(b5to19) + ')');
  console.log('20+ runs (committed):  ' + b20plus + '   (' + pct(b20plus) + ')');
  console.log('');
  console.log('Median runs (among users who logged any): ' + median);
  console.log('Most runs by one user:                    ' + max);
  console.log('\n=================================\n');
}

main()
  .then(() => process.exit(0))
  .catch(err => {
    console.error('ERROR:', err.message);
    process.exit(1);
  });
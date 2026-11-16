/**
 * How optimistic is the planner's in-sample certification? A SYNTHETIC sweep with KNOWN truth (no cache, no network, no key).
 *
 *   npm run bench:rag:certify:sweep [-- --reps 300 --jobs 10 --sizes 40,100,300,1000]
 *
 * Every cell is a synthetic workload: 3 plans (cost 1, 2, 5) whose per-question success probability is
 * s_p(d) = trueMean_p + slope_p * (0.5 - d) (clipped to [0.01, 0.995]) for a question "difficulty" d ~ U(0,1). The true mean of a plan
 * is the numeric integral of s_p over d, so the truth is exact (no finite test mean). Cells vary:
 *   calibration size n in {40, 100, 300, 1000};  scores 0/1 (Bernoulli) vs graded (Beta with the same mean, concentration 4);
 *   features random vs informative (difficulty visible: [cos(pi d/2), sin(pi d/2)]);
 *   world: BELOW (best plan truly SLA - 0.02), AT (best plan truly = SLA), POWER (best plan truly SLA + 0.05).
 * Per repetition (fresh seed, fresh questions, independent): a planner with sla.confidence 0.9 calibrates on n questions.
 *   in-sample certified  = the planner's own slaCertified;
 *   split certified      = ONE Wilson lower bound (z of 0.9) of the frozen policy's realized quality on 0.75 n FRESH questions >= SLA
 *                          (the T-025 protocol with fresh draws instead of a fixed pool: no complementary-splits effect);
 *   truth                = the frozen policy's exact expected quality over a grid of 400 difficulties (via decide()), NOT a finite test mean.
 *   false certification  = certified AND truth < SLA.
 * Reports where the in-sample rate is near the nominal 10% and where not. Does not change the planner.
 */
import { spawn } from "node:child_process";
import { cpus } from "node:os";
import { createPlanner, enumeratePlans, planKey } from "../../src/index.js";
import { normalQuantile } from "../../src/stats.js";
import { mulberry32 } from "./analyze-core.js";
import { wilson } from "./single-plan-core.js";

const CONF = (() => { const i = process.argv.indexOf("--confidence"); return i >= 0 ? Number(process.argv[i + 1]) : 0.9; })(); // 0.9 = the planner's usual setting; a Bonferroni-style variant is 1 - 0.1 / K
const SPACE = { model: ["cheap", "mid", "max"] } as const;
const PLAN_IDS = enumeratePlans(SPACE).map((p) => planKey(p));
const COSTS = [1, 2, 5];
const SLOPES = [0.4, 0.3, 0.1];
const SLA = 0.9;
const clip = (x: number) => Math.min(0.995, Math.max(0.01, x));

type Scores = "binary" | "graded";
type Feat = "random" | "informative";
type World = "below" | "at" | "power";
interface Cell { n: number; scores: Scores; feat: Feat; world: World }

function trueMeans(world: World): number[] {
  const best = world === "below" ? SLA - 0.02 : world === "at" ? SLA : SLA + 0.05;
  return [0.55, 0.8, best];
}
const prob = (means: number[], p: number, d: number) => clip(means[p] + SLOPES[p] * (0.5 - d));
/** Exact true mean of each plan (the clipped s_p integrated over a fine grid of d). */
function exactMeans(means: number[]): number[] {
  const G = 4000;
  return means.map((_, p) => Array.from({ length: G }, (_, k) => prob(means, p, (k + 0.5) / G)).reduce((a, b) => a + b, 0) / G);
}

function gaussian(rnd: () => number) {
  return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd());
}
function gamma(shape: number, rnd: () => number): number {
  if (shape < 1) return gamma(shape + 1, rnd) * Math.pow(rnd(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do {
      x = gaussian(rnd);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rnd();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}
const beta = (a: number, b: number, rnd: () => number) => {
  const x = gamma(a, rnd);
  return x / (x + gamma(b, rnd));
};

function draw(cell: Cell, means: number[], n: number, rnd: () => number) {
  const d = Array.from({ length: n }, () => rnd());
  const features = d.map((x) => (cell.feat === "informative" ? [Math.cos((Math.PI * x) / 2), Math.sin((Math.PI * x) / 2)] : [rnd() + 0.05, rnd() + 0.05]));
  const score = (p: number, x: number) => {
    const s = prob(means, p, x);
    return cell.scores === "binary" ? (rnd() < s ? 1 : 0) : beta(s * 4, (1 - s) * 4, rnd);
  };
  const f1 = PLAN_IDS.map((_, p) => d.map((x) => score(p, x)));
  return { d, features, f1 };
}

async function oneRep(cell: Cell, seed: number) {
  const rnd = mulberry32(seed);
  const means = trueMeans(cell.world);
  const nB = Math.round(0.75 * cell.n);
  const cal = draw(cell, means, cell.n, rnd);
  const fresh = draw(cell, means, nB, rnd);
  // one shared lookup: indices 0..n-1 are calibration questions, n..n+nB-1 are the fresh ones
  const all = { features: [...cal.features, ...fresh.features], f1: cal.f1.map((r, p) => [...r, ...fresh.f1[p]]) };
  const planner = createPlanner<number, number>({
    space: SPACE,
    pipeline: async (q, plan) => {
      const p = PLAN_IDS.indexOf(planKey(plan));
      return { output: all.f1[p][q], costUsd: COSTS[p] };
    },
    scorer: (_q, o) => o,
    sla: { minQuality: SLA, confidence: CONF },
    featurize: (q) => all.features[q],
    feedbackRate: 0,
    concurrency: 1,
  });
  const report = await planner.calibrate(Array.from({ length: cell.n }, (_, i) => i));
  const realized = Array.from({ length: nB }, (_, i) => {
    const q = cell.n + i;
    return all.f1[PLAN_IDS.indexOf(planKey(planner.decide(q).plan))][q];
  });
  const z = normalQuantile(CONF);
  const splitCert = wilson(realized.reduce((a, b) => a + b, 0) / nB, nB, z).lo >= SLA;
  // exact truth of the frozen policy: decide() on a grid of difficulties, true success probability of the chosen plan
  const G = 400;
  const gridFeatures = Array.from({ length: G }, (_, k) => {
    const x = (k + 0.5) / G;
    return cell.feat === "informative" ? [Math.cos((Math.PI * x) / 2), Math.sin((Math.PI * x) / 2)] : [rnd() + 0.05, rnd() + 0.05];
  });
  const base = all.features.length;
  // decide() takes the input and featurizes it through the lookup, so extend the lookup with the grid points
  all.features.push(...gridFeatures);
  let truth = 0;
  for (let k = 0; k < G; k++) truth += prob(means, PLAN_IDS.indexOf(planKey(planner.decide(base + k).plan)), (k + 0.5) / G);
  truth /= G;
  return { inSample: report.slaCertified, split: splitCert, below: truth < SLA, fallback: report.policy === "safe-plan" };
}

async function runCell(cell: Cell, reps: number, seed0: number) {
  const out = { n: reps, inSampleCert: 0, splitCert: 0, inSampleFalse: 0, splitFalse: 0, fallback: 0 };
  for (let r = 0; r < reps; r++) {
    const x = await oneRep(cell, seed0 + r);
    out.inSampleCert += x.inSample ? 1 : 0;
    out.splitCert += x.split ? 1 : 0;
    out.inSampleFalse += x.inSample && x.below ? 1 : 0;
    out.splitFalse += x.split && x.below ? 1 : 0;
    out.fallback += x.fallback ? 1 : 0;
  }
  return out;
}

const args = process.argv.slice(2);
const opt = (name: string) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? args[i + 1] : undefined;
};
const REPS = Number(opt("reps") ?? 300);

if (opt("cell")) {
  // worker: one cell, JSON on stdout
  const [n, scores, feat, world] = opt("cell")!.split(",");
  const cell: Cell = { n: Number(n), scores: scores as Scores, feat: feat as Feat, world: world as World };
  console.log(JSON.stringify({ cell, ...(await runCell(cell, REPS, 100000 + Number(n) * 7)) }));
  process.exit(0);
}

const sizes = (opt("sizes") ?? "40,100,300,1000").split(",").map(Number);
const JOBS = Number(opt("jobs") ?? Math.max(1, Math.min(10, cpus().length - 2)));
if (!(REPS >= 300)) {
  console.error("--reps must be at least 300.");
  process.exit(2);
}
const cells: Cell[] = [];
for (const world of ["below", "at", "power"] as World[])
  for (const feat of ["random", "informative"] as Feat[])
    for (const scores of ["binary", "graded"] as Scores[])
      for (const n of sizes) cells.push({ n, scores, feat, world });

const results: Array<{ cell: Cell; n: number; inSampleCert: number; splitCert: number; inSampleFalse: number; splitFalse: number; fallback: number }> = [];
let next = 0;
async function worker() {
  while (next < cells.length) {
    const c = cells[next++];
    const r = await new Promise<string>((resolve, reject) => {
      const child = spawn("npx", ["tsx", "bench/real-rag/certify-sweep.ts", "--cell", `${c.n},${c.scores},${c.feat},${c.world}`, "--reps", String(REPS), "--confidence", String(CONF)]);
      let out = "";
      child.stdout.on("data", (d) => (out += d));
      child.on("close", (code) => (code === 0 ? resolve(out) : reject(new Error(`cell ${JSON.stringify(c)} exited ${code}`))));
    });
    results.push(JSON.parse(r.trim().split("\n").pop()!));
    process.stderr.write(`\rcells done ${results.length}/${cells.length}`);
  }
}
const t0 = performance.now();
await Promise.all(Array.from({ length: JOBS }, worker));
process.stderr.write("\n");

const tm = (w: World) => exactMeans(trueMeans(w)).map((x) => x.toFixed(3)).join(" / ");
const pct = (x: number, n: number) => `${((100 * x) / n).toFixed(1).padStart(5)}%`;
console.log(`== certify sweep (SYNTHETIC, known truth): ${REPS} independent repetitions per cell, ${cells.length} cells, SLA ${SLA}, confidence ${CONF} (nominal false-certification allowance ${((1 - CONF) * 100).toFixed(2)}%), ${JOBS} parallel jobs, ${((performance.now() - t0) / 1000).toFixed(0)} s`);
console.log(`plans cheap/mid/max, exact true means ${"below"}: ${tm("below")}; at: ${tm("at")}; power: ${tm("power")} (clipped success probabilities, so slightly off the nominal values); costs 1/2/5`);
console.log("false cert = certified AND the frozen policy's exact expected quality < SLA (checked against the known truth, not a finite test mean). split = one Wilson bound on 0.75 n fresh questions.");
for (const world of ["below", "at", "power"] as World[]) {
  console.log(`\n-- world ${world.toUpperCase()}: ${world === "below" ? "best plan truly SLA - 0.02" : world === "at" ? "best plan truly = SLA" : "best plan truly SLA + 0.05"}`);
  console.log("features     scores   n     | in-sample cert | split cert | in-sample FALSE | split FALSE | always-max fallback");
  for (const feat of ["random", "informative"] as Feat[])
    for (const scores of ["binary", "graded"] as Scores[])
      for (const n of sizes) {
        const r = results.find((x) => x.cell.world === world && x.cell.feat === feat && x.cell.scores === scores && x.cell.n === n)!;
        console.log(`${feat.padEnd(12)} ${scores.padEnd(7)} ${String(n).padStart(4)}  | ${pct(r.inSampleCert, r.n)}        | ${pct(r.splitCert, r.n)}     | ${pct(r.inSampleFalse, r.n)}         | ${pct(r.splitFalse, r.n)}     | ${pct(r.fallback, r.n)}`);
      }
}
console.log("\n(SYNTHETIC. Read FALSE columns against the nominal allowance printed above; with 300 repetitions a rate has a standard error of about 1.7 points at 10% and 2.8 points at 50%. Power rows: the certification rate when the best plan really is above the SLA.)");

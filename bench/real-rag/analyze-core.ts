/**
 * Pure helpers for analyze.ts (no I/O, no network, no cache): choosing the best FIXED plan from calibration data only,
 * a deterministic paired bootstrap, the Wilson certification bound, and the verdict text. Kept separate so
 * analyze-check.ts can test them on a SYNTHETIC fixture.
 */

/** Per-plan scores and list-price costs, aligned by question index: f1[plan][question]. */
export interface Table {
  plans: string[];
  f1: number[][];
  cost: number[][];
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;
export const meanOf = mean;

export interface FixedChoice {
  fraction: number;
  /** SLA = this fraction of always-max's mean F1 on calibration, floored to 3 decimals as run.ts does. */
  sla: number;
  maxF1: number;
  plan: string;
  calF1: number;
  calCost: number;
}

/**
 * The cheapest plan whose calibration F1 meets the SLA (ties on cost: higher F1). Uses the calibration table ONLY:
 * the function takes no test data, so the choice cannot depend on it.
 */
export function chooseFixed(cal: Table, maxPlan: string, fraction: number): FixedChoice {
  const mi = cal.plans.indexOf(maxPlan);
  if (mi < 0) throw new Error(`always-max plan ${maxPlan} is not in the table`);
  const maxF1 = mean(cal.f1[mi]);
  const sla = Math.floor(fraction * maxF1 * 1000) / 1000;
  let best: FixedChoice | undefined;
  for (let p = 0; p < cal.plans.length; p++) {
    const calF1 = mean(cal.f1[p]);
    const calCost = mean(cal.cost[p]);
    if (calF1 < sla) continue;
    const better =
      !best || calCost < best.calCost - 1e-12 || (Math.abs(calCost - best.calCost) <= 1e-12 && calF1 > best.calF1);
    if (better) best = { fraction, sla, maxF1, plan: cal.plans[p], calF1, calCost };
  }
  // always-max meets any SLA of at most 100% of its own F1, so a choice always exists
  return best!;
}

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface Interval {
  mean: number;
  lo: number;
  hi: number;
}

/**
 * Paired bootstrap over questions: the 95% percentile interval of mean(a - b) for F1 and for cost. Same resamples for
 * both quantities. Deterministic for a given seed.
 */
export function pairedBootstrap(
  a: { f1: number[]; cost: number[] },
  b: { f1: number[]; cost: number[] },
  resamples = 10000,
  seed = 12345,
): { f1: Interval; cost: Interval; n: number } {
  const n = a.f1.length;
  if (n === 0 || b.f1.length !== n) throw new Error("paired bootstrap needs two equal, non-empty samples");
  const df = a.f1.map((x, i) => x - b.f1[i]);
  const dc = a.cost.map((x, i) => x - b.cost[i]);
  const rnd = mulberry32(seed);
  const mf: number[] = [];
  const mc: number[] = [];
  for (let r = 0; r < resamples; r++) {
    let sf = 0;
    let sc = 0;
    for (let k = 0; k < n; k++) {
      const i = Math.floor(rnd() * n);
      sf += df[i];
      sc += dc[i];
    }
    mf.push(sf / n);
    mc.push(sc / n);
  }
  const ci = (xs: number[], point: number): Interval => {
    const s = [...xs].sort((x, y) => x - y);
    return { mean: point, lo: s[Math.floor(0.025 * s.length)], hi: s[Math.min(s.length - 1, Math.floor(0.975 * s.length))] };
  };
  return { f1: ci(mf, mean(df)), cost: ci(mc, mean(dc)), n };
}

/** The planner's Wilson lower bound (src/planner.ts), applied to a graded mean in [0, 1] like a pass rate. */
export function wilsonLower(m: number, n: number, z: number): number {
  if (n === 0) return 0;
  const z2 = z * z;
  const centre = m + z2 / (2 * n);
  const margin = z * Math.sqrt((m * (1 - m)) / n + z2 / (4 * n * n));
  return (centre - margin) / (1 + z2 / n);
}

/** One-sided z for confidence 0.9, as in docs/quickstart.md. */
export const Z_90 = 1.2816;

/** Smallest n at which a plan whose observed mean equals `m` clears the SLA at confidence 0.9; null if never. */
export function questionsNeeded(m: number, sla: number, z = Z_90, max = 100000): number | null {
  if (m <= sla) return null;
  for (let n = 2; n <= max; n++) if (wilsonLower(m, n, z) >= sla) return n;
  return null;
}

export interface Side {
  name: string;
  f1: number;
  cost: number;
  meetsSla: boolean;
}

/**
 * Plain verdict for routing vs the fixed plan. The comparison of cost is only a "win" when both sides have the same
 * SLA outcome and the paired cost interval excludes zero.
 */
export function verdict(
  routing: Side,
  fixed: Side,
  diff: { f1: Interval; cost: Interval; n: number },
  ctx: { sla: number; calibration: number },
): string {
  const head = `${routing.name} vs the best fixed plan (${fixed.name}), SLA ${ctx.sla}, ${ctx.calibration} calibration questions, ${diff.n} test questions, one dataset, one provider, one seed:`;
  const outcome = (s: Side) => (s.meetsSla ? "meets" : "misses");
  const c = diff.cost;
  const f = diff.f1;
  const pct = (x: number, base: number) => `${Math.abs((x / base) * 100).toFixed(0)}%`;
  const costWords =
    c.hi < 0
      ? `routing is cheaper by $${(-c.mean).toFixed(5)} per query (${pct(c.mean, fixed.cost)}), 95% interval [${c.lo.toFixed(5)}, ${c.hi.toFixed(5)}]`
      : c.lo > 0
        ? `routing is dearer by $${c.mean.toFixed(5)} per query (${pct(c.mean, fixed.cost)}), 95% interval [${c.lo.toFixed(5)}, ${c.hi.toFixed(5)}]`
        : `cost difference $${c.mean.toFixed(5)} per query, 95% interval [${c.lo.toFixed(5)}, ${c.hi.toFixed(5)}] includes 0`;
  const f1Words = `F1 difference ${f.mean >= 0 ? "+" : ""}${f.mean.toFixed(3)}, 95% interval [${f.lo.toFixed(3)}, ${f.hi.toFixed(3)}]`;
  const f1Note =
    f.hi < 0
      ? " WARNING: routing's F1 is lower, with the 95% interval excluding 0."
      : f.lo > 0
        ? " Note: routing's F1 is higher, with the 95% interval excluding 0."
        : "";
  const caveat =
    " SLA outcomes use the test F1 point estimate, not an interval; the fixed plan was picked from one noisy calibration sample, so its test figures, not its calibration ones, are the honest ones.";
  if (!routing.meetsSla && !fixed.meetsSla) {
    return `${head} neither meets the test SLA, so no winner is claimed. ${costWords}; ${f1Words}.${f1Note}${caveat}`;
  }
  if (routing.meetsSla !== fixed.meetsSla) {
    const who = routing.meetsSla ? routing : fixed;
    return `${head} not comparable at an equal SLA outcome: ${who.name} ${outcome(who)} the test SLA and the other misses it. ${costWords}; ${f1Words}.${caveat}`;
  }
  if (c.hi < 0) return `${head} ROUTING WINS on cost at an equal SLA outcome (both ${outcome(routing)}): ${costWords}; ${f1Words}.${f1Note}${caveat}`;
  if (c.lo > 0) return `${head} THE FIXED PLAN WINS on cost at an equal SLA outcome (both ${outcome(routing)}): ${costWords}; ${f1Words}. Routing does not beat the best fixed plan here.${f1Note}${caveat}`;
  return `${head} NO WINNER: the cost intervals overlap (both ${outcome(routing)} the SLA). ${costWords}; ${f1Words}. Routing is not shown to beat the best fixed plan.${f1Note}${caveat}`;
}

/**
 * Plan-table coverage, offline ($0, no network). Counts which (question, plan) pairs are in the response cache
 * for the calibration and test sets, per-plan F1 and list-price cost on the cached pairs, and how many calls
 * a live run would still need per model.
 *
 *   npm run bench:rag:table [-- --calibration 100 --test 200 --seed 7] [--plan-fill]
 *
 * --plan-fill also lists the missing (question, plan) pairs. It only prints; it never calls the API.
 * The question sets are sampled exactly as run.ts / replay.ts do, so the numbers refer to the same questions.
 */
process.env.PATHWISE_BENCH_OFFLINE = "1"; // before the imports below read it: a cache miss throws, nothing is requested
const { enumeratePlans, planKey } = await import("../../src/index.js");
const { SPACE, MODELS } = await import("./config.js");
const { loadHotpot, sample } = await import("./dataset.js");
const { OfflineMiss } = await import("./gemini.js");
const { ragPipeline, scorer } = await import("./pipeline.js");

const args = process.argv.slice(2);
const num = (name: string, def: number) => {
  const i = args.indexOf(`--${name}`);
  return i >= 0 ? Number(args[i + 1]) : def;
};
const FILL = args.includes("--plan-fill");
const { calibration, test } = sample(await loadHotpot(), num("calibration", 100), num("test", 200), num("seed", 7));
const plans = enumeratePlans(SPACE);

for (const [name, questions] of [["calibration", calibration], ["test", test]] as const) {
  const stat = new Map<string, { n: number; f1: number; cost: number }>();
  const missing: { id: string; plan: string; model: string }[] = [];
  for (const plan of plans) {
    const s = { n: 0, f1: 0, cost: 0 };
    stat.set(planKey(plan), s);
    for (const q of questions) {
      try {
        const r = await ragPipeline(q, plan);
        s.n++;
        s.f1 += scorer(q, r.output);
        s.cost += r.costUsd;
      } catch (err) {
        if (!(err instanceof OfflineMiss)) throw err;
        missing.push({ id: q.id, plan: planKey(plan), model: String(plan.model) });
      }
    }
  }
  const total = plans.length * questions.length;
  console.log(`\n== ${name}: ${questions.length} questions x ${plans.length} plans, ${total - missing.length}/${total} cached, ${missing.length} missing`);
  console.log("(F1 and cost are over each plan's cached questions only, so plans with different coverage are NOT comparable)");
  console.log("plan                                   cached  mean-F1  mean-cost($ list)");
  for (const [k, s] of stat) {
    console.log(`${k.padEnd(38)} ${`${s.n}/${questions.length}`.padStart(7)}  ${s.n ? (s.f1 / s.n).toFixed(3) : "  -  "}   ${s.n ? "$" + (s.cost / s.n).toFixed(5) : "-"}`);
  }
  const perModel = new Map<string, number>();
  for (const m of missing) perModel.set(m.model, (perModel.get(m.model) ?? 0) + 1);
  console.log("missing calls per model:", [...perModel].map(([m, n]) => `${m} (${MODELS[m as keyof typeof MODELS]}) ${n}`).join(", ") || "none");
  if (FILL) for (const m of missing) console.log(`  ${m.id}  ${m.plan}`);
}
console.log("\n(offline: no API calls were made)");

/**
 * Real round trip of pathwise/postgres against YOUR Postgres (e.g. a Railway database). Run by the user, never by the
 * terminal builder: it needs your connection string, which this repo's tooling never sees.
 *
 *   npm i --no-save pg                      # temporary, NOT added to package.json (the library stays zero-dependency)
 *   DATABASE_URL="postgresql://..." npm run smoke:postgres
 *
 * It creates a temp table named pathwise_smoke_<random> (its own, never an existing table), saves a small synthetic state,
 * upserts it, loads it back, checks it, prints the timings (YOUR network, not a claim) and drops that table. It never prints
 * the URL: errors are scrubbed of it. Without DATABASE_URL or without the `pg` package it prints SKIP and exits 0.
 * Railway: use the public/proxy URL if you run this from your Mac (the private network URL only works inside Railway);
 * add `?sslmode=require` if the server asks for TLS.
 */
import { createPlanner } from "../../src/index.js";
import { postgresStore, type PgLikeClient } from "../../src/adapters/postgres.js";

const url = process.env.DATABASE_URL;
if (!url) {
  console.log("SKIP: DATABASE_URL is not set. Set it to your Postgres connection string and run again (nothing was contacted).");
  process.exit(0);
}
let pgModule: { default?: { Client: new (o: { connectionString: string }) => unknown }; Client?: new (o: { connectionString: string }) => unknown };
try {
  const name = "pg"; // a variable, so the type checker does not need the package
  pgModule = await import(name);
} catch {
  console.log("SKIP: the `pg` package is not installed. Run `npm i --no-save pg` (it is not added to package.json) and run again.");
  process.exit(0);
}
const Client = (pgModule.Client ?? pgModule.default?.Client) as new (o: { connectionString: string }) => PgLikeClient & { connect(): Promise<void>; end(): Promise<void> };
const scrub = (e: unknown) => String(e instanceof Error ? e.message : e).split(url).join("[url]").replace(/postgres(?:ql)?:\/\/\S+/gi, "[url]").slice(0, 300);

const table = `pathwise_smoke_${Math.random().toString(36).slice(2, 10)}`;
const rows: Array<[string, string, string]> = [];
const client = new Client({ connectionString: url });
const planner = () => createPlanner({ space: { model: ["a", "b"] }, pipeline: async () => ({ output: 1, costUsd: 0.001 }), scorer: () => 1, sla: { minQuality: 0.5 } });
let failed = false;
try {
  await client.connect();
  const store = postgresStore({ client, table });
  const t = (label: string, run: () => Promise<unknown>) => async () => {
    const t0 = performance.now();
    await run();
    rows.push([label, "ok", `${(performance.now() - t0).toFixed(0)} ms`]);
  };
  const p = planner();
  await p.calibrate(["one", "two", "three"]);
  const saved = p.exportState();
  await t("init (create temp table)", () => store.init())();
  await t("save (insert)", () => store.save("smoke", p))();
  await t("save again (upsert)", () => store.save("smoke", p))();
  const q = planner();
  let n: number | undefined;
  await t("load", async () => (n = await store.load("smoke", q)))();
  const same = n === saved.length && JSON.stringify(q.exportState()) === JSON.stringify(saved);
  rows.push(["loaded state equals saved state", same ? "ok" : "FAIL", `${saved.length} records`]);
  if (!same) failed = true;
  const missing = await store.load("does-not-exist", planner());
  rows.push(["missing key returns undefined", missing === undefined ? "ok" : "FAIL", ""]);
  if (missing !== undefined) failed = true;
} catch (e) {
  console.log(`FAILED: ${scrub(e)}`);
  failed = true;
} finally {
  try {
    await client.query(`DROP TABLE IF EXISTS "${table}"`);
    rows.push(["temp table dropped", "ok", table]);
  } catch (e) {
    rows.push(["temp table dropped", "FAIL", `${table}: ${scrub(e)} (drop it by hand)`]);
    failed = true;
  }
  try {
    await client.end();
  } catch {
    /* ignore */
  }
}
console.log("\ncheck".padEnd(36) + "result".padEnd(8) + "detail");
for (const [a, b, c] of rows) console.log(a.padEnd(36) + b.padEnd(8) + c);
console.log("\nTimings are your network and your database, measured once; they are not a benchmark.");
process.exit(failed ? 1 : 0);

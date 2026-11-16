/**
 * Checks for pathwise/postgres (SYNTHETIC: an in-memory fake client that understands exactly the statements the adapter
 * issues; no database, no network). The real round trip against Postgres is `npm run smoke:postgres`, run by the user.
 *   npm run bench:postgres
 */
import { createPlanner, hashedTextFeaturizer, planKey } from "../src/index.js";
import { postgresStore, type PgLikeClient } from "../src/adapters/postgres.js";
import { createWorkload, ragPipeline, SPACE } from "./workload.js";

let failures = 0;
const check = (ok: boolean, msg: string) => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}`);
  if (!ok) failures++;
};

/** Postgres jsonb sorts object keys by length, then bytewise: mimic it so a key-order dependence cannot hide. */
const jsonbLike = (v: unknown): unknown =>
  Array.isArray(v)
    ? v.map(jsonbLike)
    : v && typeof v === "object"
      ? Object.fromEntries(Object.entries(v).sort(([a], [b]) => a.length - b.length || (a < b ? -1 : a > b ? 1 : 0)).map(([k, x]) => [k, jsonbLike(x)]))
      : v;
const canon = (v: unknown): string => JSON.stringify(v, (_k, x) => (x && typeof x === "object" && !Array.isArray(x) ? Object.fromEntries(Object.entries(x).sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))) : x));

/** Fake pg client: tables as Maps; recognises only the adapter's four statement shapes and records every call. */
function fakeClient() {
  const tables = new Map<string, Map<string, unknown>>();
  const calls: Array<{ text: string; params: unknown[] }> = [];
  let failWith: Error | undefined;
  const client: PgLikeClient = {
    async query(text, params = []) {
      calls.push({ text, params });
      if (failWith) throw failWith;
      let m = /^CREATE TABLE IF NOT EXISTS "([A-Za-z_][A-Za-z0-9_]*)" \(key text PRIMARY KEY, state jsonb NOT NULL/.exec(text);
      if (m) {
        if (!tables.has(m[1])) tables.set(m[1], new Map());
        return { rows: [] };
      }
      m = /^INSERT INTO "([A-Za-z_][A-Za-z0-9_]*)" \(key, state\) VALUES \(\$1, \$2::jsonb\) ON CONFLICT \(key\) DO UPDATE SET state = EXCLUDED\.state, updated_at = now\(\)$/.exec(text);
      if (m) {
        tables.get(m[1])!.set(params[0] as string, jsonbLike(JSON.parse(params[1] as string))); // jsonb: stored parsed, keys reordered
        return { rows: [] };
      }
      m = /^SELECT state FROM "([A-Za-z_][A-Za-z0-9_]*)" WHERE key = \$1$/.exec(text);
      if (m) {
        const v = tables.get(m[1])!.get(params[0] as string);
        return { rows: v === undefined ? [] : [{ state: v }] };
      }
      throw new Error(`fake client: unexpected statement: ${text}`);
    },
  };
  return { client, tables, calls, fail: (e: Error | undefined) => (failWith = e) };
}

const feats = hashedTextFeaturizer();
type Q = { text: string; difficulty: number };
const mk = () =>
  createPlanner<Q, { correct: boolean }>({ space: SPACE, pipeline: ragPipeline, scorer: (_q, o) => (o.correct ? 1 : 0), sla: { minQuality: 0.8 }, featurize: (q) => feats(q.text), feedbackRate: 0, concurrency: 4 });

const { makeQueries } = createWorkload(11, {});
const samples = makeQueries(80);
const a = mk();
await a.calibrate(samples);

// round trip
{
  const f = fakeClient();
  const store = postgresStore({ client: f.client, table: "my_state" });
  await store.init();
  await store.save("bot-1", a);
  const b = mk();
  const n = await store.load("bot-1", b);
  const probe = makeQueries(40);
  check(n === a.exportState().length && canon(b.exportState()) === canon(a.exportState()), "round trip: the loaded planner has exactly the saved records");
  check(JSON.stringify(probe.map((q) => planKey(a.decide(q).plan))) === JSON.stringify(probe.map((q) => planKey(b.decide(q).plan))), "round trip: the loaded planner makes the same decisions");
  // upsert
  await store.save("bot-1", mk()); // empty state replaces
  check((await store.load("bot-1", mk())) === 0, "upsert: saving again under the same key replaces the state");
  check(await store.load("nope", mk()) === undefined, "a missing key returns undefined, nothing imported");
  check(f.calls.every((c) => !c.text.includes("bot-1") && !c.text.includes("features")), "the key and the state are bound parameters, never in the SQL text");
}
// injection attempts
{
  const f = fakeClient();
  for (const bad of ['x"; DROP TABLE y;--', "a b", "1abc", "", "a-b", "t".repeat(64), "x'"]) {
    let threw = false;
    try { postgresStore({ client: f.client, table: bad }); } catch { threw = true; }
    check(threw, `table name ${JSON.stringify(bad.slice(0, 20))} is rejected`);
  }
  const store = postgresStore({ client: f.client });
  await store.init();
  const evil = `'; DROP TABLE "pathwise_state"; --`;
  await store.save(evil, a);
  const b = mk();
  check((await store.load(evil, b)) === a.exportState().length, "a key with SQL in it round-trips as data");
  check(f.calls.every((c) => !c.text.includes("DROP")) && f.tables.has("pathwise_state"), "no SQL from the key ever reaches the statement text");
  let e1 = "";
  try { await store.save("", a); } catch (e) { e1 = (e as Error).message; }
  let e2 = "";
  try { await store.save("k".repeat(513), a); } catch (e) { e2 = (e as Error).message; }
  check(/non-empty string/.test(e1) && /at most 512/.test(e2), "an empty or oversized key is rejected before any query");
}
// errors never carry a connection string
{
  const f = fakeClient();
  const store = postgresStore({ client: f.client });
  f.fail(new Error("connect ECONNREFUSED postgresql://user:s3cretpass@db.example.com:5432/app?sslmode=require"));
  let msg = "";
  try { await store.init(); } catch (e) { msg = (e as Error).message; }
  check(/init failed/.test(msg) && !/s3cretpass|user:|db\.example\.com/.test(msg) && /\[redacted-url\]/.test(msg), "a client error with a postgres:// URL is rewrapped and the URL is removed");
  f.fail(new Error("password authentication failed; tried alice:hunter2secret@10.0.0.5:5432 and 'postgresql://bob:pw99@h/db'"));
  let m2 = "";
  try { await store.init(); } catch (e) { m2 = (e as Error).message; }
  check(!/hunter2secret|alice:|pw99|bob:/.test(m2), "a scheme-less user:password@host and a quoted URL are removed too");
  f.fail(new Error("x".repeat(200_000)));
  const t0 = performance.now();
  let m3 = "";
  try { await store.init(); } catch (e) { m3 = (e as Error).message; }
  const ms = performance.now() - t0;
  console.log(`     (200,000-character error message without ':' scrubbed in ${ms.toFixed(1)} ms, this machine)`);
  check(ms < 100 && m3.length < 400, "a 200,000-character error message is scrubbed in under 100 ms and truncated");
  f.fail(undefined);
}
// corrupt stored state
{
  const f = fakeClient();
  const store = postgresStore({ client: f.client });
  await store.init();
  f.tables.get("pathwise_state")!.set("bad", { not: "a list" });
  f.tables.get("pathwise_state")!.set("bad2", [{ features: [1] }]);
  const target = mk();
  let m1 = "";
  let m2 = "";
  try { await store.load("bad", target); } catch (e) { m1 = (e as Error).message; }
  try { await store.load("bad2", target); } catch (e) { m2 = (e as Error).message; }
  check(/not a list of planner records/.test(m1) && /not a list of planner records/.test(m2) && target.exportState().length === 0, "corrupt stored state is rejected and nothing is imported");
}
// large state + timing (SYNTHETIC: in-memory fake, not a network)
{
  const f = fakeClient();
  const store = postgresStore({ client: f.client });
  await store.init();
  const big = {
    exportState: () => Array.from({ length: 5000 }, (_, i) => ({ features: Array.from({ length: 64 }, (_, k) => Math.sin(i + k)), observations: [{ planKey: "m|k", quality: 1, costUsd: 0.001, latencyMs: 5 }] })),
  };
  const t0 = performance.now();
  await store.save("big", big as never);
  const t1 = performance.now();
  let got = 0;
  await store.load("big", { importState: (r) => (got = r.length) });
  const t2 = performance.now();
  console.log(`     (SYNTHETIC fake client: 5000 records x 64 features, save ${(t1 - t0).toFixed(0)} ms, load ${(t2 - t1).toFixed(0)} ms; no network, not a database timing)`);
  check(got === 5000, "a large state (5000 records) round-trips");
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC fake client)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);

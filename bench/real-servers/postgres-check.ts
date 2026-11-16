/**
 * Checks for the smoke:postgres script's SKIP paths (no database, no network, no URL). The real round trip is run by the user.
 *   npm run smoke:postgres:check
 */
import { spawnSync } from "node:child_process";

let failures = 0;
const check = (ok: boolean, msg: string, detail = "") => {
  console.log(`${ok ? "ok  " : "FAIL"} ${msg}${!ok && detail ? `\n     ${detail}` : ""}`);
  if (!ok) failures++;
};
const run = (env: Record<string, string>) => {
  const e: NodeJS.ProcessEnv = { ...process.env, ...env };
  if (!("DATABASE_URL" in env)) delete e.DATABASE_URL;
  const r = spawnSync("npx", ["tsx", "bench/real-servers/postgres.ts"], { encoding: "utf8", env: e });
  return { code: r.status, out: `${r.stdout}${r.stderr}` };
};
const noUrl = run({});
check(noUrl.code === 0 && /SKIP: DATABASE_URL is not set/.test(noUrl.out), "without DATABASE_URL: prints SKIP and exits 0", noUrl.out.slice(-300));
const noPg = run({ DATABASE_URL: "postgresql://user:pw-should-never-print@127.0.0.1:1/none" });
// `pg` is not a dependency of this repo, so unless someone installed it with --no-save this is the SKIP path; if it IS installed, the connection to port 1 fails and must not print the URL.
check(!/pw-should-never-print/.test(noPg.out), "the connection string never appears in the output", noPg.out.slice(-300));
check((noPg.code === 0 && /SKIP: the `pg` package is not installed/.test(noPg.out)) || (noPg.code === 1 && /FAILED/.test(noPg.out)), "without pg: SKIP with the install hint (or, with pg installed, a scrubbed failure)", noPg.out.slice(-300));
console.log(failures === 0 ? "\nall checks passed (no database contacted)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);

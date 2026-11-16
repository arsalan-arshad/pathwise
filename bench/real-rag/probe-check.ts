/**
 * Checks for probe.ts, musique.ts and fill.ts --phase sample. Everything here is SYNTHETIC: generated numbers, a tiny made-up
 * MuSiQue-shaped file (the real field names are NOT verified, see musique.ts), mock model answers, temp cache, no network, no key.
 *   npm run bench:rag:probe:check
 */
import { spawnSync } from "node:child_process";
import { existsSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { parseMusique } from "./musique.js";
import { auc, groupScores, knnScores, neighbours, permutationBand } from "./probe-core.js";

let failures = 0;
function check(label: string, ok: boolean, detail = "") {
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}${!ok && detail ? `\n${detail.split("\n").map((l) => "     " + l).join("\n")}` : ""}`);
}

// ---- pure helpers
check("auc: perfect, reversed, tie and one-class cases", auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) === 1 && auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) === 0 && auc([0.5, 0.5], [0, 1]) === 0.5 && auc([0.1, 0.2], [1, 1]) === null);
const grp = ["a", "a", "a", "a", "b", "b", "b", "b"];
const lab = [1, 1, 0, 0, 1, 0, 0, 0];
check("group score (in-sample rate) is the group's label rate", JSON.stringify(groupScores(grp, lab)) === JSON.stringify([0.5, 0.5, 0.5, 0.5, 0.25, 0.25, 0.25, 0.25]));
const noInfo = Array.from({ length: 60 }, (_, i) => (i % 2 === 0 ? "x" : "y"));
const noInfoLabels = Array.from({ length: 60 }, (_, i) => (i % 4 < 2 ? 1 : 0)); // independent of the group
check("leave-one-out on a two-valued feature falls far below 0.5 with no information (the artifact the probe avoids)", (auc(groupScores(noInfo, noInfoLabels, true), noInfoLabels) ?? 1) < 0.3 && Math.abs((auc(groupScores(noInfo, noInfoLabels), noInfoLabels) ?? 0) - 0.5) < 0.1);
// kNN: two clusters, labels follow the cluster -> AUC near 1; labels random -> inside the shuffled band
const vecs = Array.from({ length: 40 }, (_, i) => (i < 20 ? [1, 0.05 * (i % 5)] : [0.05 * (i % 5), 1]));
const clusterLabels = vecs.map((_, i) => (i < 20 ? 1 : 0));
const nb = neighbours(vecs, 5);
check("kNN AUC is ~1 when the label follows the clusters", (auc(knnScores(nb, clusterLabels, false), clusterLabels) ?? 0) > 0.95);
const band = permutationBand((l) => knnScores(nb, l, false), clusterLabels);
check("the shuffled-label band sits around 0.5 and is deterministic", !!band && band.lo < 0.5 && band.hi > 0.4 && JSON.stringify(band) === JSON.stringify(permutationBand((l) => knnScores(nb, l, false), clusterLabels)));

// ---- MuSiQue parser on a tiny SYNTHETIC fixture
const row = (id: string, extra: Record<string, unknown> = {}) =>
  JSON.stringify({ id, question: `q ${id}?`, answer: `a ${id}`, answerable: true, paragraphs: [{ idx: 0, title: "T0", paragraph_text: " text zero ", is_supporting: true }, { idx: 1, title: "T1", paragraph_text: "text one", is_supporting: false }], ...extra });
const parsed = parseMusique([row("2hop__1_2"), row("3hop1__3_4_5"), row("4hop3__6", { answerable: false }), "", row("4hop2__7_8")].join("\n"));
check("parser: keeps answerable rows, drops answerable=false, skips blank lines", parsed.length === 3 && !parsed.some((q) => q.id.startsWith("4hop3")));
check("parser: type is the hop count from the id prefix", parsed.map((q) => q.type).join() === "2hop,3hop,4hop");
check("parser: passages from paragraphs (trimmed), goldTitles from is_supporting", parsed[0].passages.length === 2 && parsed[0].passages[0].text === "text zero" && parsed[0].goldTitles.join() === "T0");
const throws = (t: string) => { try { parseMusique(t); return ""; } catch (e) { return (e as Error).message; } };
check("parser: a missing field names the field and the line", /line 1: expected a non-empty string field "question"/.test(throws(JSON.stringify({ id: "2hop__1", answer: "a", paragraphs: [{ title: "t", paragraph_text: "p" }] }))));
check("parser: a bad id and bad JSON fail loudly", /does not start with/.test(throws(row("weird_id"))) && /not valid JSON/.test(throws("{nope")));

// ---- end to end on mock answers (needs the HotpotQA dataset)
if (!existsSync(new URL("./data/hotpot_dev_distractor_v1.json", import.meta.url))) {
  console.log("skipped end-to-end checks: dataset not downloaded.");
} else {
  const dir = mkdtempSync(join(tmpdir(), "pathwise-probe-"));
  const mq = join(dir, "musique-fixture.jsonl");
  writeFileSync(mq, Array.from({ length: 40 }, (_, i) => row(`${2 + (i % 3)}hop__${i}`, { question: `synthetic question number ${i} about topic ${i * 7}?` })).join("\n"));
  const env = { ...process.env, PATHWISE_BENCH_MOCK: "1", PATHWISE_BENCH_CACHE_DIR: dir, GEMINI_API_KEY: "", PATHWISE_MUSIQUE_FILE: mq };
  const run = (script: string, args: string[], e: NodeJS.ProcessEnv = env) => {
    const r = spawnSync("npx", ["tsx", `bench/real-rag/${script}`, ...args], { encoding: "utf8", env: e });
    return { code: r.status, out: `${r.stdout}${r.stderr}` };
  };
  const cheap = "model=small|thinking=low|topK=2";
  const max = "model=medium|thinking=high|topK=10";
  try {
    const empty = run("probe.ts", ["--n", "30"]);
    check("probe on an empty cache: says incomplete, measures nothing, exit 0", empty.code === 0 && /incomplete: 60 .*pairs missing/.test(empty.out) && !/AUC/.test(empty.out), empty.out.slice(-300));
    const dry = run("fill.ts", ["--phase", "sample", "--plans", `${cheap},${max}`, "--n", "30"]);
    check("fill --phase sample is a dry run without --yes-live", dry.code === 0 && /dry run: no call was made/.test(dry.out) && /60 missing/.test(dry.out), dry.out.slice(-400));
    check("fill refuses --phase sample without --plans or --n, more than 4 plans, and musique with another phase", [
      run("fill.ts", ["--phase", "sample", "--n", "10"]),
      run("fill.ts", ["--phase", "sample", "--plans", cheap]),
      run("fill.ts", ["--phase", "sample", "--n", "10", "--plans", "a,b,c,d,e"]),
      run("fill.ts", ["--phase", "calibration", "--dataset", "musique"]),
    ].every((r) => r.code === 2 && /refused/.test(r.out)));
    check("fill: hotpot --phase sample beyond the calibration size is refused (it would reach the test questions)", run("fill.ts", ["--phase", "sample", "--n", "150", "--plans", cheap]).code === 2);
    check("fill refuses the paid models and --yes-live with OFFLINE", run("fill.ts", ["--phase", "sample", "--n", "5", "--plans", cheap, "--yes-live"], { ...env, PATHWISE_BENCH_OFFLINE: "1" }).code === 2 && run("fill.ts", ["--phase", "sample", "--n", "5", "--plans", cheap], { ...env, PATHWISE_BENCH_MODELS: "paid" }).code === 2);
    const nofile = run("fill.ts", ["--phase", "sample", "--dataset", "musique", "--n", "5", "--plans", cheap], { ...env, PATHWISE_MUSIQUE_FILE: join(dir, "missing.jsonl") });
    check("musique file missing: refuses, says it never downloads", nofile.code === 2 && /never downloads/.test(nofile.out), nofile.out.slice(-300));
    run("fill.ts", ["--phase", "sample", "--plans", `${cheap},${max}`, "--n", "30", "--yes-live"]);
    const p = run("probe.ts", ["--n", "30"]);
    check("probe after a mock fill: prints base rate, hashed kNN AUC with the shuffled band, type AUC, labelled SYNTHETIC", p.code === 0 && /SYNTHETIC/.test(p.out) && /base rate/.test(p.out) && /hashed text, leave-one-out kNN/.test(p.out) && /shuffled labels: 5th to 95th/.test(p.out) && /question type .*group rate/.test(p.out), p.out.slice(-900));
    const p2 = run("probe.ts", ["--n", "30"]);
    check("probe output is deterministic", p.out === p2.out);
    run("fill.ts", ["--phase", "sample", "--dataset", "musique", "--plans", `${cheap},${max}`, "--n", "30", "--yes-live"]);
    const pm = run("probe.ts", ["--dataset", "musique", "--n", "30"]);
    const bd = run("probe.ts", ["--dataset", "musique", "--n", "30", "--breakdown"]);
    check("--breakdown prints means, the tie breakdown, a per-type table and the tie-free label (no AUC on a tiny minority class)", bd.code === 0 && /mean F1 \/ list-price cost per query/.test(bd.out) && /ties \d+ \(both exactly 0: \d+; both equal and above 0: \d+\)/.test(bd.out) && /per type: n \|/.test(bd.out) && /tie-free label/.test(bd.out), bd.out.slice(-900));
    check("musique fixture end to end: probe works, types are hop counts", pm.code === 0 && /dataset musique/.test(pm.out) && /2hop|3hop|4hop/.test(pm.out) && /AUC/.test(pm.out), pm.out.slice(-700));
    check("--embed is refused under mock mode", run("probe.ts", ["--n", "30", "--embed", "ollama"]).code === 2);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
}
console.log(failures === 0 ? "\nall checks passed (SYNTHETIC)" : `\n${failures} check(s) FAILED`);
process.exit(failures === 0 ? 0 : 1);

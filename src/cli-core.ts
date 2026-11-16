/**
 * Zero-key command line over a log of MEASURED runs: `pathwise recommend` and `pathwise certify`. Zero dependencies
 * (node:fs, node:readline, node:util). No network, no environment variable, no model call: it only reads the file you name.
 * The statistics are the library's own (`recommendPlan` in recommend.ts and `certifyPolicy` in certify.ts), not a copy.
 */
import { createReadStream, statSync } from "node:fs";
import { createInterface } from "node:readline";
import { parseArgs } from "node:util";
import { certifyPolicy } from "./certify.js";
import { recommendPlan } from "./recommend.js";
import { planKey } from "./space.js";
import type { Plan } from "./types.js";

export const MAX_LOG_BYTES = 200 * 1024 * 1024;
export class InputError extends Error {}

export interface Row {
  question: string;
  plan: string;
  planObject?: Plan;
  quality: number;
  costUsd: number;
  line: number;
}

/** Parses one JSONL log line by line. Every error names the line number. Never evaluates the content. */
export async function readLog(path: string): Promise<Row[]> {
  let size: number;
  try {
    const st = statSync(path);
    if (!st.isFile()) throw new Error("not a regular file");
    size = st.size;
  } catch {
    throw new InputError(`cannot read the log file "${path}" (it must be a regular file)`);
  }
  if (size > MAX_LOG_BYTES) throw new InputError(`the log file is ${(size / 1048576).toFixed(0)} MB; the limit is ${MAX_LOG_BYTES / 1048576} MB (this tool reads at most that much)`);
  const rows: Row[] = [];
  const stream = createReadStream(path, { encoding: "utf8" });
  const streamError = new Promise<never>((_, reject) => stream.on("error", (e) => reject(new InputError(`cannot read the log file "${path}": ${(e as Error).message}`))));
  streamError.catch(() => undefined); // handled below; avoids an unhandled rejection when the loop finishes first
  const rl = createInterface({ input: stream, crlfDelay: Infinity });
  let line = 0;
  const lines = rl[Symbol.asyncIterator]();
  for (;;) {
    const next = await Promise.race([lines.next(), streamError]);
    if (next.done) break;
    const raw = next.value;
    line++;
    const text = raw.trim();
    if (!text) continue;
    let obj: unknown;
    try {
      obj = JSON.parse(text);
    } catch {
      throw new InputError(`line ${line}: not valid JSON`);
    }
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) throw new InputError(`line ${line}: expected a JSON object`);
    const o = obj as Record<string, unknown>;
    const q = o.question;
    if ((typeof q !== "string" && typeof q !== "number") || String(q).length === 0) throw new InputError(`line ${line}: "question" must be a non-empty string or number`);
    let plan: string;
    let planObject: Plan | undefined;
    if (typeof o.plan === "string" && o.plan.length > 0) plan = o.plan;
    else if (o.plan && typeof o.plan === "object" && !Array.isArray(o.plan) && Object.keys(o.plan).length > 0) {
      for (const v of Object.values(o.plan as Record<string, unknown>)) {
        if (typeof v !== "string" && typeof v !== "number" && typeof v !== "boolean") throw new InputError(`line ${line}: plan knob values must be strings, numbers or booleans`);
      }
      planObject = o.plan as Plan;
      plan = planKey(planObject);
    } else throw new InputError(`line ${line}: "plan" must be a plan key string or an object of knob values`);
    const quality = o.quality;
    if (typeof quality !== "number" || !Number.isFinite(quality) || quality < 0 || quality > 1) throw new InputError(`line ${line}: "quality" must be a number in [0, 1]`);
    const costUsd = o.costUsd;
    if (typeof costUsd !== "number" || !Number.isFinite(costUsd) || costUsd < 0) throw new InputError(`line ${line}: "costUsd" must be a finite number >= 0`);
    rows.push({ question: String(q), plan, planObject, quality, costUsd, line });
  }
  if (rows.length === 0) throw new InputError("the log has no rows");
  return rows;
}

function table(rows: Row[]) {
  const plans = new Map<string, { object?: Plan }>();
  const byQuestion = new Map<string, Map<string, { quality: number; costUsd: number }>>();
  const seen = new Map<string, number>();
  for (const r of rows) {
    const id = `${r.question}\u0000${r.plan}`;
    if (seen.has(id)) throw new InputError(`line ${r.line}: question "${r.question}" and plan "${r.plan}" already appear on line ${seen.get(id)} (one measured run per question and plan)`);
    seen.set(id, r.line);
    if (!plans.has(r.plan)) plans.set(r.plan, { object: r.planObject });
    let m = byQuestion.get(r.question);
    if (!m) byQuestion.set(r.question, (m = new Map()));
    m.set(r.plan, { quality: r.quality, costUsd: r.costUsd });
  }
  return { plans, byQuestion };
}

export interface RecommendOut {
  plan: string | undefined;
  recommendation?: {
    meanQuality: number;
    lowerBound: number;
    certified: boolean;
    confidence: number;
    costPerRunUsd: number;
    callsUsed: number;
    fullTableCalls: number;
    samplesUsed: number;
    basis: string;
    warnings: string[];
  };
  questions: number;
  plans: number;
  minQuality: number;
}

export function runRecommend(rows: Row[], minQuality: number, confidence: number): Promise<RecommendOut> {
  const { plans, byQuestion } = table(rows);
  const keys = [...plans.keys()];
  const questions = [...byQuestion.keys()];
  // The log must be a full table: every plan on every question. No guessing, no credit for a missing pair.
  const missing: string[] = [];
  let missingCount = 0;
  for (const q of questions)
    for (const k of keys)
      if (!byQuestion.get(q)!.has(k)) {
        missingCount++;
        if (missing.length < 5) missing.push(`question "${q}" has no row for plan "${k}"`);
      }
  if (missingCount > 0) {
    throw new InputError(`the log must contain every plan on every question, like the plan table: ${missingCount} of ${questions.length * keys.length} (question, plan) pairs are missing (${questions.length} questions x ${keys.length} plans). For example: ${missing.join("; ")}.`);
  }
  const planObjs: Plan[] = keys.map((k) => ({ plan: k })); // an opaque one-knob plan: the key string is the identity
  return recommendPlan<string>({
    plans: planObjs,
    samples: questions,
    minQuality,
    confidence,
    options: {},
    concurrency: 1,
    evaluate: async (q, plan) => byQuestion.get(q)!.get(String(plan.plan))!,
  }).then((rec) => ({
    plan: rec ? String(rec.plan.plan) : undefined,
    recommendation: rec
      ? { meanQuality: rec.meanQuality, lowerBound: rec.lowerBound, certified: rec.certified, confidence: rec.confidence, costPerRunUsd: rec.costPerRunUsd, callsUsed: rec.callsUsed, fullTableCalls: rec.fullTableCalls, samplesUsed: rec.samplesUsed, basis: rec.basis, warnings: rec.warnings }
      : undefined,
    questions: questions.length,
    plans: keys.length,
    minQuality,
  }));
}

export async function runCertify(rows: Row[], plan: string, minQuality: number, confidence: number) {
  const { plans, byQuestion } = table(rows);
  if (!plans.has(plan)) throw new InputError(`no rows for plan "${plan}" in the log (plans in the log: ${[...plans.keys()].slice(0, 8).join(", ")}${plans.size > 8 ? ", ..." : ""})`);
  const questions = [...byQuestion.keys()].filter((q) => byQuestion.get(q)!.has(plan));
  const p: Plan = { plan };
  return certifyPolicy<string>({
    samples: questions,
    minQuality,
    confidence,
    concurrency: 1,
    scorerErrorRate: 0,
    decide: () => ({ plan: p, planKey: plan, fallback: false }),
    evaluate: async (q) => byQuestion.get(q)!.get(plan)!,
  });
}

const HELP = `pathwise: pick the cheapest plan that meets your bar, and measure it, from a log of measured runs (no key, no network).

  pathwise recommend --log runs.jsonl --min-quality 0.8 [--confidence 0.9] [--json]
  pathwise certify   --log fresh.jsonl --plan "<plan key>" --min-quality 0.8 [--confidence 0.9] [--json]

Each log line is a JSON object: {"question": "<id>", "plan": "<plan key or {knob: value}>", "quality": 0..1, "costUsd": number}.
recommend needs EVERY plan on EVERY question (like the plan table). certify needs a log of FRESH questions the plan was never tuned on.
Exit code 0 even when nothing is certified; 2 for bad input. Reads at most ${MAX_LOG_BYTES / 1048576} MB.`;

const fmt = (x: number) => x.toFixed(3);

export async function main(argv: string[], out: (s: string) => void, err: (s: string) => void): Promise<number> {
  try {
    const [cmd, ...rest] = argv;
    if (!cmd || cmd === "--help" || cmd === "-h" || cmd === "help") {
      out(HELP);
      return cmd ? 0 : 2;
    }
    if (cmd !== "recommend" && cmd !== "certify") throw new InputError(`unknown command "${cmd}" (expected recommend or certify; try --help)`);
    let values;
    try {
      ({ values } = parseArgs({
        args: rest,
        options: { log: { type: "string" }, plan: { type: "string" }, "min-quality": { type: "string" }, confidence: { type: "string" }, json: { type: "boolean" }, help: { type: "boolean" } },
        strict: true,
      }));
    } catch (e) {
      throw new InputError((e as Error).message);
    }
    if (values.help) {
      out(HELP);
      return 0;
    }
    if (!values.log) throw new InputError("--log <file> is required");
    if (values["min-quality"] === undefined) throw new InputError("--min-quality <0..1> is required");
    if (values["min-quality"].trim() === "") throw new InputError("--min-quality must be a number between 0 and 1 (it is empty)");
    if (values.confidence !== undefined && values.confidence.trim() === "") throw new InputError("--confidence must be a number strictly between 0 and 1 (it is empty)");
    const minQuality = Number(values["min-quality"]);
    if (!(minQuality >= 0 && minQuality <= 1)) throw new InputError("--min-quality must be a number between 0 and 1");
    const confidence = values.confidence === undefined ? 0.9 : Number(values.confidence);
    if (!(confidence > 0 && confidence < 1)) throw new InputError("--confidence must be a number strictly between 0 and 1");
    const rows = await readLog(values.log);
    if (cmd === "recommend") {
      if (values.plan !== undefined) throw new InputError("--plan is for certify only");
      const r = await runRecommend(rows, minQuality, confidence);
      if (values.json) out(JSON.stringify(r, null, 2));
      else if (!r.recommendation) out(`No plan meets the bar: none of the ${r.plans} plans reached a mean quality of ${minQuality} over ${r.questions} questions (nothing is invented).`);
      else {
        const x = r.recommendation;
        out(`Recommended plan: ${r.plan}`);
        out(`mean quality ${fmt(x.meanQuality)}, Wilson lower bound ${fmt(x.lowerBound)} at confidence ${x.confidence}, certified ${x.certified ? "yes" : "no"} (on the SAME questions it was chosen on: this is not a certificate; run certify on fresh questions)`);
        out(`cost per run $${x.costPerRunUsd.toFixed(5)}; ${x.callsUsed} of ${x.fullTableCalls} (question, plan) pairs used (${x.basis}); ${x.samplesUsed} questions`);
        for (const w of x.warnings) out(`warning: ${w}`);
      }
    } else {
      if (!values.plan) throw new InputError("--plan <key> is required for certify");
      const c = await runCertify(rows, values.plan, minQuality, confidence);
      if (values.json) out(JSON.stringify(c, null, 2));
      else {
        out(`Plan ${values.plan} on ${c.n} fresh questions: mean quality ${fmt(c.meanQuality)}, Wilson lower bound ${fmt(c.lowerBound)} at confidence ${c.confidence}, certified ${c.certified ? "yes" : "no"} (minimum quality ${minQuality}); cost $${c.costUsd.toFixed(5)} in the log`);
        for (const w of c.warnings) out(`warning: ${w}`);
      }
    }
    return 0;
  } catch (e) {
    if (e instanceof InputError) {
      err(`error: ${e.message}`);
      return 2;
    }
    err(`unexpected error: ${(e as Error).message}`);
    return 1;
  }
}

/**
 * Minimal Gemini client (Interactions API, raw fetch, zero dependencies) with:
 * - a disk cache, so re-runs and baselines reuse paid calls (cached calls still report their original cost);
 * - a persistent spend ledger and a hard budget cap (refuses a call that could cross the cap);
 * - retries with backoff on 429 / 5xx;
 * - a mock mode (PATHWISE_BENCH_MOCK=1) for testing the harness without an API key or spend.
 *
 * The API key is read from GEMINI_API_KEY in your environment and never written anywhere.
 */
import { createHash } from "node:crypto";
import { existsSync } from "node:fs";
import { appendFile, mkdir, readFile } from "node:fs/promises";
import { pathToFileURL } from "node:url";
import { BUDGET_USD, DEFAULT_RPM, MAX_OUTPUT_TOKENS, MODEL_SET, MODELS, PATHS, priceFor, type Tier } from "./config.js";

export interface CallResult {
  text: string;
  costUsd: number;
  latencyMs: number;
  usage: { input: number; output: number; thought: number };
  cached: boolean;
}

export class BudgetExceeded extends Error {}
/** The key has no quota for a model at all (free tier). Retrying cannot help. */
export class QuotaError extends Error {}
/** The daily request quota is used up. Re-run tomorrow: finished calls are cached. */
export class DailyQuotaError extends Error {}
/** Offline replay (PATHWISE_BENCH_OFFLINE=1) needed a response that is not in the cache. */
export class OfflineMiss extends Error {}
/** A model kept timing out or returning server errors. Re-run later: finished calls are cached. */
export class ModelUnavailableError extends Error {}

/** Set once a run hits a fatal error, so calls still in flight stop instead of retrying in the background. */
let stopping = false;
/** Models whose daily quota ran out during this run, with the message to repeat. */
const exhausted = new Map<string, string>();
export function stopAll() {
  stopping = true;
}

/** Per-request timeout. Answers are a few words; even high thinking rarely needs more than a minute. */
const REQUEST_TIMEOUT_MS = Number(process.env.PATHWISE_BENCH_TIMEOUT_MS ?? 90_000);
/** Attempts per call before giving up on a model (timeouts, network errors, 5xx, per-minute 429s). */
const MAX_ATTEMPTS = 8;

const MOCK = process.env.PATHWISE_BENCH_MOCK === "1";
/** Replay from the cache only: never calls the API, never writes. A missing response throws OfflineMiss. */
const OFFLINE = process.env.PATHWISE_BENCH_OFFLINE === "1";
export const offlineMisses = new Set<string>();
// PATHWISE_BENCH_CACHE_DIR (absolute path) redirects the cache, for tests that must not touch the real one.
const CACHE_OVERRIDE = process.env.PATHWISE_BENCH_CACHE_DIR;
const CACHE_DIR = CACHE_OVERRIDE
  ? pathToFileURL(CACHE_OVERRIDE.endsWith("/") ? CACHE_OVERRIDE : `${CACHE_OVERRIDE}/`)
  : new URL(MOCK ? "mock/" : `live-${MODEL_SET}/`, PATHS.cache);
/** Mock mode only: after this many calls a model throws the daily-quota error, to test the stop-and-resume path. */
const MOCK_DAILY_QUOTA = Number(process.env.PATHWISE_BENCH_MOCK_DAILY_QUOTA ?? 0);
const mockCalls = new Map<string, number>();
const CACHE_FILE = new URL("responses.jsonl", CACHE_DIR);
const LEDGER_FILE = new URL("spend.jsonl", CACHE_DIR);
const ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/interactions";

let cache: Map<string, CallResult> | undefined;
let spent = 0;
let reserved = 0;
let realCalls = 0;

async function init() {
  if (cache) return;
  await mkdir(CACHE_DIR, { recursive: true });
  cache = new Map();
  if (existsSync(CACHE_FILE)) {
    for (const line of (await readFile(CACHE_FILE, "utf8")).split("\n")) {
      if (!line) continue;
      const { key, result } = JSON.parse(line) as { key: string; result: CallResult };
      cache.set(key, result);
    }
  }
  if (existsSync(LEDGER_FILE)) {
    for (const line of (await readFile(LEDGER_FILE, "utf8")).split("\n")) {
      if (line) spent += (JSON.parse(line) as { costUsd: number }).costUsd;
    }
  }
}

/** The cached response for a call, or undefined. Never calls the API and never writes. */
export async function peekCache(tier: Tier, thinking: "low" | "high", prompt: string): Promise<CallResult | undefined> {
  await init();
  const model = MODELS[tier];
  return cache!.get(createHash("sha256").update(`${model}\n${thinking}\n${prompt}`).digest("hex"));
}

export function spendSoFar() {
  return { spentUsd: spent, budgetUsd: BUDGET_USD, realCalls, mock: MOCK, modelSet: MODEL_SET };
}

// ---------------------------------------------------------------- rate limiting

let rpm = Number(process.env.PATHWISE_BENCH_RPM ?? DEFAULT_RPM);
const starts: number[] = [];
let gate: Promise<void> = Promise.resolve();

/** Requests per minute across all models (free-tier keys have low limits). */
export function setRpm(n: number) {
  if (n > 0) rpm = n;
}

/** Waits until starting one more request keeps us under `rpm` in any 60 s window. */
function throttle(): Promise<void> {
  const next = gate.then(async () => {
    for (;;) {
      const now = Date.now();
      while (starts.length > 0 && now - starts[0] >= 60_000) starts.shift();
      if (starts.length < rpm) {
        starts.push(now);
        return;
      }
      await sleep(60_000 - (now - starts[0]) + 50);
    }
  });
  gate = next.catch(() => undefined);
  return next;
}

export async function generate(tier: Tier, thinking: "low" | "high", prompt: string, mock?: MockHint): Promise<CallResult> {
  await init();
  const model = MODELS[tier];
  const key = createHash("sha256").update(`${model}\n${thinking}\n${prompt}`).digest("hex");
  const hit = cache!.get(key);
  if (hit) return { ...hit, cached: true };
  if (OFFLINE) {
    offlineMisses.add(key);
    throw new OfflineMiss(`not in the cache: ${model} ${thinking} (${prompt.length} chars)`);
  }
  // This model's daily quota already ran out in this run: fail fast so other strategies can go on.
  const exhaustedMsg = exhausted.get(model);
  if (exhaustedMsg) throw new DailyQuotaError(exhaustedMsg);

  // Refuse a call whose worst case could cross the budget (parallel calls reserve their worst case).
  const price = priceFor(tier);
  const worst = ((prompt.length / 3) * price.inputPerM + MAX_OUTPUT_TOKENS * price.outputPerM) / 1e6;
  if (spent + reserved + worst > BUDGET_USD) {
    throw new BudgetExceeded(`Budget $${BUDGET_USD} would be exceeded (spent $${spent.toFixed(2)}). Stopping.`);
  }
  if (MOCK && MOCK_DAILY_QUOTA > 0) {
    const n = mockCalls.get(model) ?? 0;
    if (n >= MOCK_DAILY_QUOTA) {
      const msg = `Daily free-tier quota for ${model} is used up (simulated by PATHWISE_BENCH_MOCK_DAILY_QUOTA).`;
      exhausted.set(model, msg);
      throw new DailyQuotaError(msg);
    }
    mockCalls.set(model, n + 1);
  }
  reserved += worst;
  try {
    const result = MOCK ? mockCall(tier, thinking, prompt, mock) : await liveCall(model, tier, thinking, prompt);
    realCalls++;
    spent += result.costUsd;
    cache!.set(key, result);
    await appendFile(CACHE_FILE, JSON.stringify({ key, model, thinking, result }) + "\n");
    await appendFile(LEDGER_FILE, JSON.stringify({ at: new Date().toISOString(), model, costUsd: result.costUsd }) + "\n");
    return result;
  } finally {
    reserved -= worst;
  }
}

async function liveCall(model: string, tier: Tier, thinking: string, prompt: string): Promise<CallResult> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("Set GEMINI_API_KEY in your environment (export GEMINI_API_KEY=...)");
  const body = JSON.stringify({
    model,
    input: prompt,
    generation_config: { thinking_level: thinking, max_output_tokens: MAX_OUTPUT_TOKENS },
    store: false,
  });
  const retry = async (attempt: number, reason: string) => {
    if (stopping) throw new ModelUnavailableError("Run is stopping.");
    if (attempt + 1 >= MAX_ATTEMPTS) {
      throw new ModelUnavailableError(
        `${model} failed ${MAX_ATTEMPTS} times in a row (last: ${reason}). It is probably overloaded. ` +
          "Run the same command again later: everything finished so far is cached.",
      );
    }
    const waitMs = Math.min(120_000, 5000 * 2 ** attempt);
    console.log(`    ${model}: ${reason}; retry ${attempt + 2}/${MAX_ATTEMPTS} in ${Math.round(waitMs / 1000)}s`);
    await sleep(waitMs);
  };
  for (let attempt = 0; ; attempt++) {
    await throttle();
    if (stopping) throw new ModelUnavailableError("Run is stopping.");
    const start = performance.now();
    let res: Response;
    try {
      res = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "content-type": "application/json", "x-goog-api-key": apiKey },
        body,
        signal: AbortSignal.timeout(REQUEST_TIMEOUT_MS),
      });
    } catch (err) {
      const reason = (err as Error).name === "TimeoutError" ? `no answer within ${REQUEST_TIMEOUT_MS / 1000}s` : `network error (${(err as Error).message})`;
      await retry(attempt, reason);
      continue;
    }
    const latencyMs = performance.now() - start;
    let errorNote = "";
    if (res.status === 429) {
      const text = await res.clone().text();
      errorNote = `: ${quotaMessage(text).slice(0, 160)}`;
      // A hard zero quota (e.g. a model unavailable on the free tier) will never succeed: stop now.
      if (/limit:\s*0\b/i.test(text)) {
        throw new QuotaError(
          `${model} has no quota for this API key (${text.slice(0, 200)}). ` +
            "Use the free model set (default) or enable billing for the key's project.",
        );
      }
      // Daily quota used up: stop cleanly; the cache lets the next run continue where this one stopped.
      if (/per\s*day|perday|daily/i.test(text)) {
        const msg =
          `Daily free-tier quota for ${model} is used up. Run the same command again after it resets: ` +
          "everything finished so far is cached and will not be re-requested.\n" +
          `  Google said: ${quotaMessage(text)}`;
        exhausted.set(model, msg);
        throw new DailyQuotaError(msg);
      }
      // Per-minute limit: slow down for the rest of this run.
      if (rpm > 2) rpm = Math.max(2, Math.floor(rpm * 0.7));
    }
    if (res.status === 429 || res.status >= 500) {
      await retry(attempt, `HTTP ${res.status}${errorNote}`);
      continue;
    }
    const json = (await res.json()) as InteractionResponse;
    if (!res.ok) throw new Error(`Gemini HTTP ${res.status}: ${JSON.stringify(json).slice(0, 500)}`);
    const u = json.usage ?? {};
    const input = u.total_input_tokens ?? 0;
    const output = u.total_output_tokens ?? 0;
    const thought = u.total_thought_tokens ?? 0;
    // Thinking is billed at the output price. If total_tokens already equals input + output, the output
    // count includes thinking; otherwise thinking comes on top.
    const billedOutput = u.total_tokens !== undefined && Math.abs(u.total_tokens - (input + output)) < 2 ? output : output + thought;
    const price = priceFor(tier);
    const costUsd = (input * price.inputPerM + billedOutput * price.outputPerM) / 1e6;
    const text = (json.steps ?? [])
      .filter((s) => s.type === "model_output")
      .flatMap((s) => s.content ?? [])
      .filter((c) => c.type === "text")
      .map((c) => c.text ?? "")
      .join("")
      .trim();
    return { text, costUsd, latencyMs, usage: { input, output, thought }, cached: false };
  }
}

interface InteractionResponse {
  steps?: Array<{ type?: string; content?: Array<{ type?: string; text?: string }> }>;
  usage?: {
    total_input_tokens?: number;
    total_output_tokens?: number;
    total_thought_tokens?: number;
    total_tokens?: number;
  };
}

/** Raw response for the smoke test, so the usage fields can be checked by eye. */
export async function rawSmokeCall(tier: Tier, thinking: "low" | "high", prompt: string): Promise<unknown> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error("Set GEMINI_API_KEY in your environment (export GEMINI_API_KEY=...)");
  // Retry temporary overload (503) a few times so one demand spike does not fail the check.
  for (let attempt = 0; ; attempt++) {
    const res = await fetch(ENDPOINT, {
      method: "POST",
      headers: { "content-type": "application/json", "x-goog-api-key": apiKey },
      body: JSON.stringify({
        model: MODELS[tier],
        input: prompt,
        generation_config: { thinking_level: thinking, max_output_tokens: MAX_OUTPUT_TOKENS },
        store: false,
      }),
    });
    if (res.status >= 500 && attempt < 4) {
      await sleep(5000 * 2 ** attempt);
      continue;
    }
    return { status: res.status, attempts: attempt + 1, body: await res.json() };
  }
}

// ---------------------------------------------------------------- mock (offline testing only)

/** What the mock needs to fake a plausible answer: the gold answer and whether gold passages were retrieved. */
export interface MockHint {
  gold: string;
  goldRetrieved: number; // 0..2 supporting passages in the prompt
}

function mockCall(tier: Tier, thinking: string, prompt: string, hint?: MockHint): CallResult {
  const h = createHash("sha256").update(prompt + tier + thinking).digest();
  const r = h[0] / 255;
  const skill = { small: 0.45, medium: 0.65, large: 0.8 }[tier] + (thinking === "high" ? 0.08 : 0);
  const p = hint ? skill * (0.4 + 0.3 * hint.goldRetrieved) : 0.5;
  const text = hint && r < p ? hint.gold : "unknown";
  const input = Math.round(prompt.length / 4);
  const thought = thinking === "high" ? 600 : 120;
  const output = 8;
  const price = priceFor(tier);
  const costUsd = (input * price.inputPerM + (output + thought) * price.outputPerM) / 1e6;
  const latencyMs = { small: 500, medium: 1200, large: 3000 }[tier] * (thinking === "high" ? 2 : 1);
  return { text, costUsd, latencyMs, usage: { input, output, thought }, cached: false };
}

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

/** The human-readable part of a Gemini error body (falls back to the raw text). */
function quotaMessage(text: string): string {
  try {
    const m = (JSON.parse(text) as { error?: { message?: string } }).error?.message;
    if (m) return m.slice(0, 400);
  } catch {
    /* not JSON */
  }
  return text.slice(0, 400);
}

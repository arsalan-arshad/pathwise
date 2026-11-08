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
import { BUDGET_USD, MAX_OUTPUT_TOKENS, MODELS, PATHS, priceFor, type Tier } from "./config.js";

export interface CallResult {
  text: string;
  costUsd: number;
  latencyMs: number;
  usage: { input: number; output: number; thought: number };
  cached: boolean;
}

export class BudgetExceeded extends Error {}

const MOCK = process.env.PATHWISE_BENCH_MOCK === "1";
const CACHE_DIR = new URL(MOCK ? "mock/" : "live/", PATHS.cache);
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

export function spendSoFar() {
  return { spentUsd: spent, budgetUsd: BUDGET_USD, realCalls, mock: MOCK };
}

export async function generate(tier: Tier, thinking: "low" | "high", prompt: string, mock?: MockHint): Promise<CallResult> {
  await init();
  const model = MODELS[tier];
  const key = createHash("sha256").update(`${model}\n${thinking}\n${prompt}`).digest("hex");
  const hit = cache!.get(key);
  if (hit) return { ...hit, cached: true };

  // Refuse a call whose worst case could cross the budget (parallel calls reserve their worst case).
  const price = priceFor(tier);
  const worst = ((prompt.length / 3) * price.inputPerM + MAX_OUTPUT_TOKENS * price.outputPerM) / 1e6;
  if (spent + reserved + worst > BUDGET_USD) {
    throw new BudgetExceeded(`Budget $${BUDGET_USD} would be exceeded (spent $${spent.toFixed(2)}). Stopping.`);
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
  for (let attempt = 0; ; attempt++) {
    const start = performance.now();
    let res: Response;
    try {
      res = await fetch(ENDPOINT, {
        method: "POST",
        headers: { "content-type": "application/json", "x-goog-api-key": apiKey },
        body,
        signal: AbortSignal.timeout(180_000),
      });
    } catch (err) {
      if (attempt < 5) {
        await sleep(2000 * 2 ** attempt);
        continue;
      }
      throw err;
    }
    const latencyMs = performance.now() - start;
    if (res.status === 429 || res.status >= 500) {
      if (attempt < 6) {
        await sleep(Math.min(60_000, 2000 * 2 ** attempt));
        continue;
      }
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
  return { status: res.status, body: await res.json() };
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

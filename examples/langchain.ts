/**
 * LangChain.js adapter example: runs with a FAKE chat model (no `@langchain/core`, no key, no network).
 * With the real library, build a ChatOpenAI / ChatAnthropic / ChatGoogle per plan in `model` instead of `fakeChat`.
 * Simulated (SYNTHETIC): a model answers correctly when its capability (+ retrieval bonus) >= the hidden difficulty.
 */
import { performance } from "node:perf_hooks";
import { createPlanner, hashedTextFeaturizer } from "../src/index.js";
import { langchainPipeline, langchainCostUsd, messageText, type LangChainChatModel } from "../src/adapters/langchain.js";
import { createWorkload, type Query } from "./workload.js";

const CAP: Record<string, number> = { small: 0.35, medium: 0.6, large: 0.85 };
const BONUS: Record<number, number> = { 2: 0, 5: 0.05, 10: 0.08 };
const PRICE: Record<string, { inputPerMTok: number; outputPerMTok: number }> = {
  small: { inputPerMTok: 0.1, outputPerMTok: 0.4 },
  medium: { inputPerMTok: 0.75, outputPerMTok: 3.75 },
  large: { inputPerMTok: 2, outputPerMTok: 10 },
};
let failures = 0;
const check = (ok: boolean, msg: string) => {
  if (!ok) {
    failures++;
    console.log("FAIL:", msg);
  }
};

// Fake chat model with the LangChain shape: invoke(prompt) -> AIMessage-like with content and usage_metadata.
const fakeChat = (model: string, withUsage = true): LangChainChatModel => ({
  async invoke(prompt: string) {
    const [topK, difficulty] = prompt.split("|").map(Number);
    const ok = CAP[model] + BONUS[topK] >= difficulty;
    return {
      content: [{ type: "text", text: ok ? "right" : "wrong" }],
      ...(withUsage ? { usage_metadata: { input_tokens: 300 + topK * 200, output_tokens: 40, total_tokens: 340 + topK * 200 } } : {}),
    };
  },
});

// Unit checks of the helpers.
check(messageText({ content: "hi" }) === "hi" && messageText({ text: "t", content: "c" }) === "t", "messageText string/text");
check(messageText({ content: [{ type: "reasoning", text: "no" }, { type: "text", text: "a" }, "b"] }) === "ab", "messageText blocks");
check(Math.abs(langchainCostUsd({ input_tokens: 1000, output_tokens: 500, output_token_details: { reasoning: 300 } }, { inputPerMTok: 1, outputPerMTok: 10 }) - 0.006) < 1e-12, "reasoning is inside output_tokens");
const noUsage = langchainPipeline<Query>({ model: () => fakeChat("small", false), price: () => PRICE.small, prepare: () => ({ input: "2|0.1" }) });
check(await noUsage({ text: "x", difficulty: 0.1 }, { model: "small" }).then(() => false, (e: Error) => /no usage_metadata/.test(e.message)), "missing usage is an error");
const viaCost = langchainPipeline<Query>({ model: () => fakeChat("small", false), cost: () => 0.001, prepare: () => ({ input: "2|0.1", extraCostUsd: 0.5 }) });
check((await viaCost({ text: "x", difficulty: 0.1 }, { model: "small" })).costUsd === 0.501, "cost callback + extra cost");
check((() => { try { langchainPipeline({ model: () => fakeChat("small"), prepare: () => ({ input: "" }) }); return false; } catch { return true; } })(), "price or cost required");

async function main() {
  const { makeQueries } = createWorkload(42);
  const feats = hashedTextFeaturizer();
  const pipeline = langchainPipeline<Query>({
    model: (plan) => fakeChat(plan.model as string),
    price: (plan) => PRICE[plan.model as string],
    prepare: (q, plan) => ({ input: `${plan.topK}|${q.difficulty}` }),
  });
  const planner = createPlanner<Query, string>({
    space: { model: ["small", "medium", "large"], topK: [2, 5, 10] },
    pipeline,
    scorer: (_q, out) => (out === "right" ? 1 : 0),
    sla: { minQuality: 0.9 },
    featurize: (q) => feats(q.text),
    feedbackRate: 0,
  });
  const calib = await planner.calibrate(makeQueries(90));
  console.log(`calibrated: ${calib.runs} runs, $${calib.costUsd.toFixed(4)} (synthetic)`);
  const prod = makeQueries(500);
  let right = 0, cost = 0;
  for (const q of prod) {
    const r = await planner.run(q);
    if (r.output === "right") right++;
    cost += r.costUsd;
  }
  const maxCost = langchainCostUsd({ input_tokens: 300 + 10 * 200, output_tokens: 40 }, PRICE.large) * prod.length;
  console.log(`production (synthetic): accuracy ${((right / prod.length) * 100).toFixed(1)}%, $${cost.toFixed(4)} vs always-large/topK10 $${maxCost.toFixed(4)}`);

  const N = 20000;
  const q = prod[0];
  const plan = { model: "small", topK: 5 };
  const raw = fakeChat("small");
  let t = performance.now();
  for (let i = 0; i < N; i++) await raw.invoke(`5|${q.difficulty}`);
  const rawMs = (performance.now() - t) / N;
  t = performance.now();
  for (let i = 0; i < N; i++) await pipeline(q, plan);
  const wrapped = (performance.now() - t) / N;
  t = performance.now();
  for (let i = 0; i < N; i++) planner.decide(q);
  console.log(`overhead per call (ms, synthetic): adapter +${(wrapped - rawMs).toFixed(4)}, decide() ${((performance.now() - t) / N).toFixed(4)}`);
  console.log(failures === 0 ? "All langchain checks passed." : `${failures} check(s) FAILED.`);
  process.exit(failures === 0 ? 0 : 1);
}
main();

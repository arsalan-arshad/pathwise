/**
 * Smoke test of the Pathwise adapters against the REAL packages (`ai`, `@langchain/core`), using the packages'
 * own mock/fake models. No network, no API key. This is NOT a test against live providers.
 * Run with `bench/real-packages/run.sh` (installs the packages in a throwaway temp directory, never in the repo).
 */
import { generateText } from "ai";
import { MockLanguageModelV4 } from "ai/test";
import { AIMessage } from "@langchain/core/messages";
import { FakeListChatModel } from "@langchain/core/utils/testing";

const src = process.env.PW_SRC;
if (!src) throw new Error("PW_SRC (path to the repo's src directory) is not set; use run.sh");
const { vercelAIPipeline } = await import(`${src}/adapters/vercel-ai.ts`);
const { langchainPipeline } = await import(`${src}/adapters/langchain.ts`);

const plan = { model: "m", topK: 2, thinking: "low" } as any;
const price = { inputPerMTok: 1, outputPerMTok: 10 };
const expectedCost = (1000 * 1 + 500 * 10) / 1e6; // 0.006: reasoning is already inside outputTokens
const near = (a: number, b: number) => Math.abs(a - b) < 1e-12;
const check = (ok: boolean, what: string) => {
  if (!ok) throw new Error(`FAILED: ${what}`);
  console.log(`ok  ${what}`);
};
const version = async (p: string) => (await import(`${p}/package.json`, { with: { type: "json" } })).default.version;
console.log(`ai ${await version("ai")}, @langchain/core ${await version("@langchain/core")}`);

// 1) Vercel AI SDK: the real generateText on a mock model from `ai/test`.
const mock = new MockLanguageModelV4({
  doGenerate: async () =>
    ({
      content: [{ type: "text", text: "paris" }],
      finishReason: { unified: "stop", raw: "stop" },
      usage: {
        inputTokens: { total: 1000, noCache: 1000, cacheRead: 0, cacheWrite: 0 },
        outputTokens: { total: 500, text: 200, reasoning: 300 },
      },
      warnings: [],
    }) as any,
});
const v = await vercelAIPipeline({ generateText: generateText as any, models: { m: { model: mock, price } }, prepare: (q: string) => ({ prompt: q }) })("capital of France?", plan);
check(v.output === "paris", "vercel-ai: output text from the real generateText");
check(near(v.costUsd, expectedCost), `vercel-ai: cost from the real usage shape (${v.costUsd} = ${expectedCost})`);

// 2) LangChain: a real AIMessage (usage_metadata, text getter) and a real FakeListChatModel.
const msg = new AIMessage({ content: "paris", usage_metadata: { input_tokens: 1000, output_tokens: 500, total_tokens: 1500, output_token_details: { reasoning: 300 } } });
const l = await langchainPipeline({ model: () => ({ invoke: async () => msg }), prepare: (q: string) => ({ input: q }), price: () => price })("capital of France?", plan);
check(l.output === "paris", "langchain: output text from a real AIMessage");
check(near(l.costUsd, expectedCost), `langchain: cost from real usage_metadata (${l.costUsd} = ${expectedCost})`);
let threw = false;
try {
  await langchainPipeline({ model: () => new FakeListChatModel({ responses: ["paris"] }) as any, prepare: (q: string) => ({ input: q }), price: () => price })("capital of France?", plan);
} catch {
  threw = true;
}
check(threw, "langchain: a model that reports no usage fails loudly instead of guessing the cost");
console.log("all real-package smoke checks passed");

/**
 * OpenAI-compatible adapter example and tests. Runs against a LOCAL fake server started with node:http: a real HTTP
 * round trip on 127.0.0.1, no external network, no API key, no live provider (SYNTHETIC).
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import type { AddressInfo } from "node:net";
import { performance } from "node:perf_hooks";
import { createPlanner, hashedTextFeaturizer } from "../src/index.js";
import { chatCostUsd, openAICompatiblePipeline } from "../src/adapters/openai-compatible.js";
import { createWorkload, type Query } from "./workload.js";

const CAP: Record<string, number> = { "fake-small": 0.35, "fake-medium": 0.6, "fake-large": 0.85 };
const BONUS: Record<number, number> = { 2: 0, 5: 0.05, 10: 0.08 };
const PRICE = {
  small: { inputPerMTok: 0.1, outputPerMTok: 0.4 },
  medium: { inputPerMTok: 0.75, outputPerMTok: 3.75 },
  large: { inputPerMTok: 2, outputPerMTok: 10 },
};
const models = {
  small: { model: "fake-small", price: PRICE.small },
  medium: { model: "fake-medium", price: PRICE.medium },
  large: { model: "fake-large", price: PRICE.large },
};
let failures = 0;
const check = (ok: boolean, msg: string) => {
  if (!ok) {
    failures++;
    console.log("FAIL:", msg);
  }
};

type Handler = (req: IncomingMessage, body: any, res: ServerResponse, n: number) => void;
let handler: Handler = () => {};
let hits = 0;
let lastReq: { headers: IncomingMessage["headers"]; body: any; url?: string } | undefined;
const server = createServer((req, res) => {
  let raw = "";
  req.on("data", (c) => (raw += c));
  req.on("end", () => {
    const body = raw ? JSON.parse(raw) : {};
    lastReq = { headers: req.headers, body, url: req.url };
    handler(req, body, res, ++hits);
  });
});
await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
const baseUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1/`;
const json = (res: ServerResponse, status: number, obj: unknown, headers: Record<string, string> = {}) => {
  res.writeHead(status, { "content-type": "application/json", ...headers });
  res.end(JSON.stringify(obj));
};
const ok = (content: string, usage?: object) => ({ choices: [{ message: { role: "assistant", content }, finish_reason: "stop" }], ...(usage ? { usage } : {}) });
const reset = (h: Handler) => ((handler = h), (hits = 0));
const make = (extra: Partial<Parameters<typeof openAICompatiblePipeline<string>>[0]> = {}) =>
  openAICompatiblePipeline<string>({ baseUrl, models, prepare: (q) => ({ messages: [{ role: "user", content: q }] }), retryBaseMs: 5, ...extra });
const plan = { model: "small" };
const fails = (p: Promise<unknown>, re: RegExp) => p.then(() => false, (e: Error) => re.test(e.message));

// normal + request shape + auth
reset((_r, _b, res) => json(res, 200, ok("hello", { prompt_tokens: 1000, completion_tokens: 500, total_tokens: 1500 })));
const r1 = await make({ apiKey: "sk-secret", settings: () => ({ reasoning_effort: "low", max_completion_tokens: 50 }) })("hi", plan);
check(r1.output === "hello" && Math.abs(r1.costUsd - (1000 * 0.1 + 500 * 0.4) / 1e6) < 1e-12, "normal response and cost");
check(lastReq?.url === "/v1/chat/completions" && lastReq.headers.authorization === "Bearer sk-secret", "url and auth header");
check(lastReq?.body.model === "fake-small" && lastReq.body.reasoning_effort === "low" && lastReq.body.max_completion_tokens === 50 && lastReq.body.messages[0].content === "hi", "request body");
await make()("hi", plan);
check(lastReq?.headers.authorization === undefined, "no key: no auth header");

// usage with reasoning tokens (inside completion_tokens by default, on top when configured)
reset((_r, _b, res) => json(res, 200, ok("x", { prompt_tokens: 1000, completion_tokens: 500, completion_tokens_details: { reasoning_tokens: 300 } })));
const inside = await make()("q", { model: "large" });
const onTop = await make({ reasoningInCompletionTokens: false })("q", { model: "large" });
check(Math.abs(inside.costUsd - 0.007) < 1e-12 && Math.abs(onTop.costUsd - 0.01) < 1e-12, `reasoning tokens: ${inside.costUsd} / ${onTop.costUsd}`);
check(Math.abs(chatCostUsd({ prompt_tokens: 1000, completion_tokens: 500 }, { inputPerMTok: 1, outputPerMTok: 10 }) - 0.006) < 1e-12, "chatCostUsd");

// missing usage: loud error, or cost callback
reset((_r, _b, res) => json(res, 200, ok("x")));
check(await fails(make()("q", plan), /no usage/), "missing usage fails loudly");
check((await make({ cost: () => 0.002, })("q", plan)).costUsd === 0.002, "cost callback without usage");

// 429 then success (Retry-After honoured as a header, small value), 500 exhausted
reset((_r, _b, res, n) => (n === 1 ? json(res, 429, { error: { message: "slow down" } }) : json(res, 200, ok("late", { prompt_tokens: 1, completion_tokens: 1 }))));
check((await make()("q", plan)).output === "late" && hits === 2, `429 then success (${hits} hits)`);
reset((_r, _b, res) => json(res, 500, { error: { message: "boom" } }));
check((await fails(make({ retries: 2 })("q", plan), /HTTP 500.*after 3 attempts/)) && hits === 3, `500 exhausted (${hits} hits)`);
reset((_r, _b, res) => json(res, 400, { error: { message: "bad request" } }));
check((await fails(make()("q", plan), /HTTP 400/)) && hits === 1, "400 is not retried");

// timeout (server never answers), abort, bad JSON
reset(() => {});
check((await fails(make({ timeoutMs: 40, retries: 1 })("q", plan), /timed out after 40 ms.*after 2 attempts/)) && hits === 2, "timeout retried then fails");
{
  reset(() => {});
  const ac = new AbortController();
  setTimeout(() => ac.abort(), 30);
  const t = performance.now();
  const aborted = await make({ signal: ac.signal, timeoutMs: 5000 })("q", plan).then(() => "no", (e: Error) => e.name);
  check(aborted === "AbortError" && performance.now() - t < 1000 && hits === 1, `abort: ${aborted} after ${(performance.now() - t).toFixed(0)} ms, ${hits} hit(s)`);
}
reset((_r, _b, res) => (res.writeHead(200), res.end("<html>not json</html>")));
check((await fails(make()("q", plan), /not JSON/)) && hits === 1, "bad JSON");
reset((_r, _b, res) => json(res, 200, { choices: [] }));
check(await fails(make()("q", plan), /no choices\[0\]/), "no choices");

// pre-aborted signal: no request is made
{
  reset((_r, _b, res) => json(res, 200, ok("x", { prompt_tokens: 1, completion_tokens: 1 })));
  const ac = new AbortController();
  ac.abort();
  const name = await make({ signal: ac.signal })("q", plan).then(() => "no", (e: Error) => e.name);
  check(name === "AbortError" && hits === 0, `pre-aborted: ${name}, ${hits} hit(s)`);
}
// abort during the backoff sleep
{
  reset((_r, _b, res) => json(res, 503, { error: { message: "later" } }));
  const ac = new AbortController();
  setTimeout(() => ac.abort(), 30);
  const t = performance.now();
  const name = await make({ signal: ac.signal, retryBaseMs: 5000 })("q", plan).then(() => "no", (e: Error) => e.name);
  check(name === "AbortError" && performance.now() - t < 1000 && hits === 1, `abort during backoff: ${name}, ${hits} hit(s)`);
}
// a stalled body (headers sent, body never finishes) is a timeout, retried
reset((_r, _b, res) => (res.writeHead(200, { "content-type": "application/json" }), res.write('{"choices":')));
check((await fails(make({ timeoutMs: 40, retries: 1 })("q", plan), /timed out after 40 ms.*after 2 attempts/)) && hits === 2, `stalled body retried (${hits} hits)`);
// HTTP-date Retry-After is parsed (a date in the past: falls back to the backoff)
reset((_r, _b, res, n) => (n === 1 ? json(res, 429, { error: { message: "x" } }, { "retry-after": new Date(Date.now() + 40).toUTCString() }) : json(res, 200, ok("ok", { prompt_tokens: 1, completion_tokens: 1 }))));
check((await make({ retryBaseMs: 5 })("q", plan)).output === "ok" && hits === 2, "HTTP-date Retry-After");
// 429 then success: cost only from the successful response
reset((_r, _b, res, n) => (n === 1 ? json(res, 429, { error: { message: "x" }, usage: { prompt_tokens: 9e6, completion_tokens: 9e6 } }) : json(res, 200, ok("ok", { prompt_tokens: 1000, completion_tokens: 500 }))));
check(Math.abs((await make()("q", plan)).costUsd - (1000 * 0.1 + 500 * 0.4) / 1e6) < 1e-12, "retry cost is from the successful attempt only");
// invalid usage values fail loudly instead of mis-costing
for (const bad of [{ prompt_tokens: 10, completion_tokens: null }, { prompt_tokens: 10, completion_tokens: "50" }, { prompt_tokens: -1, completion_tokens: 5 }]) {
  reset((_r, _b, res) => json(res, 200, ok("x", bad)));
  check(await fails(make()("q", plan), /non-negative numbers/), `bad usage ${JSON.stringify(bad)}`);
}
reset((_r, _b, res) => json(res, 200, ok("x", { prompt_tokens: 10, completion_tokens: 5 })));
check(await fails(make({ reasoningInCompletionTokens: false })("q", plan), /reasoning_tokens/), "reasoning on top needs reasoning_tokens");
// construction-time validation
for (const [label, bad] of [["baseUrl", { baseUrl: "not a url" }], ["retries", { retries: -1 }], ["retries", { retries: 1.5 }], ["timeout", { timeoutMs: 0 }]] as const) {
  check((() => { try { make(bad as never); return false; } catch { return true; } })(), `invalid ${label} ${JSON.stringify(bad)} throws at construction`);
}
// secrets in headers are scrubbed too
reset((_r, _b, res) => json(res, 400, { error: { message: "denied for key-abc-123456" } }));
check(await make({ headers: { "x-api-key": "key-abc-123456" } })("q", plan).then(() => false, (e: Error) => !e.message.includes("key-abc-123456")), "header secret scrubbed from errors");

// the key never appears in an error message, even if the server echoes it
reset((_r, _b, res) => json(res, 400, { error: { message: "invalid key sk-secret-123" } }));
check(await make({ apiKey: "sk-secret-123" })("q", plan).then(() => false, (e: Error) => !e.message.includes("sk-secret-123") && /redacted/.test(e.message)), "api key scrubbed from errors");
check(await fails(make()("q", { model: "nope" }), /no model "nope"/), "unknown model in plan");

// planner end to end over HTTP (SYNTHETIC): the fake server answers from capability + retrieval depth + hidden difficulty
reset((_r, b, res) => {
  const [topK, difficulty] = (b.messages[0].content as string).split("|").map(Number);
  json(res, 200, ok(CAP[b.model] + BONUS[topK] >= difficulty ? "right" : "wrong", { prompt_tokens: 300 + topK * 200, completion_tokens: 40 }));
});
const { makeQueries } = createWorkload(42);
const feats = hashedTextFeaturizer();
const pipeline = openAICompatiblePipeline<Query>({ baseUrl, models, prepare: (q, p) => ({ messages: [{ role: "user", content: `${p.topK}|${q.difficulty}` }] }) });
const planner = createPlanner<Query, string>({
  space: { model: ["small", "medium", "large"], topK: [2, 5, 10] },
  pipeline,
  scorer: (_q, out) => (out === "right" ? 1 : 0),
  sla: { minQuality: 0.9 },
  featurize: (q) => feats(q.text),
  feedbackRate: 0,
});
const calib = await planner.calibrate(makeQueries(90));
const prod = makeQueries(300);
let right = 0, cost = 0;
for (const q of prod) {
  const r = await planner.run(q);
  if (r.output === "right") right++;
  cost += r.costUsd;
}
const maxCost = chatCostUsd({ prompt_tokens: 300 + 10 * 200, completion_tokens: 40 }, PRICE.large) * prod.length;
console.log(`calibrated: ${calib.runs} runs over HTTP; production (synthetic): accuracy ${((right / prod.length) * 100).toFixed(1)}%, $${cost.toFixed(4)} vs always-large/topK10 $${maxCost.toFixed(4)}`);

// overhead: pipeline call vs a bare fetch of the same request to the same server (server time cancels out)
reset((_r, _b, res) => json(res, 200, ok("right", { prompt_tokens: 100, completion_tokens: 10 })));
const N = 1500;
const bare = JSON.stringify({ model: "fake-small", messages: [{ role: "user", content: "5|0.1" }] });
const pipe = openAICompatiblePipeline<string>({ baseUrl, models, prepare: (q) => ({ messages: [{ role: "user", content: q }] }) });
for (let i = 0; i < 100; i++) await fetch(`${baseUrl}chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: bare }).then((r) => r.text());
let t = performance.now();
for (let i = 0; i < N; i++) await fetch(`${baseUrl}chat/completions`, { method: "POST", headers: { "content-type": "application/json" }, body: bare }).then((r) => r.text());
const bareMs = (performance.now() - t) / N;
t = performance.now();
for (let i = 0; i < N; i++) await pipe("5|0.1", plan);
const pipeMs = (performance.now() - t) / N;
t = performance.now();
for (let i = 0; i < 2000; i++) planner.decide(prod[i % prod.length]);
console.log(`overhead per call (ms, synthetic, localhost, excluding server time): adapter ${(pipeMs - bareMs).toFixed(3)} (bare fetch ${bareMs.toFixed(3)}, adapter ${pipeMs.toFixed(3)}); decide() ${((performance.now() - t) / 2000).toFixed(3)}`);

server.close();
server.closeAllConnections?.();
console.log(failures === 0 ? "All openai-compatible checks passed." : `${failures} check(s) FAILED.`);
process.exit(failures === 0 ? 0 : 1);

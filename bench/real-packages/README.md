# Real-package smoke test

Checks that the Pathwise adapters work against the REAL `ai` and `@langchain/core` packages, using the packages' own mock/fake models.

```bash
bench/real-packages/run.sh        # or: npm run smoke:real-packages
```

`run.sh` installs `ai` and `@langchain/core` into a throwaway temp directory (never into this repo, never as a dependency), runs
`smoke.mts` against `src/adapters/`, and deletes the directory. It needs network access for `npm install` only.

**What it checks** (smoke-tested on Nov 13, 2026 against `ai` 7.0.171 and `@langchain/core` 1.4.0):
- `pathwise/vercel-ai` with the real `generateText` on `MockLanguageModelV4`: output text and cost from the real usage shape (1000 in / 500 out with 300 reasoning inside out = 0.006).
- `pathwise/langchain`: a real `AIMessage` (`usage_metadata`) for output text and cost (same 0.006), and a real `FakeListChatModel` for the no-usage check (a model that reports no usage fails loudly instead of guessing the cost).

**What it does not check:** live providers (no API key, no provider calls), streaming, tool calls, structured output, or other versions of the packages.

Desktop ran it on Nov 13, 2026 and all checks passed; the terminal did not re-run it (its sandbox cannot execute installed packages).

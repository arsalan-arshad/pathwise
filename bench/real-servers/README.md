# Real-server smoke test (Ollama)

`npm run smoke:ollama` checks `@arsalandev/pathwise/embeddings` and `/openai-compatible` against a **real** local Ollama server
at `http://localhost:11434/v1`. It is separate from `bench/real-packages` (which installs SDK packages into a throwaway
directory) because it needs no packages, only a running server.

```bash
ollama serve                         # if the Ollama app is not already running
ollama pull nomic-embed-text         # the one approved model (274 MB per the Ollama library page); this script never pulls anything
npm run smoke:ollama                 # embeddings checks only
OLLAMA_CHAT_MODEL=<model> npm run smoke:ollama   # adds the chat checks: a SECOND model, needs a separate yes before downloading
```

Environment: `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`), `OLLAMA_EMBED_MODEL` (default `nomic-embed-text`),
`OLLAMA_CHAT_MODEL` (none by default: chat checks are skipped).

- No server reachable: prints `SKIP: ...` and exits 0.
- Prints a table: check, result with the local round-trip time, and the difference from what the docs / README say
  ("matches" or the difference). It edits no docs. Read the output first. The times describe the machine it ran on only.
- Embeddings: batch of 5 returns 5 vectors in input order, the featurizer's dimension check, whether `dimensions` is honored,
  whether the response has `usage` (Ollama does not document it), abort/timeout (the embeddings adapter has none).
- Chat (with a chat model): completion, `usage` present, the `cost` callback, whether `reasoning_effort` is accepted or
  rejected, abort via signal, timeout. 429 / 5xx are not testable against a healthy local server.
- Exit 1 if any check hit an unexpected error (a missing `usage` makes the plain pipeline fail loudly, by design: that row
  shows as an error and is the finding).

`npm run smoke:ollama:check` tests the script itself against FAKE local servers (SYNTHETIC). It says nothing about real Ollama.

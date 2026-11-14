# Security policy

## Supported versions

| Version | Supported |
|---|---|
| 0.1.x | yes |
| older | no |

## Reporting a vulnerability

Please report it **privately** through GitHub: open the repository's **Security** tab and choose **Report a vulnerability**
(<https://github.com/arsalan-arshad/pathwise/security/advisories/new>). Do not open a public issue or pull request for a
security problem.

Include the affected version, what you did, what you expected and what happened, and a minimal reproduction that does not
contain real API keys or private data.

Response times are best effort: this is a small, independently maintained project, so there is no guaranteed turnaround.

## Scope

In scope: the `@arsalandev/pathwise` library and its adapters (`/vercel-ai`, `/langchain`, `/embeddings`, `/openai-compatible`).

Facts that shape the risk (as of 0.1.0):

- The library has **zero runtime dependencies**.
- The library never reads API keys from the environment. Keys reach the adapters only through options you pass (for example
  `apiKey`), and the OpenAI-compatible adapter removes the configured key and header values (6 characters or longer) from error messages.
- `bench/` and `examples/` are development tools, not part of the published package.

Out of scope: vulnerabilities in the third-party SDKs or servers you connect to (Vercel AI SDK, LangChain.js, model
providers, Ollama), and problems that need an attacker who already controls your pipeline function or scorer.

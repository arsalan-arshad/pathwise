# Pathwise brand assets: brief for the design agent

You are creating the visual identity assets for **Pathwise**, an open-source TypeScript SDK.
Read this whole brief before starting. Put **every file inside `assets/`** in this repo, using the
folder layout below. Do not modify anything outside `assets/` except the single README change in step 7.

## 1. What Pathwise is (so the visuals mean something)

Pathwise is a **query planner for RAG and agent pipelines**. A developer declares a service level
("90% of answers must be correct, under 3 seconds"). For every request, Pathwise looks at the many
possible ways to run it (which model, how many documents to retrieve, reranker on/off, agent step budget)
and **picks the cheapest path that still meets the quality bar**.

- Tagline: **The cheapest path to the answer.**
- Pitch: **Databases have had query optimizers for 50 years. AI pipelines don't. Pathwise is that optimizer.**
- Audience: senior software engineers and AI engineers. Tone: precise, calm, technical, confident.
  Think Linear, Vercel, Stripe docs. Not playful, not "AI magic", no robots, no brains, no sparkles, no glowing orbs.
- Made by **Arsalan Arshad**, studio: **arsalandev.studio**. License: MIT.

## 2. Core visual idea

**Many possible paths, one chosen path.** A small graph: a start node fans out into several
branches (the possible plans). All branches are thin and muted grey except **one**, which is bold,
in the accent colour, and reaches the target. That single highlighted route is the brand.
Keep it geometric: circles for nodes, straight lines or gentle 90°/45° turns, consistent stroke widths.

## 3. Brand tokens (use exactly these)

| Token | Hex | Use |
|---|---|---|
| Ink | `#0B1220` | Dark backgrounds, primary text on light |
| Studio Navy | `#1E3A5F` | Primary brand colour (matches arsalandev.studio) |
| Path Teal | `#2DD4BF` | The chosen path, accents on dark backgrounds |
| Path Teal Deep | `#0D9488` | The chosen path / accents on light backgrounds (contrast) |
| Slate | `#94A3B8` | Unchosen paths, secondary text on dark |
| Line | `#E2E8F0` | Hairlines, grid, borders on light |
| Paper | `#F7F8FA` | Light backgrounds |
| White | `#FFFFFF` | Text on dark |

- Fonts: **Inter** (600/700 for headings, 400/500 for body) and **JetBrains Mono** for code and numbers.
  Both are free (Google Fonts). Convert text to outlines in final SVGs so they render anywhere.
- Wordmark: `pathwise` in lowercase, Inter 700, letter-spacing -2%. Colour Ink on light, White on dark.
- Optional subtle background: a faint dot grid (Line colour on light, `#1E293B` on dark), never busy.

## 4. Deliverables and exact sizes

Make **SVG masters first**, then export PNGs from those SVGs (do not redraw for each size).

```
assets/
  logo/
    pathwise-mark.svg                 square mark only, 512x512 viewBox, transparent
    pathwise-mark-on-dark.svg         same mark tuned for dark backgrounds
    pathwise-logo-horizontal.svg      mark + wordmark, for light backgrounds
    pathwise-logo-horizontal-dark.svg mark + wordmark, for dark backgrounds
  icon/
    icon-16.png  icon-32.png  icon-180.png  icon-192.png  icon-512.png
    favicon.ico                       (16 + 32 inside)
  banner/
    readme-banner-light.svg / .png    1280x320  (GitHub README, light mode)
    readme-banner-dark.svg  / .png    1280x320  (GitHub README, dark mode)
    github-social-preview.png         1280x640  (repo Settings > Social preview), under 1 MB
    studio-card.png                   1600x900  (listing on arsalandev.studio)
  social/
    01-launch-1200x675.png            X / LinkedIn landscape
    01-launch-1080x1080.png           Instagram / LinkedIn square
    02-pitch-1200x675.png
    02-pitch-1080x1080.png
    03-how-it-works-1200x675.png
    03-how-it-works-1080x1080.png
    04-results-1200x675.png
    04-results-1080x1080.png
  README.md                           index of every asset with a thumbnail and where it is used
```

### Mark / icon
- The fan-out graph from section 2, in a square. One start node (left or bottom), 3 to 4 branches,
  one highlighted branch to a target node (a ring or a filled circle with a ring).
- Must stay readable at **16x16**: for icon-16/32 use a simplified version (start dot, one bold path,
  target dot, at most one grey branch). Check it by viewing the PNGs at real size.
- The app-icon PNGs (180/192/512) sit on a rounded square in Ink with the mark in Teal/Slate/White.

### README banners (1280x320)
- Left: horizontal logo. Under it the tagline "The cheapest path to the answer."
- Right: the path graph drawn larger and more detailed (6 to 10 grey branches, one teal route),
  optionally with tiny mono labels on a few nodes like `small · topK 2`, `large · rerank`.
- Leave ~48px safe padding. Light version on Paper, dark version on Ink.

### GitHub social preview (1280x640)
- Logo, tagline, one line: "Query planner for RAG and agent pipelines · TypeScript · MIT",
  and `github.com/arsalan-arshad/pathwise` in mono at the bottom. Keep 40px clear on all sides
  (GitHub crops). Dark (Ink) background.

### Studio card (1600x900)
- For listing Pathwise as a project on arsalandev.studio. Same look as the social preview, plus a
  small footer line: "Built by Arsalan Arshad · arsalandev.studio". Light (Paper) background to match the studio site.

### Social posts (two formats each: 1200x675 and 1080x1080)
1. **Launch**: "Pathwise is open source." + tagline + logo + `github.com/arsalan-arshad/pathwise`.
2. **Pitch**: the pitch sentence from section 1, large, with the path graph as a quiet background element.
3. **How it works**: three numbered steps, each with a tiny icon drawn in the same line style:
   1 "Declare your SLA" · 2 "Calibrate on real queries" · 3 "Cheapest plan per request".
4. **Results**: a simple horizontal bar chart, mono numbers:
   - "Always the biggest model: $33.00"  (full-width grey bar)
   - "Pathwise: $4.93"  (short teal bar)
   - Caption: "Same 100% accuracy on a 1,000-query workload."
   - **Required label, clearly visible, not tiny: "Synthetic benchmark (simulated workload). Real savings will be smaller."**
   Do not round, change, or add any numbers. Do not invent other statistics anywhere.

## 5. Hard rules

- Spell every text string exactly as written here. Lowercase `pathwise` in the wordmark,
  "Pathwise" in sentences.
- No logos or names of other companies or models (no OpenAI, Anthropic, Google, LangChain, etc.).
- No stock photos, no people, no AI-generated illustrations of robots or brains. Flat vector only.
- WCAG AA contrast for all text (Teal `#2DD4BF` only on dark backgrounds; use `#0D9488` on light).
- Keep files small: SVGs clean (no embedded bitmaps), PNGs optimised; social images under 1 MB.

## 6. How to produce the PNGs

- Use a one-off tool such as `npx @resvg/resvg-js-cli`, `npx sharp-cli`, or a headless-browser screenshot.
- **Do not add any dependency to `package.json`.** The Pathwise core must stay at zero dependencies.
- After exporting, open every PNG and check: correct size, no clipped text, fonts rendered (not fallback serif).

## 7. Git and README

- Work on a new branch `assets/brand`. Never commit to `main` directly (it is protected; changes go
  through a pull request, and CI must pass).
- On that branch, add the banner to the top of the root `README.md`, replacing the `# Pathwise` heading line,
  so it switches with the viewer's theme:

```html
<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner/readme-banner-dark.png">
  <img alt="Pathwise: the cheapest path to the answer" src="assets/banner/readme-banner-light.png">
</picture>
```

  Keep the existing badges line right below it. Change nothing else in the README.
- Write `assets/README.md` listing every file, its size, and where it is used.
- Commit with a clear message, push the branch, and open a pull request titled "Add brand assets".

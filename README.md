<div align="center">
  <img src="./logo.png" alt="ModelArchViz" width="420" />

  **🧭 Explore model architectures beside the code that defines them 🧭**
</div>

ModelArchViz is a Next.js app for inspecting neural network architecture examples alongside PyTorch/JAX code and source-paper context.

Use it to switch between embedded model specs, expand architecture blocks, select layers, and see the matching code lines highlighted in the editor.

## Install

```bash
git clone https://github.com/tsilva/modelarchviz.git
cd modelarchviz
pnpm install --frozen-lockfile
infisical login
pnpm dev --port auto
```

Open the local URL printed by the development server.

## Commands

```bash
pnpm generate:model-artifacts  # generate the UI source map, Colab notebooks, and PDF worker
pnpm dev --port auto           # fetch Infisical dev secrets, generate artifacts and start Next.js on an available local port
pnpm build                     # generate artifacts and build the production app
pnpm start                     # serve the production build after pnpm build
pnpm typecheck                 # generate artifacts and run TypeScript checks
```

## Optional chat

The chat pane uses the server-side OpenRouter API route. Set `OPENROUTER_API_KEY` to enable it. The optional `OPENROUTER_MODEL`, `OPENROUTER_APP_URL`, and `OPENROUTER_APP_NAME` variables override the default model and request attribution.

## Notes

- Model identity, route, paper, and source metadata live in `app/model-routes.ts`; architecture nodes and code highlights live in `app/model-arch-viz-app.tsx`. The optional chat pane uses `app/api/chat/route.ts`.
- Canonical model source files live in `app/model-notebooks` as Jupytext-style `py:percent` notebooks.
- `pnpm generate:model-artifacts` writes cleaned Python sources into `app/generated/model-sources.ts`, Colab notebooks to `public/notebooks`, and the pinned PDF.js worker to `public/pdf.worker.min.mjs`.
- `# %% [notebook-only]` cells are included in generated notebooks for examples and smoke tests, but excluded from generated site preview code.
- The code pane reads the generated source map; edit the canonical notebook sources instead of editing generated artifacts directly.
- Colab buttons use `NEXT_PUBLIC_GITHUB_REPOSITORY` and `NEXT_PUBLIC_GITHUB_BRANCH` when set. They default to `tsilva/modelarchviz` and `main`.
- Current examples range from MLPs and recurrent/Seq2Seq models through classic CNNs, Inception, ResNet, U-Net, BERT, GPT-2, and ViT.
- Paper panes render the checked-in PDFs under `public/papers`.
- No database, server-side storage, or user-data persistence is configured. Google Analytics and Sentry provide analytics, error monitoring, tracing, and replay.
- `NEXT_PUBLIC_SITE_URL` is optional and sets the absolute base URL for social metadata, sitemap, and robots output. It falls back to `https://modelarch.tsilva.eu`.
- Authoring brand assets live under `assets/brand`, runtime web and SEO assets live under `public/brand/web-seo`, and the root `logo.png` is used for repository and README display.

## Local credentials

Private local values declared in `.keyenv.toml` are now managed in the linked Infisical `modelarchviz` project, Development environment, root folder. `.infisical.json` contains only public project connection settings. Authenticate with `infisical login`, then run `pnpm secrets:check` and `pnpm dev --port auto`.

The launcher fetches only `OPENROUTER_API_KEY` and `SENTRY_AUTH_TOKEN` in memory and removes manager tokens from the app process. Missing credentials cannot fall back to old dotenv values; failed fetches stop before app startup. Public configuration can remain in dotenv files.

Use `pnpm build:secrets` or `pnpm start:secrets` for local commands that need credentials. Vercel keeps its ordinary `pnpm build` command and receives production variables through the isolated `modelarchviz-production` Infisical project and its Vercel Production sync. Sync changes need a new deployment before they affect the live app.

`pnpm secrets:migrate` is a one-time, manifest-bound Keychain import with conflict checks and exact readback verification. Keychain originals remain until migration and credential rotation are verified. Separate Infisical projects isolate access; independent provider keys are needed for independent dev/prod budgets and revocation.

## License

No license file is present in this repository yet.

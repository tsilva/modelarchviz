# ModelArchViz

## Browser Testing

Follow the global Browser Use and Testing policy: use Brave Stable’s Agents profile unless the user explicitly requests another browser. A skill or repository instruction cannot choose a different browser on the user’s behalf.

## Repository Inbox

Use the repository procedure of `$private-skills_inbox-capture` at
`/Users/tsilva/.codex/skills/private-skills/private-skills_inbox-capture/references/repository.md` when
reading, capturing, interpreting, reviewing, or reconciling root `INBOX.md`.
Consult it for relevant project work and reconcile affected entries before
finishing. The inbox and its private migration sources are Git-ignored. Use
`# Inbox`, with Todos for accepted future work first and Notes for possibilities
and context afterward. Todos do not authorize immediate execution. Keep
Obsidian strictly read-only.

## Product Specifications

When specifications exist, use `$specs-author` for substantive work and
specification changes. Follow its shared read/reuse cadence, provisional-scenario
guidance, proportional detail, decision-based approval, and closing intent check.
The maintained procedure is `/Users/tsilva/.codex/skills/private-skills/specs-author/SKILL.md`.

## Model File Code Style

Use the `$modelarchviz_model-code-author` skill when creating, editing, reviewing, or normalizing model implementation files, including canonical notebook-source files under `app/model-notebooks/*.py` and their generated UI source map under `app/generated/model-sources.ts`.

Edit `app/model-notebooks/*.py` as the source of truth. Do not hand-edit `app/generated/model-sources.ts` or `public/notebooks/*.ipynb`; regenerate them with `pnpm generate:model-artifacts`.

## Model Route SEO

When adding, renaming, or removing a model in `app/model-arch-viz-app.tsx`, update `app/model-routes.ts` in the same change so `/models/[modelId]` routes and `app/sitemap.ts` stay aligned.

Each model catalog entry must include the canonical model id, label, published date, route title and description, and paper metadata. Add `sourceBaseName` only when the canonical source basename differs from the model id. Do not add model-specific URLs directly to `app/sitemap.ts`; keep sitemap coverage derived from `modelCatalog`.

## Architecture Code Highlights

When editing architecture node `codeLines` in `app/model-arch-viz-app.tsx`, keep highlights tightly scoped to the code that directly implements or consumes the selected concept.

For input nodes such as sequence/image/token inputs, highlight concrete tensor ingress or consumption lines, such as example tensor construction, model invocation, embedding lookup, or timestep/patch/token slicing. Do not include broad container scaffolding such as `forward` declarations, batch-size extraction, loop setup, or state initialization unless that scaffolding is the selected architecture concept.

Every architecture node must define both `codeLines` and `jaxCodeLines`; never reuse one language's positions for the other. Keep both mappings non-empty and within the generated preview source bounds so catalog validation succeeds during the build.

<!-- BEGIN:nextjs-agent-rules -->

# This is NOT the Next.js you know

This version has breaking changes — APIs, conventions, and file structure may all differ from your training data. Read the relevant guide in `node_modules/next/dist/docs/` (resolved from this file's directory; in monorepos the `next` package may not be visible from the repo root) before writing any code. Heed deprecation notices.

This block is written and re-added by `next dev` — verify at `node_modules/next/dist/server/lib/generate-agent-files.js`. Removing it from a diff only re-creates the uncommitted change; committing it with your work keeps the tree clean.

<!-- END:nextjs-agent-rules -->

## Secrets

`pnpm dev --port auto` fetches only the linked Infisical development project through the saved human CLI login. Private credentials must stay out of dotenv files, command arguments, logs, and browser bundles. Use `pnpm secrets:check` for presence-only verification. Production uses the isolated `modelarchviz-production` project and GitHub Actions delivery to Vercel Production. A standalone Cloudflare relay dispatches the pinned shared delivery workflow; see `docs/production-delivery.md`. Preserve Keychain originals until migration and provider rotation are verified.

New captures without an explicit destination use `$private-skills_inbox-capture` with its
Obsidian default. A request addressed to this repository selects repository mode.

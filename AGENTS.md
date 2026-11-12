# ModelArchViz

## Browser Testing

Follow the global Browser Use and Testing policy for browser/profile selection, runtime setup, and availability.

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

When specifications exist, use `$private-skills_specs-author` for substantive work and
specification changes. Follow its shared read/reuse cadence, provisional-scenario
guidance, proportional detail, decision-based approval, and closing intent check.
The maintained procedure is `/Users/tsilva/.codex/skills/private-skills/private-skills_specs-author/SKILL.md`.

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

Use `pnpm dev:vault --port auto` for agent-operated local AgentBridge work as mapped below. The existing `pnpm dev --port auto` command retains the linked Infisical development injection flow; preserving it does not authorize agents to use raw injection. Private credentials must stay out of dotenv files, command arguments, logs, and browser bundles. Production uses the isolated `modelarchviz-production` project and GitHub Actions delivery to Vercel Production. A standalone Cloudflare relay dispatches the pinned shared delivery workflow; see `docs/production-delivery.md`. Preserve Keychain originals until migration and provider rotation are verified.

New captures without an explicit destination use `$private-skills_inbox-capture` with its
Obsidian default. A request addressed to this repository selects repository mode.

<!-- agent-vault-agentbridge:read:start -->
## Actual local AgentBridge calls

Use `pnpm dev:vault --port auto` for the local app's real `/api/chat` requests.
Reuse an existing verified vault-launched server when available; otherwise report
the printed automatic loopback URL. Never stop another server or choose a fixed
port after automatic startup fails. Missing bindings and failures stop without
raw injection or direct-provider fallback.

The mapping is bundle `local-modelarchviz-agentbridge`, service
`modelarchviz-agentbridge-chat`, on `127.0.0.1:8082`, allowing GET
`/api/v1/models` and POST `/api/v1/chat/completions`. Its owner-only metadata profile
is `~/.config/infisical/agentbridge-clients/modelarchviz-app.json`.
`pnpm agentbridge:app-check` performs only a model-catalog GET through that bundle.

The launcher uses the reviewed side-by-side Infisical CLI 0.45.6 and explicitly
selects the Agent Vault project. Protected metadata pins the native binding,
auth-critical launcher/client/sandbox files, chat backend/route, artifact generation,
Next/Sentry configuration, manifest/lock, Node, CLI and public CA. Changed pins
require trusted review. Serving stops before session creation when installed
Next/React versions differ from the reviewed manifest. Refresh only the locked
environment after checking package-age gates. Run
`python3 -I -S -B scripts/infisical/test_agentbridge_profile.py` alongside the
transport tests. See [local Vault maintenance](docs/local-agent-vault.md). The app receives only `agentbridge_public_placeholder` and
a scoped proxy capability. No source application or upstream key is retrieved;
AgentBridge's existing upstream route handles provider credentials.

This opt-in command uses Webpack for the tested sandbox compatibility and runs
the existing `generate:model-artifacts` script before Next starts. Only generated
model artifacts/notebooks, the PDF worker, Next output and next-env types are
writable. No model implementation or notebook is executed. Dotenv and home
credential files, Keychain lookup and credential-helper execution are blocked.
Node workers inherit the proxy/placeholder guard; outbound connections can reach
only the local proxy. Fetch accepts only the reviewed model/chat paths and marker,
rejects redirects, and withholds provider errors without retry or fallback.

ModelArchViz's local model is explicitly set to `openai/gpt-6.1-terra`, routed as
`openrouter/openai/gpt-6.1-terra`. Its existing JSON chat/code-highlight contract,
900-token ceiling, 60-second request timeout and hosted production behavior remain
unchanged. The normal `dev`, `build` and `start` commands, production delivery and
Sentry configuration are unchanged. App sessions last at most 30 minutes; stop
the owned server normally when finished. Starting it does not authorize arbitrary
prompts or uploads. Live inference requires the usual outbound draft approval.
The trusted launcher and full-access host processes remain outside the child
sandbox boundary. No global trust, environment or app launcher is changed.

## Read-only AgentBridge check

For local agent-operated model-catalog checks, use `pnpm agentbridge:check`.
The reviewed mapping is Agent Vault bundle `local-modelarchviz-agentbridge-read`,
service `modelarchviz-agentbridge-models`, for GET `/api/v1/models` on
`127.0.0.1:8082`. The owner-only local metadata profile is
`~/.config/infisical/agentbridge-clients/modelarchviz-read.json`.

The profile validates this repository's native Development source identity and
pins the existing routing helper's digest. No application key or Sentry token
is fetched. The local gateway accepts public placeholders; AgentBridge's own
upstream credential route remains responsible for provider authentication.
The sandboxed Node client imports the reviewed app routing helper, requires
`agentbridge_public_placeholder`, and enables proxy support explicitly.
Its fixed request returns only catalog size and whether the configured default
is advertised. A missing catalog entry is informational; a catalog does not
prove that an inference request will succeed or that upstream login is valid.

Outbound access is limited to the local proxy. Home reads are limited to the
reviewed runtime, public CA and required client/config files. Only the pinned
Node runtime can be executed inside the sandbox; Keychain service lookup is
blocked. The trusted launcher and other full-access host processes remain
outside that isolation boundary. The instance and CA are pinned without
changing system trust, and sessions last at most ten minutes.

The separate model-only profile permits only read-only catalog access. Use
`dev:vault` above for actual local chat. Other image-generation, embedding, file
and MCP endpoints are not enabled. Missing/changed bindings and failed checks
stop without raw-key or direct-network fallback. Other app defaults,
production delivery, Sentry settings/uploads and other service profiles remain
unchanged. Run `pnpm test:agentbridge` after modifying this client or profile.
<!-- agent-vault-agentbridge:read:end -->

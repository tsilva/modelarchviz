# Global Codex Rules

## Browser Testing

Always use the official OpenAI Browser Use plugin for browser testing unless explicitly overridden by the user prompt, a skill, or another more specific instruction.

## Git Branching

Do not create new branches by default. Work on the current branch unless the user explicitly asks you to create or switch branches; when branching is appropriate, the user will create the branch.

## Product Specifications

Before every task in this repository, use the `$specs-author` skill to read the entire root `SPECS.md`. Before finishing, reread it and check the task and conversation for new or changed stakeholder intent.

- Treat `SPECS.md` as the persistent source of stakeholder requirements that cannot be inferred reliably from code or remembered conversations.
- Apply the scope test to proposed and existing requirements: root `SPECS.md` contains only project-wide intent; scoped intent belongs in its nearest authoritative specification and must not be broadened to fit the root.
- If the task, repository, or user request contradicts, omits, or ambiguously interprets the specification, tell the user. Continue safe exploration and work that does not depend on resolving the issue, but never silently choose an interpretation.
- Never edit `SPECS.md` from inference. Propose the exact change, explain why it reflects stakeholder intent, and edit the file only after the user explicitly approves that exact change.
- Keep `SPECS.md` complete, concise, and compacted. It must contain stakeholder intent rather than implementation, architecture, operations, or transient project detail.

## Model File Code Style

Use the `$model-code-author` skill when creating, editing, reviewing, or normalizing model implementation files, including canonical notebook-source files under `app/model-notebooks/*.py` and their generated UI source map under `app/generated/model-sources.ts`.

Edit `app/model-notebooks/*.py` as the source of truth. Do not hand-edit `app/generated/model-sources.ts` or `public/notebooks/*.ipynb`; regenerate them with `pnpm generate:model-artifacts`.

## Model Route SEO

When adding, renaming, or removing a model in `app/model-arch-viz-app.tsx`, update `app/model-routes.ts` in the same change so `/models/[modelId]` routes and `app/sitemap.ts` stay aligned.

Each model catalog entry must include the canonical model id, label, published date, route title and description, and paper metadata. Add `sourceBaseName` only when the canonical source basename differs from the model id. Do not add model-specific URLs directly to `app/sitemap.ts`; keep sitemap coverage derived from `modelCatalog`.

## Architecture Code Highlights

When editing architecture node `codeLines` in `app/model-arch-viz-app.tsx`, keep highlights tightly scoped to the code that directly implements or consumes the selected concept.

For input nodes such as sequence/image/token inputs, highlight concrete tensor ingress or consumption lines, such as example tensor construction, model invocation, embedding lookup, or timestep/patch/token slicing. Do not include broad container scaffolding such as `forward` declarations, batch-size extraction, loop setup, or state initialization unless that scaffolding is the selected architecture concept.

Every architecture node must define both `codeLines` and `jaxCodeLines`; never reuse one language's positions for the other. Keep both mappings non-empty and within the generated preview source bounds so catalog validation succeeds during the build.

## Skill Retrospective

Whenever a skill is used for a task:

1. After completing the main task, and before handing control back to the user, review how the skill performed during the task.
2. Consider whether the skill could be changed so that a future similar trajectory accomplishes the same goal using fewer tokens, and/or accomplishes other closely related goals while still minimizing tokens. By default, reducing tokens means avoiding repeated mistakes, incorrect rabbit holes, and commands that, in hindsight, would never work, not merely shortening the wording of the response.
3. If there is a meaningful optimization to suggest under that definition, tell the user briefly and ask whether they want the skill updated.
4. Only update the skill if the user explicitly says yes.
5. If there is no meaningful skill improvement to suggest, do not add unnecessary commentary about the retrospective.

## JavaScript Supply Chain Protection

When creating or modifying JavaScript/TypeScript projects, keep npm supply-chain hardening in place:

- Prefer pnpm for new projects.
- For pnpm projects, configure `minimumReleaseAge: 10080` and `blockExoticSubdeps: true` in `pnpm-workspace.yaml`; for pnpm 10 compatibility, also set `minimum-release-age=10080` and `block-exotic-subdeps=true` in `.npmrc`.
- For npm projects, set `min-release-age=10080`; if dependency lifecycle scripts are not explicitly required, set `ignore-scripts=true`.
- Do not add Git, tarball, or other exotic dependency sources, or package versions published within the last 7 days, unless the user explicitly approves the risk.

## Python Supply Chain Protection

When creating or modifying Python projects, keep Python package supply-chain hardening in place:

- Prefer `uv` with a committed `uv.lock` for new Python projects.
- Configure `exclude-newer = "7 days"` for `uv` projects and keep known-bad package constraints such as `mistralai!=2.4.6` and `guardrails-ai!=0.10.1` in `[tool.uv].constraint-dependencies` when relevant.
- For pip requirements projects, use a checked-in `constraints.txt` and reference it from requirement files with `-c constraints.txt`.
- Do not add direct URL, Git, local path, or alternate-index Python dependencies unless the user explicitly approves the risk.

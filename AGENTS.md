# Global Codex Rules

## Browser Testing

Always use the official OpenAI Browser Use plugin for browser testing unless explicitly overridden by the user prompt, a skill, or another more specific instruction.

## Git Branching

Do not create new branches by default. Work on the current branch unless the user explicitly asks you to create or switch branches; when branching is appropriate, the user will create the branch.

## Model File Code Style

Use the `$model-code-author` skill when creating, editing, reviewing, or normalizing model implementation files, including canonical notebook-source files under `app/model-notebooks/*.py` and generated UI sources under `app/generated/model-code/*.py`.

Edit `app/model-notebooks/*.py` as the source of truth. Do not hand-edit `app/generated/model-code/*.py` or `public/notebooks/*.ipynb`; regenerate them with `pnpm generate:model-artifacts`.

Structure canonical notebook sources for both site previews and notebook consumption. Use `# %%` cells for imports and implementation blocks, especially one top-level class or function per cell where practical.

Use `# %% [notebook-only]` for example construction, smoke-test, shape-inspection, or tiny training cells that should appear in generated notebooks but be excluded from generated site preview code.

For major model components, prefer a class or function implementation cell followed by a small `# %% [notebook-only]` cell that constructs the component and verifies a representative output shape.

For model implementation files such as `mlp.py`, avoid performing multiple operations in a single line. Split indexing, function calls, arithmetic, assignments, and returns into separate named steps where practical.

Add an empty line before each standalone line comment.

Do not force an empty line after a standalone line comment.

Do not add an empty line between a method signature and the first explanatory comment inside that method.

When several consecutive code lines form one logical block, keep them together and use one block-level comment.

Prefer one comment per logical block, especially when the block keeps tensor shapes unchanged or changes shape only once.

Use multiline method signatures when parameter comments are useful.

Put parameter explanations as inline comments on the parameter lines.

Represent tensor shapes in comments with parentheses, for example `(batch, steps, features)`, not square brackets.

Do not add comments above import statements.

Do not add comments above base initializer calls such as `super().__init__()`.

## Model Route SEO

When adding, renaming, or removing a model in `app/model-arch-viz-app.tsx`, update `app/model-routes.ts` in the same change so `/models/[modelId]` routes and `app/sitemap.ts` stay aligned.

Each model route entry must include the canonical model id, label, published date, route title, and route description. Do not add model-specific URLs directly to `app/sitemap.ts`; keep sitemap coverage derived from `modelRouteSummaries`.

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

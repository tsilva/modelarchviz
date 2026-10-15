# Production delivery

Push to `main` triggers `.github/workflows/production-delivery.yml`. It tests delivery boundaries and waits for the same commit’s CI before retrieving secrets. Then it syncs the fixed production allowlist in `.github/scripts/production-delivery.json`, deploys that exact commit, and verifies the resulting deployment. Vercel’s automatic Git deployments are disabled so they cannot race the sync.

Infisical remains the source of application secrets. Local development uses the existing human CLI login and Development environment. Production uses one shared GitHub reader with Viewer access to the assigned production vaults. GitHub OIDC issues a 15-minute read-only session only for approved repositories on main, in the Production environment and delivery workflow. Each repository’s script pins its own source vault and key allowlist; the shared identity itself can read all assigned production vaults. The provider credential is a project-scoped `VERCEL_TOKEN` stored in the repository’s GitHub Production environment. No general CLI login is copied into GitHub.

The workflow does not run on pull requests or forks, export `.env` files, upload secret artifacts, delete destination variables, or change preview/development variables. All required source keys and destination targets are checked before writes. Failed CI or sync prevents deployment. Existing production remains live on failure.

An Infisical HTTP failure identifies the required production key and status code while suppressing provider responses and credentials. Check that key in the pinned production vault and the reader's access before retrying delivery.

After changing a production secret, use **Actions → Production delivery → Run workflow → main** to sync and deploy. A secret change alone does not trigger delivery. There is no schedule or automatic retry; rerun a failed workflow manually after fixing the cause. For an old failed commit superseded by a newer main commit, run a fresh workflow on main. Deployment tokens expire after one year and need rotation before expiry.

Native Infisical syncs are disabled after this workflow has successfully replaced them. Updating a deployment credential in GitHub does not replace application keys in Infisical.

Production chat calls OpenRouter directly using the production vault's `OPENROUTER_API_KEY`. AgentBridge is used only for local development and is not in the production delivery allowlist. The default production model is `deepseek/deepseek-v4.1-flash`, with reasoning disabled and a 900-token response limit; `OPENROUTER_MODEL` can override it in the deployment environment. Vercel previews also require their own OpenRouter key, configured separately; this workflow does not modify preview variables.

A missing production key makes chat return HTTP 503 without falling back to a local gateway. Keep production credentials in the isolated `modelarchviz-production` vault and use the existing OIDC delivery flow to sync them to Vercel. No development credential is copied into production.

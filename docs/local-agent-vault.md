# Local Vault maintenance

The opt-in `pnpm dev:vault --port auto` command runs the local app with a public
placeholder, a bounded native Agent Vault session and the reviewed OS sandbox.
`pnpm agentbridge:check` and `pnpm agentbridge:app-check` only read the model catalog.
They do not prove inference or upstream authentication.

Both protected profiles require CLI 0.45.6 and explicitly select the reviewed Vault
project. Profiles pin the native source binding, routing helper, chat backend/route,
launcher, transport, sandbox, artifact preparation, Next/Sentry configuration,
manifest/lock, Node, CLI and public CA. A change to these files stops the launcher
until a trusted administrator reviews and updates its exact pins. Profiles cannot
add arbitrary commands, scopes or alternate source folders.

Before serving, installed Next, React and React DOM must match the reviewed
manifest. Check the complete locked dependency graph against the package-age and
integrity policy before installing with `pnpm install --frozen-lockfile --ignore-scripts`.
Do not weaken the age policy or enable lifecycle scripts to fix a startup failure.
The startup preparation parses model notebook sources as text; it does not execute
the notebooks or install their Python requirements.

Run these credential-free regression checks after changing the launcher:

```sh
python3 -I -S -B scripts/infisical/test_agentbridge_profile.py
pnpm test:agentbridge
```

New route/client verification uses a disposable fake upstream first, exercising
the actual API handler, placeholder replacement, rejected-error filtering, request
restrictions and sandbox denials. Follow it with the bounded catalog checks. A
real chat requires its own approved prompt; do not silently retry or switch models.

Dotenv reads are deliberately denied by the local sandbox, so Next may warn about
an existing dotenv file. Do not grant access to that file to silence the warning.
Existing app/model defaults, hosted delivery and Sentry settings remain unchanged.
The trusted launcher and other full-access processes running as the same OS user
remain outside the child sandbox boundary.

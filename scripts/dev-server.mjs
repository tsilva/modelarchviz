#!/usr/bin/env node

import { spawn } from "node:child_process";
import { createServer } from "node:net";

async function randomAvailablePort() {
  const server = createServer();

  return await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();
      server.close((error) => {
        if (error) {
          reject(error);
          return;
        }

        if (!address || typeof address === "string") {
          reject(new Error("Could not resolve random server port"));
          return;
        }

        resolve(address.port);
      });
    });
  });
}

async function resolveAutoPortArgs(args) {
  const nextArgs = [...args];

  for (let index = 0; index < nextArgs.length; index += 1) {
    const arg = nextArgs[index];

    if ((arg === "--port" || arg === "-p") && nextArgs[index + 1] === "auto") {
      const port = await randomAvailablePort();
      nextArgs[index + 1] = String(port);
      console.error(`Using random available port ${port}`);
      return nextArgs;
    }

    if (arg === "--port=auto" || arg === "-p=auto") {
      const port = await randomAvailablePort();
      const [flag] = arg.split("=");
      nextArgs[index] = `${flag}=${port}`;
      console.error(`Using random available port ${port}`);
      return nextArgs;
    }
  }

  return nextArgs;
}

async function main() {
  const [mode, ...rawArgs] = process.argv.slice(2);

  if (mode !== "dev" && mode !== "start") {
    console.error("Usage: node scripts/dev-server.mjs <dev|start> [...args]");
    process.exit(1);
  }

  const nextArgs = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;
  const args = await resolveAutoPortArgs([mode, ...nextArgs]);
  const child = spawn("next", args, {
    env: process.env,
    stdio: "inherit",
    shell: process.platform === "win32",
  });

  for (const signal of ["SIGINT", "SIGTERM"]) {
    process.on(signal, () => {
      child.kill(signal);
    });
  }

  child.on("exit", (code, signal) => {
    if (signal) {
      process.kill(process.pid, signal);
      return;
    }

    process.exit(code ?? 1);
  });

  child.on("error", (error) => {
    console.error(error);
    process.exit(1);
  });
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});

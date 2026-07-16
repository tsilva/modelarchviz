#!/usr/bin/env node

import { spawn } from "node:child_process";

function resolveAutoPortArgs(args) {
  const nextArgs = [...args];

  for (let index = 0; index < nextArgs.length; index += 1) {
    const arg = nextArgs[index];

    if ((arg === "--port" || arg === "-p") && nextArgs[index + 1] === "auto") {
      nextArgs[index + 1] = "0";
      return nextArgs;
    }

    if (arg === "--port=auto" || arg === "-p=auto") {
      const [flag] = arg.split("=");
      nextArgs[index] = `${flag}=0`;
      return nextArgs;
    }
  }

  return nextArgs;
}

async function main() {
  const rawArgs = process.argv.slice(2);
  const nextArgs = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;
  const args = resolveAutoPortArgs(["dev", ...nextArgs]);
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

#!/usr/bin/env node

import { spawn } from "node:child_process";
import { pathToFileURL } from "node:url";

export function resolveDevArgs(args) {
  const nextArgs = [...args];
  let hasExplicitPort = false;

  for (let index = 0; index < nextArgs.length; index += 1) {
    const arg = nextArgs[index];

    if (arg === "--port" || arg === "-p") {
      hasExplicitPort = true;

      if (nextArgs[index + 1] === "auto") {
        nextArgs[index + 1] = "0";
      }

      continue;
    }

    if (arg.startsWith("--port=") || arg.startsWith("-p=")) {
      hasExplicitPort = true;

      if (arg.endsWith("=auto")) {
        const [flag] = arg.split("=");
        nextArgs[index] = `${flag}=0`;
      }
    }
  }

  return hasExplicitPort ? nextArgs : [...nextArgs, "--port", "0"];
}

async function main() {
  const rawArgs = process.argv.slice(2);
  const nextArgs = rawArgs[0] === "--" ? rawArgs.slice(1) : rawArgs;
  const args = resolveDevArgs(["dev", ...nextArgs]);
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

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((error) => {
    console.error(error);
    process.exit(1);
  });
}

import net from "node:net";
import { spawn } from "node:child_process";

function resolveAutoPort(args) {
  const nextArgs = args[0] === "--" ? args.slice(1) : args;
  const portIndex = nextArgs.findIndex((arg) => arg === "--port" || arg === "-p");

  if (portIndex !== -1 && nextArgs[portIndex + 1] === "auto") {
    return resolveFreePort().then((port) => {
      const resolvedArgs = [...nextArgs];
      resolvedArgs[portIndex + 1] = String(port);
      return resolvedArgs;
    });
  }

  const equalsIndex = nextArgs.findIndex(
    (arg) => arg === "--port=auto" || arg === "-p=auto",
  );

  if (equalsIndex !== -1) {
    return resolveFreePort().then((port) => {
      const resolvedArgs = [...nextArgs];
      const [flag] = resolvedArgs[equalsIndex].split("=");
      resolvedArgs[equalsIndex] = `${flag}=${port}`;
      return resolvedArgs;
    });
  }

  return Promise.resolve(nextArgs);
}

function resolveFreePort() {
  return new Promise((resolve, reject) => {
    const server = net.createServer();

    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const address = server.address();

      if (!address || typeof address === "string") {
        server.close(() => reject(new Error("Could not resolve a free port.")));
        return;
      }

      const { port } = address;
      server.close(() => resolve(port));
    });
  });
}

const cliArgs = await resolveAutoPort(process.argv.slice(2));
const nextProcess = spawn("next", ["dev", ...cliArgs], {
  env: process.env,
  stdio: "inherit",
});

nextProcess.on("exit", (code, signal) => {
  if (signal) {
    process.kill(process.pid, signal);
    return;
  }

  process.exit(code ?? 0);
});

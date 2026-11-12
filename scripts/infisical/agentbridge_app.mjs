import { createRequire } from 'node:module';
import { spawnSync } from 'node:child_process';
import { pathToFileURL } from 'node:url';
import { validateSession } from './agentbridge_client.mjs';

validateSession();
if (process.argv.slice(2).join(' ') !== '--port auto') throw new Error('Only --port auto is supported.');
const adapter = await import(pathToFileURL(process.env.AGENTBRIDGE_APP_CONFIG).href);
if (adapter.agentBridgeBaseUrl() !== process.env.AGENTBRIDGE_BASE_URL
    || adapter.agentBridgeModel(process.env.AGENTBRIDGE_DEFAULT_MODEL) !== 'openrouter/openai/gpt-6.1-terra') {
  throw new Error('Unexpected local application routing.');
}
// Preserve the project's normal generated-source, notebook and PDF-worker preparation.
const prepare = spawnSync(process.execPath, [new URL('../generate-model-artifacts.mjs', import.meta.url).pathname], {
  stdio: 'inherit', env: process.env,
});
if (prepare.status !== 0) throw new Error('Local artifact preparation failed.');
const require = createRequire(import.meta.url);
const cli = require.resolve('next/dist/bin/next');
// The opt-in Webpack launch avoids Turbopack compiler sockets outside the proxy.
// Port 0 lets the OS select the port; Next prints the actual loopback URL.
process.argv = [process.execPath, cli, 'dev', '--webpack', '--hostname', '127.0.0.1', '--port', '0'];
await import(pathToFileURL(cli).href);

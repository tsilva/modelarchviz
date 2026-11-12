/** Synthetic-only native probe: imports the real API handler without a dev server. */
import assert from 'node:assert/strict';
import { createRequire, registerHooks } from 'node:module';
import { existsSync, readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { fileURLToPath, pathToFileURL } from 'node:url';
import http from 'node:http';
import net from 'node:net';
import { validateSession } from '../agentbridge_client.mjs';

const root = fileURLToPath(new URL('../../../', import.meta.url));
const require = createRequire(new URL('../../../package.json', import.meta.url));
const nextServer = pathToFileURL(require.resolve('next/server')).href;
registerHooks({ resolve(specifier, context, next) {
  if (specifier === 'next/server') return next(nextServer, context);
  let base;
  if (specifier.startsWith('@/')) base = pathToFileURL(root + 'app/' + specifier.slice(2));
  else if (specifier.startsWith('.') && context.parentURL?.startsWith(pathToFileURL(root + 'app/').href))
    base = new URL(specifier, context.parentURL);
  if (base) {
    for (const suffix of ['', '.ts', '.mts', '.tsx', '/index.ts']) {
      const candidate = base.href + suffix;
      if (existsSync(fileURLToPath(candidate))) return next(candidate, context);
    }
  }
  return next(specifier, context);
} });

function proxyStatus(method, path) {
  const proxy = new URL(process.env.HTTP_PROXY);
  const destination = new URL(validateSession());
  return new Promise((resolve, reject) => {
    const request = http.request({ hostname: proxy.hostname, port: proxy.port,
      method, path: destination.origin + path, agent: false,
      headers: { Host: destination.host,
        'Proxy-Authorization': 'Basic ' + Buffer.from(decodeURIComponent(proxy.username) + ':' + decodeURIComponent(proxy.password)).toString('base64'),
        Authorization: 'Bearer agentbridge_public_placeholder' } }, response => {
      response.resume(); response.once('end', () => resolve(response.statusCode));
    });
    request.once('error', reject);request.setTimeout(10000, () => request.destroy(new Error('Fixture timeout')));
    request.end();
  });
}

let stage = 'session';
try {
  const base = validateSession();
  stage = 'handler-import';
  const { POST } = await import(new URL('../../../app/api/chat/route.ts', import.meta.url));
  const body = { messages: [{ role: 'user', content: 'Synthetic Vault transport fixture.' }] };
  const request = data => new Request('http://localhost/api/chat', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(data),
  });
  stage = 'normal-chat';
  const normal = await POST(request(body));
  assert.equal(normal.status, 200);
  assert.equal((await normal.json()).message, 'Synthetic reply');
  stage = 'invalid-input';
  assert.equal((await POST(request({ messages: [] }))).status, 400);
  stage = 'reflected-error';
  const failed = await POST(request({ messages: [{ role: 'user', content: 'Synthetic rejection fixture.' }] }));
  assert.equal(failed.status, 502);
  assert.doesNotMatch(await failed.text(), /reflected-fake-key-/);
  stage = 'transport-guard';
  await assert.rejects(fetch(base + '/chat/completions', { method: 'POST',
    headers: { Authorization: 'Bearer wrong-marker' }, body: '{}' }));
  await assert.rejects(fetch(base + '/openrouter/key', {
    headers: { Authorization: 'Bearer agentbridge_public_placeholder' } }));
  stage = 'native-proxy-policy';
  assert.equal(await proxyStatus('DELETE', '/api/v1/models'), 403);
  assert.equal(await proxyStatus('GET', '/api/v1/openrouter/key'), 403);
  stage = 'sandbox';
  assert.throws(() => readFileSync(process.env.FIXTURE_PRIVATE_CANARY),
    error => ['EPERM', 'EACCES'].includes(error.code));
  assert.ok(['EPERM', 'EACCES'].includes(spawnSync(process.env.FIXTURE_CLI, ['--version']).error?.code));
  const direct = await new Promise(resolve => {
    const socket = net.connect(Number(new URL(base).port), '127.0.0.1');
    socket.once('error', error => resolve(['EPERM', 'EACCES'].includes(error.code)));
    socket.once('connect', () => { socket.destroy(); resolve(false); });
    socket.setTimeout(2000, () => { socket.destroy(); resolve(false); });
  });
  assert.equal(direct, true);
  console.log(JSON.stringify({ ok: true, fake_only: true,
    checks: ['actual-api-json', 'invalid-input-not-dispatched', 'reflected-error-withheld',
      'wrong-marker-denied', 'management-path-denied', 'native-delete-denied', 'native-management-denied',
      'private-file-denied', 'credential-helper-exec-denied', 'direct-network-denied'],
    real_prompts_sent: 0, server_started: false }));
} catch (error) {
  const code = ['ERR_MODULE_NOT_FOUND', 'ERR_UNSUPPORTED_DIR_IMPORT', 'ERR_UNKNOWN_FILE_EXTENSION',
    'ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX', 'EACCES', 'EPERM', 'ENOENT', 'ERR_REQUIRE_CYCLE_MODULE'].includes(error.code)
    ? error.code : 'withheld';
  const type = ['SyntaxError', 'ReferenceError', 'TypeError', 'Error'].includes(error.name) ? error.name : 'withheld';
  const importSignals = stage === 'handler-import' ? {
    named_export: /export|named/i.test(error.message),
    module_resolution: /module|package|resolution/i.test(error.message),
    directory: /directory/i.test(error.message),
    permission: /permission|permitted|access/i.test(error.message),
    typescript: /typescript|annotation|type syntax/i.test(error.message),
  } : undefined;
  console.log(JSON.stringify({ ok: false, stage, code, type, importSignals, reason: 'Synthetic app probe failed; details withheld' }));
  process.exitCode = 1;
}

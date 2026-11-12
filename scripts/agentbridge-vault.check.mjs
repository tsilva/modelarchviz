// Explicit Node runner; the .check name avoids Vitest suite discovery.
import test from 'node:test';
import assert from 'node:assert/strict';
import { validateSession } from './infisical/agentbridge_client.mjs';
import { gatewayFetch } from './infisical/agentbridge_transport.mjs';

function fixtureProxy(host = '127.0.0.1', identity = 'fixture-session') {
  const url = new URL(`http://${host}:8283`);
  url.username = identity;
  url.password = identity;
  return url.href;
}

const proxy = fixtureProxy();
const session = {
  NODE_ENV: 'development', NODE_USE_ENV_PROXY: '1',
  AGENTBRIDGE_BASE_URL: 'http://127.0.0.1:8082/api/v1',
  AGENTBRIDGE_API_KEY: 'agentbridge_public_placeholder',
  HTTP_PROXY: proxy, HTTPS_PROXY: proxy, NO_PROXY: '', no_proxy: '',
};

test('catalog client requires only a public placeholder in a development proxy session', () => {
  assert.equal(validateSession(session), session.AGENTBRIDGE_BASE_URL);
});

test('catalog client rejects source keys, Sentry credentials and hosted environments', () => {
  for (const change of [
    { AGENTBRIDGE_API_KEY: '' }, { AGENTBRIDGE_API_KEY: 'raw-fixture-key' },
    { OPENROUTER_API_KEY: 'upstream-fixture-key' }, { SENTRY_AUTH_TOKEN: 'sentry-fixture-key' },
    { NODE_ENV: 'production' }, { VERCEL_ENV: 'preview' }, { NODE_USE_ENV_PROXY: '0' },
    { NODE_TLS_REJECT_UNAUTHORIZED: '0' },
  ]) assert.throws(() => validateSession({ ...session, ...change }));
});

test('catalog client rejects bypasses, missing proxy authentication and conflicting sessions', () => {
  for (const change of [
    { NO_PROXY: '*' }, { no_proxy: '127.0.0.1' }, { HTTP_PROXY: '' },
    { HTTP_PROXY: 'http://127.0.0.1:8283' },
    { HTTP_PROXY: fixtureProxy('unreviewed.invalid') },
    { HTTPS_PROXY: fixtureProxy('127.0.0.1', 'other') },
    { http_proxy: fixtureProxy('127.0.0.1', 'other') },
    { https_proxy: fixtureProxy('127.0.0.1', 'other') },
  ]) assert.throws(() => validateSession({ ...session, ...change }));
});

test('catalog client rejects unreviewed gateway origins and URL modifications', () => {
  for (const base of [
    'https://openrouter.ai/api/v1', 'http://remote.invalid/api/v1',
    'http://user:password@127.0.0.1:8082/api/v1',
    'http://127.0.0.1:8082/dashboard', 'http://127.0.0.1:8082/api/v1?override=yes',
    'http://127.0.0.1:8082/api/v1#override',
  ]) assert.throws(() => validateSession({ ...session, AGENTBRIDGE_BASE_URL: base }));
});

test('app transport preserves JSON requests and streaming replies through the guarded fetch', async () => {
  let calls = 0;
  const guarded = gatewayFetch(async (url, init) => {
    calls++;
    assert.equal(url, `${session.AGENTBRIDGE_BASE_URL}/chat/completions`);
    assert.equal(init.redirect, 'error');
    assert.equal(init.body, '{"stream":true}');
    return new Response('data: synthetic\n\n', { headers: { 'Content-Type': 'text/event-stream' } });
  }, session.AGENTBRIDGE_BASE_URL, () => validateSession(session));
  const response = await guarded(`${session.AGENTBRIDGE_BASE_URL}/chat/completions`, {
    method: 'POST', headers: { Authorization: 'Bearer agentbridge_public_placeholder' }, body: '{"stream":true}',
  });
  assert.equal(await response.text(), 'data: synthetic\n\n');
  assert.equal(calls, 1);
});

test('app transport withholds rejected provider bodies without retrying or falling back', async () => {
  let calls = 0;
  const guarded = gatewayFetch(async () => {
    calls++;
    return new Response('{"error":{"message":"reflected-fake-upstream-key"}}', { status: 401 });
  }, session.AGENTBRIDGE_BASE_URL, () => validateSession(session));
  const response = await guarded(`${session.AGENTBRIDGE_BASE_URL}/models`, {
    headers: { Authorization: 'Bearer agentbridge_public_placeholder' },
  });
  assert.equal(response.status, 401);
  assert.doesNotMatch(await response.text(), /reflected-fake-upstream-key/);
  assert.equal(calls, 1);
});

test('app transport blocks alternate hosts, credentials, methods and management endpoints before fetch', async () => {
  let calls = 0;
  const guarded = gatewayFetch(async () => { calls++; return new Response('unexpected'); },
    session.AGENTBRIDGE_BASE_URL, () => validateSession(session));
  for (const [url, method, key] of [
    ['https://openrouter.ai/api/v1/chat/completions', 'POST', 'agentbridge_public_placeholder'],
    ['http://127.0.0.1:8082/dashboard', 'GET', 'agentbridge_public_placeholder'],
    [`${session.AGENTBRIDGE_BASE_URL}/models?override=1`, 'GET', 'agentbridge_public_placeholder'],
    [`${session.AGENTBRIDGE_BASE_URL}/models`, 'DELETE', 'agentbridge_public_placeholder'],
    [`${session.AGENTBRIDGE_BASE_URL}/chat/completions`, 'POST', 'raw-fixture-key'],
  ]) await assert.rejects(guarded(url, { method, headers: { Authorization: `Bearer ${key}` } }));
  assert.equal(calls, 0);
});

test('app transport withholds network diagnostics without retrying', async () => {
  let calls = 0;
  const guarded = gatewayFetch(async () => { calls++; throw new Error('fixture-proxy-credential'); },
    session.AGENTBRIDGE_BASE_URL, () => validateSession(session));
  await assert.rejects(guarded(`${session.AGENTBRIDGE_BASE_URL}/models`, {
    headers: { Authorization: 'Bearer agentbridge_public_placeholder' },
  }), error => error.message === 'AgentBridge connection failed; diagnostic details withheld.' && !error.cause);
  assert.equal(calls, 1);
});

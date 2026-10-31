import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import test from "node:test";

const require = createRequire(import.meta.url);
const ts = require("typescript");
function evaluate(relativePath, dependencies = {}) {
  const file = new URL(relativePath, import.meta.url);
  const source = ts.transpileModule(readFileSync(file, "utf8"), {
    fileName: file.pathname,
    compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.CommonJS },
  }).outputText;
  const module = { exports: {} };
  new Function("require", "module", "exports", source)(
    (name) => Object.hasOwn(dependencies, name) ? dependencies[name] : require(name), module, module.exports,
  );
  return module.exports;
}
const bridge = evaluate("../app/agentbridge.ts");
const backend = evaluate("../app/chat-backend.ts", { "./agentbridge": bridge });
const contract = evaluate("../app/chat-contract.ts");
const route = evaluate("../app/api/chat/route.ts", {
  "../../chat-backend": backend,
  "../../chat-contract": contract,
  "next/server": { NextResponse: { json: (value, init) => Response.json(value, init) } },
});

function productionEnvironment(t, values = {}) {
  const keys = ["NODE_ENV", "VERCEL_ENV", "OPENROUTER_API_KEY", "OPENROUTER_MODEL", "AGENTBRIDGE_API_KEY", "AGENTBRIDGE_BASE_URL"];
  const original = Object.fromEntries(keys.map((key) => [key, process.env[key]]));
  for (const key of keys) delete process.env[key];
  Object.assign(process.env, { NODE_ENV: "production", ...values });
  t.after(() => {
    for (const key of keys) {
      if (original[key] === undefined) delete process.env[key];
      else process.env[key] = original[key];
    }
  });
}
function request() {
  return new Request("https://app.invalid/api/chat", {
    method: "POST", body: JSON.stringify({ messages: [{ role: "user", content: "Explain the pooling layer." }] }),
  });
}
function fetchStub(t, implementation) {
  const original = globalThis.fetch;
  globalThis.fetch = implementation;
  t.after(() => { globalThis.fetch = original; });
}

test("production and previews use their own OpenRouter key, never AgentBridge", () => {
  for (const target of [{ NODE_ENV: "production" }, { VERCEL_ENV: "production" }, { VERCEL_ENV: "preview" }]) {
    const config = backend.chatBackend({ ...target, OPENROUTER_API_KEY: "production-only", AGENTBRIDGE_API_KEY: "development-only", AGENTBRIDGE_BASE_URL: "http://localhost:8082/api/v1" });
    assert.equal(config.baseUrl, "https://openrouter.ai/api/v1");
    assert.equal(config.apiKey, "production-only");
    assert.equal(config.model, "deepseek/deepseek-v4.1-flash-1017");
  }
  assert.throws(() => backend.chatBackend({ NODE_ENV: "production", AGENTBRIDGE_API_KEY: "dev-only" }), /own OpenRouter key/);
});

test("development preserves AgentBridge routing and model prefixes", () => {
  const config = backend.chatBackend({ NODE_ENV: "development", OPENROUTER_API_KEY: "unused", AGENTBRIDGE_API_KEY: "dev-only", OPENROUTER_MODEL: "codex/gpt-6.1-sol" });
  assert.equal(config.baseUrl, "http://127.0.0.1:8082/api/v1");
  assert.equal(config.apiKey, "dev-only");
  assert.equal(config.model, "codex/gpt-6.1-sol");
  assert.equal(backend.chatBackend({ OPENROUTER_MODEL: "deepseek/deepseek-v4.1-flash-1017" }).model, "openrouter/deepseek/deepseek-v4.1-flash-1017");
});

test("production model overrides accept provider IDs and reject gateway IDs", () => {
  assert.equal(backend.chatBackend({ NODE_ENV: "production", OPENROUTER_API_KEY: "test", OPENROUTER_MODEL: "openai/gpt-4o-mini" }).model, "openai/gpt-4o-mini");
  assert.throws(() => backend.chatBackend({ NODE_ENV: "production", OPENROUTER_API_KEY: "test", OPENROUTER_MODEL: "openrouter/deepseek/deepseek-v4.1-flash-1017" }), /OpenRouter model ID/);
});

test("missing production credentials return 503 before any outbound request", async (t) => {
  productionEnvironment(t, { AGENTBRIDGE_API_KEY: "must-not-be-used" });
  fetchStub(t, () => { throw new Error("No outbound request should occur"); });
  const response = await route.POST(request());
  assert.equal(response.status, 503);
  assert.ok((await response.json()).error.includes("unavailable"));
});

test("production chat uses the approved model and returns the assistant contract", async (t) => {
  productionEnvironment(t, { OPENROUTER_API_KEY: "dummy-production-only" });
  fetchStub(t, async (url, options) => {
    assert.equal(url, "https://openrouter.ai/api/v1/chat/completions");
    assert.equal(options.headers.Authorization, "Bearer dummy-production-only");
    const body = JSON.parse(options.body);
    assert.equal(body.model, "deepseek/deepseek-v4.1-flash-1017");
    assert.equal(body.reasoning.enabled, false);
    assert.equal(body.max_tokens, 900);
    assert.deepEqual(body.response_format, { type: "json_object" });
    return Response.json({ choices: [{ message: { content: JSON.stringify({ message: "A pooled embedding.", codeSelection: null }) } }] });
  });
  const response = await route.POST(request());
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { message: "A pooled embedding.", codeSelection: null });
});

test("provider failures do not expose response bodies or credentials", async (t) => {
  productionEnvironment(t, { OPENROUTER_API_KEY: "private-key" });
  fetchStub(t, async () => new Response("private-key and provider details", { status: 401 }));
  const response = await route.POST(request());
  assert.equal(response.status, 502);
  assert.doesNotMatch(await response.text(), /private-key|provider details/);
});

test("network failures and malformed provider JSON return controlled errors", async (t) => {
  productionEnvironment(t, { OPENROUTER_API_KEY: "dummy" });
  fetchStub(t, async () => { throw new Error("private network details"); });
  const unavailable = await route.POST(request());
  assert.equal(unavailable.status, 502);
  assert.doesNotMatch(await unavailable.text(), /private network/);
  globalThis.fetch = async () => new Response("invalid provider JSON");
  const malformed = await route.POST(request());
  assert.equal(malformed.status, 502);
});

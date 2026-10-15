export function agentBridgeBaseUrl(env: Record<string, string | undefined> = process.env): string {
  const value = (env.AGENTBRIDGE_BASE_URL ?? "http://127.0.0.1:8082/api/v1").replace(/\/+$/, "");
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || url.hostname === "openrouter.ai") {
    throw new Error("Configure an AgentBridge HTTP endpoint.");
  }
  return value;
}

export function agentBridgeModel(model: string): string {
  if (/^(codex|claudecode)\//.test(model) || /^openrouter\/[^/]+\/.+/.test(model) || model === "openrouter/default") return model;
  return `openrouter/${model}`;
}

import { agentBridgeBaseUrl, agentBridgeModel } from "./agentbridge";

export const productionChatModel = "deepseek/deepseek-v4.1-flash";

export function chatBackend(env: Record<string, string | undefined> = process.env) {
  const production = env.NODE_ENV === "production"
    || env.VERCEL_ENV === "production"
    || env.VERCEL_ENV === "preview";

  if (production) {
    const apiKey = env.OPENROUTER_API_KEY?.trim();
    if (!apiKey) throw new Error("Production chat requires its own OpenRouter key.");
    const model = env.OPENROUTER_MODEL?.trim() || productionChatModel;
    if (!/^[^/]+\/[^/]+$/.test(model)) {
      throw new Error("Production chat requires an OpenRouter model ID.");
    }
    return {
      name: "OpenRouter",
      baseUrl: "https://openrouter.ai/api/v1",
      apiKey,
      model,
      direct: true,
    };
  }

  return {
    name: "AgentBridge",
    baseUrl: agentBridgeBaseUrl(env),
    apiKey: env.AGENTBRIDGE_API_KEY || "not-needed",
    model: agentBridgeModel(env.OPENROUTER_MODEL || "openai/gpt-4o-mini"),
    direct: false,
  };
}

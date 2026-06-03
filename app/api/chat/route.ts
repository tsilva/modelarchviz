import { NextResponse } from "next/server";

export const runtime = "nodejs";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type SourceLine = {
  lineNumber: number;
  text: string;
};

type ChatContext = {
  model?: {
    id?: string;
    label?: string;
    breadcrumb?: string;
    stats?: string;
  };
  paper?: {
    title?: string;
    authors?: string;
    year?: string;
    venue?: string;
    focus?: string[];
  };
  selection?: {
    id?: string;
    label?: string;
    type?: string;
    kind?: string;
    summary?: string | null;
    badges?: string[];
    codeLines?: number[];
  } | null;
  source?: {
    language?: string;
    fileName?: string;
    code?: string[];
    selectedLines?: SourceLine[];
  };
  searchQuery?: string;
};

const defaultModel = "openai/gpt-4o-mini";
const maxMessages = 16;
const maxSourceCharacters = 24000;

function isChatMessage(value: unknown): value is ChatMessage {
  if (!value || typeof value !== "object") {
    return false;
  }

  const candidate = value as ChatMessage;
  return (
    (candidate.role === "user" || candidate.role === "assistant") &&
    typeof candidate.content === "string" &&
    candidate.content.trim().length > 0
  );
}

function safeString(value: unknown, fallback = "unknown") {
  return typeof value === "string" && value.trim().length > 0 ? value : fallback;
}

function formatSourceLines(lines: SourceLine[] | undefined) {
  if (!Array.isArray(lines) || lines.length === 0) {
    return "No selected source lines.";
  }

  return lines
    .filter((line) => Number.isFinite(line.lineNumber) && typeof line.text === "string")
    .map((line) => `${line.lineNumber}: ${line.text}`)
    .join("\n");
}

function formatContext(context: ChatContext) {
  const model = context.model ?? {};
  const paper = context.paper ?? {};
  const selection = context.selection ?? null;
  const source = context.source ?? {};
  const sourceCode = Array.isArray(source.code) ? source.code.join("\n").slice(0, maxSourceCharacters) : "";
  const selectedLines = formatSourceLines(source.selectedLines);
  const codeTruncated = sourceCode.length >= maxSourceCharacters ? "\n\n[Source truncated for request size.]" : "";

  return `Current ModelArchViz state:
- Model: ${safeString(model.label)} (${safeString(model.id)})
- Breadcrumb: ${safeString(model.breadcrumb)}
- Stats: ${safeString(model.stats)}
- Paper: ${safeString(paper.title)} (${safeString(paper.year)}) by ${safeString(paper.authors)}
- Venue: ${safeString(paper.venue)}
- Paper focus: ${Array.isArray(paper.focus) ? paper.focus.join(", ") : "unknown"}
- Search/filter text: ${safeString(context.searchQuery, "none")}

Current selection:
${
  selection
    ? `- Node id: ${safeString(selection.id)}
- Label: ${safeString(selection.label)}
- Type: ${safeString(selection.type)}
- Kind: ${safeString(selection.kind)}
- Summary: ${safeString(selection.summary, "none")}
- Badges: ${Array.isArray(selection.badges) && selection.badges.length > 0 ? selection.badges.join(", ") : "none"}
- Highlighted code lines: ${Array.isArray(selection.codeLines) ? selection.codeLines.join(", ") : "none"}`
    : "- No selected node."
}

Visible source file:
- Language: ${safeString(source.language)}
- File: ${safeString(source.fileName)}

Selected source lines:
${selectedLines}

Full visible source:
\`\`\`python
${sourceCode}
\`\`\`${codeTruncated}`;
}

export async function POST(request: Request) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    return NextResponse.json(
      { error: "OPENROUTER_API_KEY is not configured on the server." },
      { status: 500 },
    );
  }

  let payload: { messages?: unknown; context?: ChatContext };
  try {
    payload = await request.json();
  } catch {
    return NextResponse.json({ error: "Request body must be valid JSON." }, { status: 400 });
  }

  const messages = Array.isArray(payload.messages) ? payload.messages.filter(isChatMessage) : [];
  if (messages.length === 0) {
    return NextResponse.json({ error: "At least one chat message is required." }, { status: 400 });
  }

  const model = process.env.OPENROUTER_MODEL ?? defaultModel;
  const context = formatContext(payload.context ?? {});
  const requestMessages = messages.slice(-maxMessages);
  const referer = process.env.OPENROUTER_APP_URL;
  const title = process.env.OPENROUTER_APP_NAME ?? "ModelArchViz";

  const headers: Record<string, string> = {
    Authorization: `Bearer ${apiKey}`,
    "Content-Type": "application/json",
    "X-Title": title,
  };

  if (referer) {
    headers["HTTP-Referer"] = referer;
  }

  const openRouterResponse = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers,
    body: JSON.stringify({
      model,
      temperature: 0.2,
      max_tokens: 900,
      messages: [
        {
          role: "system",
          content:
            "You are the embedded ModelArchViz assistant. Use the provided current app state as ground truth. When the user says 'this', assume they mean the current selected architecture node and highlighted code lines. Explain architecture and code precisely, cite line numbers when useful, and keep answers concise.",
        },
        {
          role: "system",
          content: context,
        },
        ...requestMessages,
      ],
    }),
  });

  if (!openRouterResponse.ok) {
    const detail = await openRouterResponse.text();
    return NextResponse.json(
      { error: `OpenRouter returned ${openRouterResponse.status}: ${detail.slice(0, 500)}` },
      { status: openRouterResponse.status },
    );
  }

  const data = await openRouterResponse.json();
  const message = data?.choices?.[0]?.message?.content;
  if (typeof message !== "string" || message.trim().length === 0) {
    return NextResponse.json({ error: "OpenRouter returned an empty response." }, { status: 502 });
  }

  return NextResponse.json({
    message,
    model: data?.model ?? model,
  });
}

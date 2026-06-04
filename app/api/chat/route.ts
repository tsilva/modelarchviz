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

type CodeLineRange = {
  start?: number;
  end?: number;
};

type CodeSelection = {
  language?: string;
  fileName?: string;
  lines?: number[];
  ranges?: CodeLineRange[];
  reason?: string;
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
  paperSelection?: {
    pageNumber?: number;
    text?: string;
  } | null;
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
    agentSelectedLines?: SourceLine[];
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

function formatFullSource(lines: string[] | undefined) {
  if (!Array.isArray(lines) || lines.length === 0) {
    return "";
  }

  return lines.map((line, index) => `${index + 1}: ${line}`).join("\n").slice(0, maxSourceCharacters);
}

function parseAssistantPayload(content: string): { message: string; codeSelection: CodeSelection | null } {
  try {
    const parsed = JSON.parse(content) as { message?: unknown; codeSelection?: unknown; code_selection?: unknown };
    const message = typeof parsed.message === "string" ? parsed.message.trim() : "";
    const codeSelection = parsed.codeSelection ?? parsed.code_selection ?? null;

    return {
      message: message || content,
      codeSelection: codeSelection && typeof codeSelection === "object" ? (codeSelection as CodeSelection) : null,
    };
  } catch {
    return {
      message: content,
      codeSelection: null,
    };
  }
}

function normalizeLineNumbers(selection: CodeSelection, lineCount: number) {
  const lines = new Set<number>();

  if (Array.isArray(selection.lines)) {
    for (const line of selection.lines) {
      if (Number.isInteger(line) && line >= 1 && line <= lineCount) {
        lines.add(line);
      }
    }
  }

  if (Array.isArray(selection.ranges)) {
    for (const range of selection.ranges) {
      const rangeStart = range.start;
      const rangeEnd = range.end;
      if (
        typeof rangeStart !== "number" ||
        typeof rangeEnd !== "number" ||
        !Number.isInteger(rangeStart) ||
        !Number.isInteger(rangeEnd)
      ) {
        continue;
      }

      const start = Math.max(1, Math.min(rangeStart, rangeEnd));
      const end = Math.min(lineCount, Math.max(rangeStart, rangeEnd));
      for (let line = start; line <= end; line += 1) {
        lines.add(line);
      }
    }
  }

  return [...lines].sort((left, right) => left - right);
}

function sanitizeCodeSelection(selection: CodeSelection | null, context: ChatContext) {
  if (!selection) {
    return null;
  }

  const source = context.source ?? {};
  const sourceCode = Array.isArray(source.code) ? source.code : [];
  const lines = normalizeLineNumbers(selection, sourceCode.length);
  if (lines.length === 0) {
    return null;
  }

  return {
    modelId: safeString(context.model?.id, ""),
    language: safeString(selection.language, safeString(source.language, "")),
    fileName: safeString(selection.fileName, safeString(source.fileName, "")),
    lines,
    reason: safeString(selection.reason, ""),
  };
}

function formatContext(context: ChatContext) {
  const model = context.model ?? {};
  const paper = context.paper ?? {};
  const paperSelection = context.paperSelection ?? null;
  const selection = context.selection ?? null;
  const source = context.source ?? {};
  const sourceCode = formatFullSource(source.code);
  const selectedLines = formatSourceLines(source.selectedLines);
  const agentSelectedLines = formatSourceLines(source.agentSelectedLines);
  const codeTruncated = sourceCode.length >= maxSourceCharacters ? "\n\n[Source truncated for request size.]" : "";

  return `Current ModelArchViz state:
- Model: ${safeString(model.label)} (${safeString(model.id)})
- Breadcrumb: ${safeString(model.breadcrumb)}
- Stats: ${safeString(model.stats)}
- Paper: ${safeString(paper.title)} (${safeString(paper.year)}) by ${safeString(paper.authors)}
- Venue: ${safeString(paper.venue)}
- Paper focus: ${Array.isArray(paper.focus) ? paper.focus.join(", ") : "unknown"}
- Search/filter text: ${safeString(context.searchQuery, "none")}
- Selected paper text: ${
    paperSelection && typeof paperSelection.text === "string" && paperSelection.text.trim().length > 0
      ? `page ${Number.isFinite(paperSelection.pageNumber) ? paperSelection.pageNumber : "unknown"}: ${paperSelection.text.trim().slice(0, 4000)}`
      : "none"
  }

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

Assistant-selected source lines:
${agentSelectedLines}

Full visible source with line numbers:
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
            'You are the embedded ModelArchViz assistant. Use the provided current app state as ground truth. When the user says "this", assume they mean the selected paper text when present, otherwise the current selected architecture node and highlighted code lines. Explain architecture, paper text, and code precisely, cite line numbers or paper pages when useful, and keep answers concise. Return only valid JSON shaped as {"message":"user-facing markdown answer","codeSelection":null} or {"message":"user-facing markdown answer","codeSelection":{"language":"current source language","fileName":"current source file","lines":[line numbers to highlight],"ranges":[{"start":number,"end":number}],"reason":"short reason"}}. Include codeSelection when the user asks where paper text, a concept, or architecture behavior appears in code, or when highlighting code would directly answer the question. Use line numbers from the full visible source.',
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
  const content = data?.choices?.[0]?.message?.content;
  if (typeof content !== "string" || content.trim().length === 0) {
    return NextResponse.json({ error: "OpenRouter returned an empty response." }, { status: 502 });
  }
  const parsed = parseAssistantPayload(content.trim());
  const codeSelection = sanitizeCodeSelection(parsed.codeSelection, payload.context ?? {});

  return NextResponse.json({
    message: parsed.message,
    codeSelection,
    model: data?.model ?? model,
  });
}

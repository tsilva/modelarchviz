export type ChatCodeLanguage = "pytorch" | "jax";

export type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

export type SourceLine = {
  lineNumber: number;
  text: string;
};

type CodeLineRange = {
  start?: number;
  end?: number;
};

export type CodeSelection = {
  language?: string;
  fileName?: string;
  lines?: number[];
  ranges?: CodeLineRange[];
  reason?: string;
};

export type ChatContext = {
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
    userSelectedLines?: SourceLine[];
    userSelectedText?: string;
    agentSelectedLines?: SourceLine[];
  };
  searchQuery?: string;
};

export type ChatResponse = {
  message?: string;
  error?: string;
  codeSelection?: unknown;
  model?: string;
};

export function normalizeLineNumbers(selection: CodeSelection, lineCount: number) {
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

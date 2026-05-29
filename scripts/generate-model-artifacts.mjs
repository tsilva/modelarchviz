import { mkdir, readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const sourceDir = path.join(repoRoot, "app", "model-notebooks");
const generatedCodeDir = path.join(repoRoot, "app", "generated", "model-code");
const notebookDir = path.join(repoRoot, "public", "notebooks");

function stripJupytextHeader(lines) {
  if (lines[0] !== "# ---") {
    return lines;
  }

  const headerEnd = lines.findIndex((line, index) => index > 0 && line === "# ---");
  if (headerEnd === -1) {
    return lines;
  }

  return lines.slice(headerEnd + 1);
}

function parseNotebookSource(source) {
  const normalizedSource = source.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = stripJupytextHeader(normalizedSource.split("\n"));
  const cells = [];
  let currentCell = null;

  function pushCell() {
    if (!currentCell) {
      return;
    }

    while (currentCell.lines.length > 0 && currentCell.lines.at(-1) === "") {
      currentCell.lines.pop();
    }

    cells.push(currentCell);
  }

  for (const line of lines) {
    const codeCell = line.match(/^# %%\s*$/);
    const markdownCell = line.match(/^# %% \[markdown\]\s*$/);

    if (codeCell || markdownCell) {
      pushCell();
      currentCell = {
        cell_type: markdownCell ? "markdown" : "code",
        lines: [],
      };
      continue;
    }

    if (!currentCell) {
      currentCell = {
        cell_type: "code",
        lines: [],
      };
    }

    currentCell.lines.push(line);
  }

  pushCell();

  return cells.filter((cell) => cell.lines.length > 0);
}

function markdownSource(lines) {
  return lines.map((line) => {
    if (line === "#") {
      return "";
    }

    if (line.startsWith("# ")) {
      return line.slice(2);
    }

    if (line.startsWith("#")) {
      return line.slice(1).trimStart();
    }

    return line;
  });
}

function sourceLines(lines) {
  return lines.map((line, index) => {
    const suffix = index === lines.length - 1 ? "" : "\n";
    return `${line}${suffix}`;
  });
}

function notebookFromCells(fileName, cells) {
  return {
    cells: cells.map((cell) => {
      if (cell.cell_type === "markdown") {
        return {
          cell_type: "markdown",
          metadata: {},
          source: sourceLines(markdownSource(cell.lines)),
        };
      }

      return {
        cell_type: "code",
        execution_count: null,
        metadata: {},
        outputs: [],
        source: sourceLines(cell.lines),
      };
    }),
    metadata: {
      jupytext: {
        formats: "ipynb,py:percent",
        text_representation: {
          extension: ".py",
          format_name: "percent",
          format_version: "1.3",
          jupytext_version: "local-generator",
        },
      },
      kernelspec: {
        display_name: "Python 3",
        language: "python",
        name: "python3",
      },
      language_info: {
        name: "python",
        pycodemirror_mode: {
          name: "ipython",
          version: 3,
        },
      },
      modelarchviz: {
        source: `app/model-notebooks/${fileName}`,
      },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

function cleanedPythonFromCells(cells) {
  const codeCells = cells.filter((cell) => cell.cell_type === "code");
  const blocks = codeCells.map((cell) => cell.lines.join("\n"));
  return `${blocks.join("\n\n").trimEnd()}\n`;
}

async function main() {
  await mkdir(generatedCodeDir, { recursive: true });
  await mkdir(notebookDir, { recursive: true });

  const entries = await readdir(sourceDir, { withFileTypes: true });
  const sourceFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".py"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  if (sourceFiles.length === 0) {
    throw new Error(`No notebook source files found in ${sourceDir}`);
  }

  for (const fileName of sourceFiles) {
    const sourcePath = path.join(sourceDir, fileName);
    const source = await readFile(sourcePath, "utf8");
    const cells = parseNotebookSource(source);
    const cleanedPython = cleanedPythonFromCells(cells);
    const notebook = notebookFromCells(fileName, cells);
    const notebookName = fileName.replace(/\.py$/, ".ipynb");

    await writeFile(path.join(generatedCodeDir, fileName), cleanedPython, "utf8");
    await writeFile(path.join(notebookDir, notebookName), `${JSON.stringify(notebook, null, 2)}\n`, "utf8");
  }

  console.log(`Generated ${sourceFiles.length} Python files and ${sourceFiles.length} notebooks.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

import { copyFile, mkdir, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  cleanedPythonAndHighlightManifest,
  stripNotebookAnchorMarkers,
} from "./model-highlight-anchors.mjs";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const require = createRequire(import.meta.url);
const sourceDir = path.join(repoRoot, "app", "model-notebooks");
const templateDir = path.join(repoRoot, "app", "model-templates");
const architecturePath = path.join(repoRoot, "app", "model-arch-viz-app.tsx");
const generatedManifestPath = path.join(repoRoot, "app", "generated", "model-sources.ts");
const notebookDir = path.join(repoRoot, "public", "notebooks");
const pdfWorkerSourcePath = require.resolve("pdfjs-dist/build/pdf.worker.min.mjs");
const pdfWorkerPath = path.join(repoRoot, "public", "pdf.worker.min.mjs");

function parseNotebookSource(source) {
  const normalizedSource = source.replace(/\r\n/g, "\n").replace(/\r/g, "\n");
  const lines = normalizedSource.split("\n");
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
    const cellMarker = line.match(/^# %%(?:\s+\[([^\]]+)\])?\s*$/);

    if (cellMarker) {
      const marker = cellMarker[1]?.trim() ?? "";
      const markdownCell = marker === "markdown";
      const tags = marker && !markdownCell ? marker.split(",").map((tag) => tag.trim()).filter(Boolean) : [];
      pushCell();
      currentCell = {
        cell_type: markdownCell ? "markdown" : "code",
        tags,
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

function cellMetadata(cell) {
  if (!cell.tags || cell.tags.length === 0) {
    return {};
  }

  return { tags: cell.tags };
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

function notebookFromCells(fileName, cells, sourceLabel = `app/model-notebooks/${fileName}`) {
  return {
    cells: cells.map((cell) => {
      if (cell.cell_type === "markdown") {
        return {
          cell_type: "markdown",
          metadata: cellMetadata(cell),
          source: sourceLines(markdownSource(cell.lines)),
        };
      }

      return {
        cell_type: "code",
        execution_count: null,
        metadata: cellMetadata(cell),
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
        source: sourceLabel,
      },
    },
    nbformat: 4,
    nbformat_minor: 5,
  };
}

function templateValue(context, expression) {
  return expression.split(".").reduce((value, key) => value?.[key], context);
}

function renderTemplate(template, context) {
  return template.replace(/{{\s*([A-Za-z0-9_.]+)\s*}}/g, (_, expression) => {
    const value = templateValue(context, expression);

    if (value === undefined || value === null) {
      throw new Error(`Missing template value: ${expression}`);
    }

    return String(value);
  });
}

async function safeReaddir(directory) {
  try {
    return await readdir(directory, { withFileTypes: true });
  } catch (error) {
    if (error?.code === "ENOENT") {
      return [];
    }

    throw error;
  }
}

async function writeArtifacts({ fileName, source, sourceLabel }) {
  const sourceCells = parseNotebookSource(source);
  const { cleanedPython, highlights } = cleanedPythonAndHighlightManifest(sourceCells, fileName);
  const notebookCells = stripNotebookAnchorMarkers(sourceCells, fileName);
  const notebook = notebookFromCells(fileName, notebookCells, sourceLabel);
  const notebookName = fileName.replace(/\.py$/, ".ipynb");

  await writeFile(path.join(notebookDir, notebookName), `${JSON.stringify(notebook, null, 2)}\n`, "utf8");

  return { cleanedPython, fileName, highlights, notebookName };
}

async function pruneUnexpectedArtifacts(directory, extension, expectedNames) {
  const entries = await safeReaddir(directory);

  await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(extension) && !expectedNames.has(entry.name))
      .map((entry) => rm(path.join(directory, entry.name))),
  );
}

function modelSourceManifest(sources, highlightManifests, variantDefinitions) {
  const entries = [...sources.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(
      ([fileName, source]) => `  ${JSON.stringify(fileName)}: ${JSON.stringify(source)},`,
    );

  const resnetVariants = variantDefinitions.get("resnet");
  if (!resnetVariants) {
    throw new Error("Missing ResNet variant definitions");
  }

  const highlightEntries = [...highlightManifests.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([fileName, highlights]) => `  ${JSON.stringify(fileName)}: ${JSON.stringify(highlights)},`);

  return `export type ModelHighlightAnchor = { readonly lines: readonly number[]; readonly focusLine: number };\n\nexport const modelSources: Readonly<Record<string, string>> = {\n${entries.join("\n")}\n};\n\nexport const modelHighlightManifest: Readonly<Record<string, Readonly<Record<string, ModelHighlightAnchor>>>> = {\n${highlightEntries.join("\n")}\n};\n\nexport const resnetVariantDefinitions = ${JSON.stringify(resnetVariants, null, 2)} as const;\n`;
}

async function main() {
  const architectureSource = await readFile(architecturePath, "utf8");
  if (/\b(?:codeLines|jaxCodeLines)\s*:/.test(architectureSource)) {
    throw new Error("Legacy numeric architecture mappings are not allowed; use semantic sourceRefs");
  }

  await mkdir(path.dirname(generatedManifestPath), { recursive: true });
  await mkdir(notebookDir, { recursive: true });

  const generatedSources = new Map();
  const generatedHighlights = new Map();
  const generatedNotebookNames = new Set();
  const variantDefinitions = new Map();

  async function generateAndTrack(options) {
    const artifact = await writeArtifacts(options);
    generatedSources.set(artifact.fileName, artifact.cleanedPython);
    generatedHighlights.set(artifact.fileName, artifact.highlights);
    generatedNotebookNames.add(artifact.notebookName);
  }

  const entries = await safeReaddir(sourceDir);
  const sourceFiles = entries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".py"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  for (const fileName of sourceFiles) {
    const sourcePath = path.join(sourceDir, fileName);
    const source = await readFile(sourcePath, "utf8");
    await generateAndTrack({
      fileName,
      source,
      sourceLabel: `app/model-notebooks/${fileName}`,
    });
  }

  const templateEntries = await safeReaddir(templateDir);
  const variantFiles = templateEntries
    .filter((entry) => entry.isFile() && entry.name.endsWith(".variants.json"))
    .map((entry) => entry.name)
    .sort((a, b) => a.localeCompare(b));

  for (const variantFile of variantFiles) {
    const family = variantFile.replace(/\.variants\.json$/, "");
    const variantsPath = path.join(templateDir, variantFile);
    const variants = JSON.parse(await readFile(variantsPath, "utf8"));
    variantDefinitions.set(family, variants);
    const pytorchTemplateName = `${family}.py.template`;
    const jaxTemplateName = `${family}_jax.py.template`;
    const pytorchTemplate = await readFile(path.join(templateDir, pytorchTemplateName), "utf8");
    const jaxTemplate = await readFile(path.join(templateDir, jaxTemplateName), "utf8");

    for (const variant of variants) {
      const templateContext = {
        ...variant,
        stage1Blocks: variant.stageBlocks[0],
        stage2Blocks: variant.stageBlocks[1],
        stage3Blocks: variant.stageBlocks[2],
        stage4Blocks: variant.stageBlocks[3],
      };

      await generateAndTrack({
        fileName: `${variant.id}.py`,
        source: renderTemplate(pytorchTemplate, templateContext),
        sourceLabel: `app/model-templates/${pytorchTemplateName}#${variant.id}`,
      });

      await generateAndTrack({
        fileName: `${variant.id}_jax.py`,
        source: renderTemplate(jaxTemplate, templateContext),
        sourceLabel: `app/model-templates/${jaxTemplateName}#${variant.id}`,
      });
    }
  }

  if (sourceFiles.length === 0 && variantFiles.length === 0) {
    throw new Error(`No notebook source files found in ${sourceDir} or ${templateDir}`);
  }

  await pruneUnexpectedArtifacts(notebookDir, ".ipynb", generatedNotebookNames);
  await writeFile(generatedManifestPath, modelSourceManifest(generatedSources, generatedHighlights, variantDefinitions), "utf8");
  await copyFile(pdfWorkerSourcePath, pdfWorkerPath);

  console.log(`Generated model artifacts from ${sourceFiles.length} notebook sources and ${variantFiles.length} variant families.`);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});

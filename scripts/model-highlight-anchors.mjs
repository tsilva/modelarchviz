const archMarkerPattern = /^\s*# @arch ([a-z0-9][a-z0-9_.-]*):(start|end)\s*$/;

function markerForLine(line, fileName) {
  const marker = line.match(archMarkerPattern);
  if (!marker && line.includes("@arch")) {
    throw new Error(`Malformed architecture anchor in ${fileName}: ${line.trim()}`);
  }

  return marker ? { key: marker[1], boundary: marker[2] } : null;
}

function validateAnchorContent(fileName, key, lines) {
  if (lines.length === 0) {
    throw new Error(`Empty architecture anchor in ${fileName}: ${key}`);
  }

  const hasConcreteCode = lines.some((line) => {
    const trimmed = line.trim();
    return trimmed.length > 0 && !trimmed.startsWith("#");
  });
  if (!hasConcreteCode) {
    throw new Error(`Metadata-only architecture anchor in ${fileName}: ${key}`);
  }
}

export function stripNotebookAnchorMarkers(cells, fileName) {
  const seenKeys = new Set();

  return cells.map((cell) => {
    let activeAnchor = null;
    const strippedLines = [];

    for (const line of cell.lines) {
      const marker = markerForLine(line, fileName);
      if (!marker) {
        strippedLines.push(line);
        if (activeAnchor) {
          activeAnchor.lines.push(line);
        }
        continue;
      }

      if (cell.cell_type !== "code" || cell.tags?.includes("notebook-only")) {
        throw new Error(`Architecture anchor outside web-visible code in ${fileName}: ${marker.key}`);
      }

      if (marker.boundary === "start") {
        if (activeAnchor) {
          throw new Error(`Nested architecture anchor in ${fileName}: ${marker.key} inside ${activeAnchor.key}`);
        }
        if (seenKeys.has(marker.key)) {
          throw new Error(`Duplicate architecture anchor in ${fileName}: ${marker.key}`);
        }
        activeAnchor = { key: marker.key, lines: [] };
        continue;
      }

      if (!activeAnchor) {
        throw new Error(`Orphaned architecture anchor end in ${fileName}: ${marker.key}`);
      }
      if (activeAnchor.key !== marker.key) {
        throw new Error(`Mismatched architecture anchor in ${fileName}: expected ${activeAnchor.key}, received ${marker.key}`);
      }
      validateAnchorContent(fileName, marker.key, activeAnchor.lines);
      seenKeys.add(marker.key);
      activeAnchor = null;
    }

    if (activeAnchor) {
      throw new Error(`Orphaned architecture anchor start in ${fileName}: ${activeAnchor.key}`);
    }

    return { ...cell, lines: strippedLines };
  });
}

export function cleanedPythonAndHighlightManifest(cells, fileName) {
  const codeCells = cells.filter((cell) => cell.cell_type === "code" && !cell.tags?.includes("notebook-only"));
  const outputLines = [];
  const highlights = {};

  for (const cell of codeCells) {
    if (outputLines.length > 0) {
      outputLines.push("");
    }

    let activeAnchor = null;
    for (const line of cell.lines) {
      const marker = markerForLine(line, fileName);
      if (!marker) {
        outputLines.push(line);
        if (activeAnchor) {
          activeAnchor.lines.push(outputLines.length);
          activeAnchor.content.push(line);
        }
        continue;
      }

      if (marker.boundary === "start") {
        if (activeAnchor) {
          throw new Error(`Nested architecture anchor in ${fileName}: ${marker.key} inside ${activeAnchor.key}`);
        }
        if (Object.hasOwn(highlights, marker.key)) {
          throw new Error(`Duplicate architecture anchor in ${fileName}: ${marker.key}`);
        }
        activeAnchor = { key: marker.key, lines: [], content: [] };
        continue;
      }

      if (!activeAnchor) {
        throw new Error(`Orphaned architecture anchor end in ${fileName}: ${marker.key}`);
      }
      if (activeAnchor.key !== marker.key) {
        throw new Error(`Mismatched architecture anchor in ${fileName}: expected ${activeAnchor.key}, received ${marker.key}`);
      }
      validateAnchorContent(fileName, marker.key, activeAnchor.content);
      highlights[marker.key] = {
        lines: activeAnchor.lines,
        focusLine: activeAnchor.lines[0],
      };
      activeAnchor = null;
    }

    if (activeAnchor) {
      throw new Error(`Orphaned architecture anchor start in ${fileName}: ${activeAnchor.key}`);
    }
  }

  while (outputLines.length > 0 && outputLines.at(-1) === "") {
    outputLines.pop();
  }

  return {
    cleanedPython: `${outputLines.join("\n")}\n`,
    highlights,
  };
}

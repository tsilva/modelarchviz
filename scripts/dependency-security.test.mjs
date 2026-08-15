import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import test from "node:test";

const lockfile = readFileSync("pnpm-lock.yaml", "utf8");

function lockedVersions(packageName) {
  const escaped = packageName.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const pattern = new RegExp(
    `^  ['"]?${escaped}@([^:\\s('"\\)]+)(?:\\([^)]*\\))?['"]?:`,
    "gm",
  );
  return [...new Set([...lockfile.matchAll(pattern)].map((match) => match[1]))].sort();
}

test("formerly vulnerable packages stay on patched versions", () => {
  assert.deepEqual(lockedVersions("@opentelemetry/core"), ["2.10.0"]);
  assert.deepEqual(lockedVersions("brace-expansion"), ["5.0.9"]);
  assert.deepEqual(lockedVersions("fast-uri"), ["3.1.5"]);
  assert.deepEqual(lockedVersions("nanoid"), ["3.3.18"]);
});

test("the complete dependency graph has no known vulnerabilities", () => {
  const audit = JSON.parse(
    execFileSync("pnpm", ["audit", "--json"], {
      encoding: "utf8",
      maxBuffer: 8 * 1024 * 1024,
    }),
  );

  assert.deepEqual(audit.metadata.vulnerabilities, {
    info: 0,
    low: 0,
    moderate: 0,
    high: 0,
    critical: 0,
  });
});

test("dependency sources are registry-only and age protected", () => {
  const importers = lockfile.slice(lockfile.indexOf("importers:"), lockfile.indexOf("packages:"));
  assert.doesNotMatch(
    importers,
    /specifier:\s*(?:git\+|github:|https?:|file:|link:|workspace:)/,
  );

  const npmrc = readFileSync(".npmrc", "utf8");
  assert.match(npmrc, /^minimum-release-age=10080$/m);
  assert.match(npmrc, /^block-exotic-subdeps=true$/m);
  assert.match(npmrc, /^ignore-scripts=true$/m);
});

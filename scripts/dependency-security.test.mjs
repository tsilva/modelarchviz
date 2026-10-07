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
  for (const [name, floor] of Object.entries({
    "@opentelemetry/core": "2.10.0",
    "brace-expansion": "5.0.12",
    "fast-uri": "3.1.8",
    "nanoid": "3.3.19",
    "sharp": "0.35.5",
  })) {
    const versions = lockedVersions(name);
    assert.ok(versions.length, `missing dependency ${name}`);
    for (const version of versions) {
      const actual = version.split(".").map(Number);
      const minimum = floor.split(".").map(Number);
      assert.ok(actual.every(Number.isFinite), `unexpected version ${version}`);
      const difference = actual.findIndex((part, index) => part !== minimum[index]);
      assert.ok(difference === -1 || actual[difference] > minimum[difference], `${name}@${version} is below ${floor}`);
    }
  }
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

import { modelHighlightManifest, modelSources, resnetVariantDefinitions } from "./generated/model-sources";
import type { ModelHighlightAnchor } from "./generated/model-sources";
import type { modelCatalog } from "./model-routes";

export type ModelSourceBundle = {
  sources: Readonly<Record<string, string>>;
  highlights: Readonly<Record<string, Readonly<Record<string, ModelHighlightAnchor>>>>;
};

export function getModelSourceBundle(model: (typeof modelCatalog)[number]): ModelSourceBundle {
  const baseName = "sourceBaseName" in model ? model.sourceBaseName : model.id;
  const names = new Set<string>([baseName]);
  if (model.id === "resnet18") {
    for (const variant of resnetVariantDefinitions) names.add(variant.id);
  }
  const files = [...names].flatMap((name) => [`${name}.py`, `${name}_jax.py`]);
  return {
    sources: Object.fromEntries(files.map((file) => [file, modelSources[file]])),
    highlights: Object.fromEntries(files.map((file) => [file, modelHighlightManifest[file]])),
  };
}

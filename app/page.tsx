import ModelArchVizApp from "./model-arch-viz-app";
import { modelCatalogByPublicationDate } from "./model-routes";
import { getModelSourceBundle } from "./model-source-bundle";

export default function Home() {
  const model = modelCatalogByPublicationDate[0];
  return <ModelArchVizApp modelId={model.id} sourceBundle={getModelSourceBundle(model)} />;
}

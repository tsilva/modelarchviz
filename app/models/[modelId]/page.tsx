import type { Metadata } from "next";
import { notFound } from "next/navigation";
import ModelArchVizApp from "../../model-arch-viz-app";
import { getModelRoute, modelCatalog, modelRoutePath, siteConfig } from "../../model-routes";

type ModelPageProps = {
  params: Promise<{
    modelId: string;
  }>;
};

export function generateStaticParams() {
  return modelCatalog.map((model) => ({
    modelId: model.id,
  }));
}

export async function generateMetadata({ params }: ModelPageProps): Promise<Metadata> {
  const { modelId } = await params;
  const model = getModelRoute(modelId);

  if (!model) {
    return {};
  }

  const path = modelRoutePath(model.id);
  const title = `${model.title} | ${siteConfig.name}`;

  return {
    title,
    description: model.description,
    alternates: {
      canonical: path,
    },
    openGraph: {
      title,
      description: model.description,
      url: path,
      images: [
        {
          url: siteConfig.socialImagePath,
          width: siteConfig.socialImageWidth,
          height: siteConfig.socialImageHeight,
          alt: `${model.label} architecture in ${siteConfig.name}`,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description: model.description,
      images: [siteConfig.socialImagePath],
    },
  };
}

export default async function ModelPage({ params }: ModelPageProps) {
  const { modelId } = await params;
  const model = getModelRoute(modelId);

  if (!model) {
    notFound();
  }

  return <ModelArchVizApp initialModelId={model.id} />;
}

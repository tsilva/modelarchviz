import type { Metadata } from "next";
import { notFound } from "next/navigation";
import ModelArchVizApp from "../../model-arch-viz-app";
import { getModelRoute, modelCatalog, modelRoutePath } from "../../model-routes";

type ModelPageProps = {
  params: {
    modelId: string;
  };
};

export function generateStaticParams() {
  return modelCatalog.map((model) => ({
    modelId: model.id,
  }));
}

export function generateMetadata({ params }: ModelPageProps): Metadata {
  const model = getModelRoute(params.modelId);

  if (!model) {
    return {};
  }

  const path = modelRoutePath(model.id);
  const title = `${model.title} | ModelArchViz`;

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
          url: "/brand/web-seo/og-image-1200x630.png",
          width: 1200,
          height: 630,
          alt: `${model.label} architecture in ModelArchViz`,
        },
      ],
    },
    twitter: {
      card: "summary_large_image",
      title,
      description: model.description,
      images: ["/brand/web-seo/og-image-1200x630.png"],
    },
  };
}

export default function ModelPage({ params }: ModelPageProps) {
  const model = getModelRoute(params.modelId);

  if (!model) {
    notFound();
  }

  return <ModelArchVizApp initialModelId={model.id} />;
}

import type { MetadataRoute } from "next";
import { modelCatalog, modelRoutePath, siteUrl } from "./model-routes";

export default function sitemap(): MetadataRoute.Sitemap {
  const origin = siteUrl();

  return [
    {
      url: new URL("/", origin).toString(),
      changeFrequency: "weekly",
      priority: 1,
    },
    ...modelCatalog.map((model) => ({
      url: new URL(modelRoutePath(model.id), origin).toString(),
      changeFrequency: "weekly" as const,
      priority: 0.8,
    })),
  ];
}

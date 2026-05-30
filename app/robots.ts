import type { MetadataRoute } from "next";
import { siteUrl } from "./model-routes";

export default function robots(): MetadataRoute.Robots {
  const origin = siteUrl();

  return {
    rules: {
      userAgent: "*",
      allow: "/",
    },
    sitemap: new URL("/sitemap.xml", origin).toString(),
  };
}

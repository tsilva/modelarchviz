import type { Metadata, Viewport } from "next";
import "./globals.css";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "https://modelarch.tsilva.eu"),
  title: {
    default: "ModelArchViz",
    template: "%s",
  },
  description: "Explore neural network architecture blocks alongside PyTorch, JAX, and source-paper context.",
  manifest: "/brand/web-seo/site.webmanifest",
  alternates: {
    canonical: "/",
  },
  icons: {
    icon: [
      { url: "/brand/web-seo/favicon/favicon.ico" },
      { url: "/brand/web-seo/favicon/favicon-32.png", sizes: "32x32", type: "image/png" },
      { url: "/brand/web-seo/favicon/favicon-16.png", sizes: "16x16", type: "image/png" },
    ],
    apple: [{ url: "/brand/web-seo/apple-touch-icon.png", sizes: "180x180", type: "image/png" }],
  },
  openGraph: {
    title: "ModelArchViz",
    description: "Explore neural network architecture blocks alongside PyTorch, JAX, and source-paper context.",
    url: "/",
    siteName: "ModelArchViz",
    type: "website",
    images: [
      {
        url: "/brand/web-seo/og-image-1200x630.png",
        width: 1200,
        height: 630,
        alt: "ModelArchViz architecture and code visualization",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "ModelArchViz",
    description: "Explore neural network architecture blocks alongside PyTorch, JAX, and source-paper context.",
    images: ["/brand/web-seo/og-image-1200x630.png"],
  },
};

export const viewport: Viewport = {
  themeColor: "#f8fafc",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

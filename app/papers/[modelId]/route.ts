const paperPdfUrls: Record<string, string> = {
  lenet5: "https://www.cs.princeton.edu/courses/archive/spring08/cos598B/Lectures/LeCunEtAl.pdf",
  gpt2: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
  resnet18: "https://arxiv.org/pdf/1512.03385",
};

export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: { modelId: string } }) {
  const url = paperPdfUrls[params.modelId];

  if (!url) {
    return new Response("Paper not found", { status: 404 });
  }

  const range = request.headers.get("range");
  const response = await fetch(url, {
    headers: {
      ...(range ? { Range: range } : {}),
      "User-Agent": "ModelArchViz/0.1 paper viewer",
    },
  });

  if (!response.ok || !response.body) {
    return new Response("Paper PDF could not be loaded", { status: 502 });
  }

  const headers = new Headers({
    "Accept-Ranges": "bytes",
    "Cache-Control": "public, max-age=86400",
    "Content-Type": response.headers.get("content-type") ?? "application/pdf",
  });

  for (const header of ["content-length", "content-range"]) {
    const value = response.headers.get(header);
    if (value) {
      headers.set(header, value);
    }
  }

  return new Response(response.body, {
    status: response.status,
    headers: {
      ...Object.fromEntries(headers),
    },
  });
}

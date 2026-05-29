const paperPdfUrls: Record<string, string> = {
  mlp: "https://www-edlab.cs.umass.edu/cs697l/readings/Learning%20representations%20by%20back-propagating%20errors.pdf",
  rnn: "https://gwern.net/doc/ai/nn/rnn/1990-elman.pdf",
  lstm: "https://gwern.net/doc/ai/nn/rnn/1997-hochreiter.pdf",
  gru: "https://arxiv.org/pdf/1406.1078",
  lenet5: "https://www.cs.princeton.edu/courses/archive/spring08/cos598B/Lectures/LeCunEtAl.pdf",
  alexnet: "https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf",
  googlenet: "https://arxiv.org/pdf/1409.4842",
  unet: "https://arxiv.org/pdf/1505.04597",
  transformer: "https://arxiv.org/pdf/1706.03762",
  bert: "https://arxiv.org/pdf/1810.04805",
  gpt2: "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
  vit: "https://arxiv.org/pdf/2010.11929",
  resnet18: "https://arxiv.org/pdf/1512.03385",
};

export const dynamic = "force-dynamic";

export async function GET(request: Request, { params }: { params: { modelId: string } }) {
  const modelId = params.modelId.replace(/\.pdf$/i, "");
  const url = paperPdfUrls[modelId];

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
    "Content-Disposition": `inline; filename="${modelId}.pdf"`,
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

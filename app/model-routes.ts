export type ModelRouteSummary = {
  id: string;
  label: string;
  publishedDate: string;
  title: string;
  description: string;
};

export const modelRouteSummaries = [
  {
    id: "mlp",
    label: "MLP",
    publishedDate: "1986-10-09",
    title: "MLP Architecture",
    description: "Inspect a multilayer perceptron architecture alongside PyTorch, JAX, and source-paper context.",
  },
  {
    id: "rnn",
    label: "RNN (Elman)",
    publishedDate: "1990-03-01",
    title: "Elman RNN Architecture",
    description: "Explore an Elman recurrent neural network with architecture blocks, code, and paper context.",
  },
  {
    id: "lstm",
    label: "LSTM",
    publishedDate: "1997-11-01",
    title: "LSTM Architecture",
    description: "Use the LSTM architecture card to trace sequence inputs, input/forget/output gates, cell-state updates, PyTorch and JAX code, notebooks, and the original paper.",
  },
  {
    id: "autoencoder",
    label: "Autoencoder",
    publishedDate: "2006-07-28",
    title: "Autoencoder Architecture",
    description: "Trace an autoencoder encoder, bottleneck code, decoder reconstruction path, reconstruction loss, implementation code, and paper context.",
  },
  {
    id: "lenet5",
    label: "LeNet-5",
    publishedDate: "1998-11-01",
    title: "LeNet-5 Architecture",
    description: "Study the LeNet-5 convolutional network through its architecture, code, and original paper context.",
  },
  {
    id: "alexnet",
    label: "AlexNet",
    publishedDate: "2012-12-03",
    title: "AlexNet Architecture",
    description: "Inspect AlexNet convolutional stages, classifier layers, implementation code, and paper context.",
  },
  {
    id: "vgg16",
    label: "VGG-16",
    publishedDate: "2014-09-04",
    title: "VGG-16 Architecture",
    description: "Inspect VGG-16 stacked 3x3 convolutional stages, dense classifier layers, implementation code, and paper context.",
  },
  {
    id: "vae",
    label: "VAE",
    publishedDate: "2013-12-20",
    title: "Variational Autoencoder Architecture",
    description: "Trace a variational autoencoder Gaussian encoder, reparameterization trick, decoder, ELBO loss, implementation code, and paper context.",
  },
  {
    id: "gan",
    label: "GAN",
    publishedDate: "2014-06-10",
    title: "Generative Adversarial Network Architecture",
    description: "Trace the original GAN generator, discriminator, adversarial losses, alternating training updates, implementation code, and paper context.",
  },
  {
    id: "gru",
    label: "GRU",
    publishedDate: "2014-06-03",
    title: "GRU Architecture",
    description: "Explore GRU update and reset gates with architecture blocks, code, and source-paper context.",
  },
  {
    id: "seq2seq",
    label: "Seq2Seq",
    publishedDate: "2014-09-10",
    title: "Seq2Seq Architecture",
    description: "Inspect recurrent encoder-decoder sequence transduction with PyTorch, JAX, and source-paper context.",
  },
  {
    id: "googlenet",
    label: "GoogLeNet / Inception v1",
    publishedDate: "2014-09-17",
    title: "GoogLeNet Inception Architecture",
    description: "Inspect GoogLeNet Inception modules, branches, implementation code, and paper context.",
  },
  {
    id: "unet",
    label: "U-Net",
    publishedDate: "2015-05-18",
    title: "U-Net Architecture",
    description: "Trace U-Net encoder-decoder skips with visual architecture blocks, code, and paper context.",
  },
  {
    id: "resnet18",
    label: "ResNet",
    publishedDate: "2015-12-10",
    title: "ResNet Architecture Variants",
    description: "Inspect ResNet-18 through ResNet-152 residual stages, generated implementation code, and paper context.",
  },
  {
    id: "widenet",
    label: "WideNet",
    publishedDate: "2016-05-23",
    title: "Wide Residual Network Architecture",
    description: "Explore wide residual blocks with architecture diagrams, PyTorch and JAX code, and paper context.",
  },
  {
    id: "densenet",
    label: "DenseNet-121",
    publishedDate: "2016-08-25",
    title: "DenseNet-121 Architecture",
    description: "Inspect DenseNet feature concatenation, dense blocks, implementation code, and paper context.",
  },
  {
    id: "transformer",
    label: "Transformer",
    publishedDate: "2017-06-12",
    title: "Transformer Architecture",
    description: "Explore encoder-decoder attention blocks, PyTorch and JAX code, and source-paper context.",
  },
  {
    id: "vqvae",
    label: "VQ-VAE",
    publishedDate: "2017-11-02",
    title: "VQ-VAE Architecture",
    description: "Trace VQ-VAE encoder outputs, nearest-code vector quantization, straight-through estimator, decoder reconstruction, implementation code, and paper context.",
  },
  {
    id: "mobilenetv2",
    label: "MobileNetV2",
    publishedDate: "2018-01-13",
    title: "MobileNetV2 Architecture",
    description: "Inspect MobileNetV2 inverted residual blocks, depthwise separable convolutions, linear bottlenecks, implementation code, and paper context.",
  },
  {
    id: "bert",
    label: "BERT base",
    publishedDate: "2018-10-11",
    title: "BERT Base Architecture",
    description: "Inspect BERT embeddings, bidirectional encoder layers, implementation code, and paper context.",
  },
  {
    id: "gpt2",
    label: "GPT-2 small",
    publishedDate: "2019-02-14",
    title: "GPT-2 Small Architecture",
    description: "Trace GPT-2 decoder-only attention blocks with visual architecture, code, and paper context.",
  },
  {
    id: "efficientnet",
    label: "EfficientNet-B0",
    publishedDate: "2019-05-28",
    title: "EfficientNet-B0 Architecture",
    description: "Inspect EfficientNet MBConv stages, squeeze-excitation blocks, code, and paper context.",
  },
  {
    id: "ddpm",
    label: "DDPM",
    publishedDate: "2020-06-19",
    title: "DDPM Diffusion U-Net Architecture",
    description: "Trace DDPM forward noising, timestep-conditioned U-Net noise prediction, reverse denoising steps, implementation code, and paper context.",
  },
  {
    id: "vit",
    label: "ViT-B/16",
    publishedDate: "2020-10-22",
    title: "Vision Transformer Architecture",
    description: "Explore ViT patch embeddings, transformer encoder blocks, implementation code, and paper context.",
  },
  {
    id: "clip",
    label: "CLIP",
    publishedDate: "2021-02-26",
    title: "CLIP Architecture",
    description: "Trace CLIP dual image-text encoders, shared embedding projections, contrastive logits, implementation code, and paper context.",
  },
] as const satisfies readonly ModelRouteSummary[];

export function getModelRoute(modelId: string) {
  return modelRouteSummaries.find((model) => model.id === modelId);
}

export function modelRoutePath(modelId: string) {
  return `/models/${modelId}`;
}

export function siteUrl() {
  return new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "https://modelarch.tsilva.eu");
}

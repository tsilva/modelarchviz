export type ModelCatalogEntry = {
  id: string;
  label: string;
  publishedDate: string;
  title: string;
  description: string;
  sourceBaseName: string;
  paper: {
    title: string;
    authors: string;
    venue: string;
    focus: string[];
  };
};

export const modelCatalog = [
  {
    id: "mlp",
    label: "MLP",
    publishedDate: "1986-10-09",
    title: "MLP Architecture",
    description: "Inspect a multilayer perceptron architecture alongside PyTorch, JAX, and source-paper context.",
    sourceBaseName: "mlp",
    paper: {
      title: "Learning representations by back-propagating errors",
      authors: "David E. Rumelhart, Geoffrey E. Hinton, Ronald J. Williams",
      venue: "Nature",
      focus: ["backpropagation", "hidden representations", "multilayer perceptrons"],
    },
  },
  {
    id: "rnn",
    label: "RNN (Elman)",
    publishedDate: "1990-03-01",
    title: "Elman RNN Architecture",
    description: "Explore an Elman recurrent neural network with architecture blocks, code, and paper context.",
    sourceBaseName: "elman_rnn",
    paper: {
      title: "Finding Structure in Time",
      authors: "Jeffrey L. Elman",
      venue: "Cognitive Science",
      focus: ["recurrent hidden state", "dynamic memory", "sequence structure"],
    },
  },
  {
    id: "lstm",
    label: "LSTM",
    publishedDate: "1997-11-01",
    title: "LSTM Architecture",
    description: "Use the LSTM architecture card to trace sequence inputs, input/forget/output gates, cell-state updates, PyTorch and JAX code, notebooks, and the original paper.",
    sourceBaseName: "lstm",
    paper: {
      title: "Long Short-Term Memory",
      authors: "Sepp Hochreiter, Jurgen Schmidhuber",
      venue: "Neural Computation",
      focus: ["cell state memory", "input/forget/output gates", "long-range dependencies"],
    },
  },
  {
    id: "autoencoder",
    label: "Autoencoder",
    publishedDate: "2006-07-28",
    title: "Autoencoder Architecture",
    description: "Trace an autoencoder encoder, bottleneck code, decoder reconstruction path, reconstruction loss, implementation code, and paper context.",
    sourceBaseName: "autoencoder",
    paper: {
      title: "Reducing the Dimensionality of Data with Neural Networks",
      authors: "Geoffrey E. Hinton, Ruslan R. Salakhutdinov",
      venue: "Science",
      focus: ["dimensionality reduction", "encoder-decoder reconstruction", "bottleneck representations"],
    },
  },
  {
    id: "lenet5",
    label: "LeNet-5",
    publishedDate: "1998-11-01",
    title: "LeNet-5 Architecture",
    description: "Study the LeNet-5 convolutional network through its architecture, code, and original paper context.",
    sourceBaseName: "lenet5",
    paper: {
      title: "Gradient-Based Learning Applied to Document Recognition",
      authors: "Yann LeCun, Leon Bottou, Yoshua Bengio, Patrick Haffner",
      venue: "Proceedings of the IEEE",
      focus: ["convolutional feature maps", "subsampling", "document recognition"],
    },
  },
  {
    id: "alexnet",
    label: "AlexNet",
    publishedDate: "2012-12-03",
    title: "AlexNet Architecture",
    description: "Inspect AlexNet convolutional stages, classifier layers, implementation code, and paper context.",
    sourceBaseName: "alexnet",
    paper: {
      title: "ImageNet Classification with Deep Convolutional Neural Networks",
      authors: "Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton",
      venue: "NeurIPS 2012",
      focus: ["large-scale CNNs", "ReLU activations", "GPU training"],
    },
  },
  {
    id: "vgg16",
    label: "VGG-16",
    publishedDate: "2014-09-04",
    title: "VGG-16 Architecture",
    description: "Inspect VGG-16 stacked 3x3 convolutional stages, dense classifier layers, implementation code, and paper context.",
    sourceBaseName: "vgg16",
    paper: {
      title: "Very Deep Convolutional Networks for Large-Scale Image Recognition",
      authors: "Karen Simonyan, Andrew Zisserman",
      venue: "arXiv / ICLR 2015",
      focus: ["deep plain CNNs", "3x3 convolution stacks", "ImageNet classification"],
    },
  },
  {
    id: "vae",
    label: "VAE",
    publishedDate: "2013-12-20",
    title: "Variational Autoencoder Architecture",
    description: "Trace a variational autoencoder Gaussian encoder, reparameterization trick, decoder, ELBO loss, implementation code, and paper context.",
    sourceBaseName: "vae",
    paper: {
      title: "Auto-Encoding Variational Bayes",
      authors: "Diederik P. Kingma, Max Welling",
      venue: "arXiv / ICLR 2014",
      focus: ["variational inference", "reparameterization trick", "latent-variable generative models"],
    },
  },
  {
    id: "gan",
    label: "GAN",
    publishedDate: "2014-06-10",
    title: "Generative Adversarial Network Architecture",
    description: "Trace the original GAN generator, discriminator, adversarial losses, alternating training updates, implementation code, and paper context.",
    sourceBaseName: "gan",
    paper: {
      title: "Generative Adversarial Nets",
      authors: "Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio",
      venue: "arXiv / NeurIPS 2014",
      focus: ["adversarial training", "generator-discriminator game", "implicit generative models"],
    },
  },
  {
    id: "gru",
    label: "GRU",
    publishedDate: "2014-06-03",
    title: "GRU Architecture",
    description: "Explore GRU update and reset gates with architecture blocks, code, and source-paper context.",
    sourceBaseName: "gru",
    paper: {
      title: "Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation",
      authors: "Kyunghyun Cho, Bart van Merrienboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, Yoshua Bengio",
      venue: "arXiv / EMNLP 2014",
      focus: ["update gate", "reset gate", "encoder-decoder sequence modeling"],
    },
  },
  {
    id: "seq2seq",
    label: "Seq2Seq",
    publishedDate: "2014-09-10",
    title: "Seq2Seq Architecture",
    description: "Inspect recurrent encoder-decoder sequence transduction with PyTorch, JAX, and source-paper context.",
    sourceBaseName: "seq2seq",
    paper: {
      title: "Sequence to Sequence Learning with Neural Networks",
      authors: "Ilya Sutskever, Oriol Vinyals, Quoc V. Le",
      venue: "arXiv / NeurIPS 2014",
      focus: ["encoder-decoder LSTMs", "fixed-length context", "sequence transduction"],
    },
  },
  {
    id: "googlenet",
    label: "GoogLeNet / Inception v1",
    publishedDate: "2014-09-17",
    title: "GoogLeNet Inception Architecture",
    description: "Inspect GoogLeNet Inception modules, branches, implementation code, and paper context.",
    sourceBaseName: "googlenet",
    paper: {
      title: "Going Deeper with Convolutions",
      authors: "Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich",
      venue: "arXiv / CVPR 2015",
      focus: ["Inception modules", "parallel convolutions", "channel concatenation"],
    },
  },
  {
    id: "unet",
    label: "U-Net",
    publishedDate: "2015-05-18",
    title: "U-Net Architecture",
    description: "Trace U-Net encoder-decoder skips with visual architecture blocks, code, and paper context.",
    sourceBaseName: "unet",
    paper: {
      title: "U-Net: Convolutional Networks for Biomedical Image Segmentation",
      authors: "Olaf Ronneberger, Philipp Fischer, Thomas Brox",
      venue: "arXiv / MICCAI 2015",
      focus: ["encoder-decoder segmentation", "skip concatenations", "biomedical images"],
    },
  },
  {
    id: "resnet18",
    label: "ResNet",
    publishedDate: "2015-12-10",
    title: "ResNet Architecture Variants",
    description: "Inspect ResNet-18 through ResNet-152 residual stages, generated implementation code, and paper context.",
    sourceBaseName: "resnet18",
    paper: {
      title: "Deep Residual Learning for Image Recognition",
      authors: "Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun",
      venue: "arXiv / CVPR 2016",
      focus: ["identity shortcuts", "residual blocks", "very deep CNNs"],
    },
  },
  {
    id: "widenet",
    label: "WideNet",
    publishedDate: "2016-05-23",
    title: "Wide Residual Network Architecture",
    description: "Explore wide residual blocks with architecture diagrams, PyTorch and JAX code, and paper context.",
    sourceBaseName: "widenet",
    paper: {
      title: "Wide Residual Networks",
      authors: "Sergey Zagoruyko, Nikos Komodakis",
      venue: "arXiv / BMVC 2016",
      focus: ["widened residual blocks", "feature reuse", "CIFAR image classification"],
    },
  },
  {
    id: "densenet",
    label: "DenseNet-121",
    publishedDate: "2016-08-25",
    title: "DenseNet-121 Architecture",
    description: "Inspect DenseNet feature concatenation, dense blocks, implementation code, and paper context.",
    sourceBaseName: "densenet",
    paper: {
      title: "Densely Connected Convolutional Networks",
      authors: "Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger",
      venue: "arXiv / CVPR 2017",
      focus: ["dense connectivity", "feature reuse", "vanishing-gradient mitigation"],
    },
  },
  {
    id: "transformer",
    label: "Transformer",
    publishedDate: "2017-06-12",
    title: "Transformer Architecture",
    description: "Explore encoder-decoder attention blocks, PyTorch and JAX code, and source-paper context.",
    sourceBaseName: "transformer",
    paper: {
      title: "Attention Is All You Need",
      authors: "Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin",
      venue: "NeurIPS 2017",
      focus: ["scaled dot-product attention", "encoder-decoder stacks", "positional encoding"],
    },
  },
  {
    id: "vqvae",
    label: "VQ-VAE",
    publishedDate: "2017-11-02",
    title: "VQ-VAE Architecture",
    description: "Trace VQ-VAE encoder outputs, nearest-code vector quantization, straight-through estimator, decoder reconstruction, implementation code, and paper context.",
    sourceBaseName: "vqvae",
    paper: {
      title: "Neural Discrete Representation Learning",
      authors: "Aaron van den Oord, Oriol Vinyals, Koray Kavukcuoglu",
      venue: "arXiv / NeurIPS 2017",
      focus: ["vector quantization", "discrete latent codes", "straight-through estimator"],
    },
  },
  {
    id: "mobilenetv2",
    label: "MobileNetV2",
    publishedDate: "2018-01-13",
    title: "MobileNetV2 Architecture",
    description: "Inspect MobileNetV2 inverted residual blocks, depthwise separable convolutions, linear bottlenecks, implementation code, and paper context.",
    sourceBaseName: "mobilenetv2",
    paper: {
      title: "MobileNetV2: Inverted Residuals and Linear Bottlenecks",
      authors: "Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen",
      venue: "arXiv / CVPR 2018",
      focus: ["inverted residuals", "linear bottlenecks", "mobile-efficient CNNs"],
    },
  },
  {
    id: "bert",
    label: "BERT base",
    publishedDate: "2018-10-11",
    title: "BERT Base Architecture",
    description: "Inspect BERT embeddings, bidirectional encoder layers, implementation code, and paper context.",
    sourceBaseName: "bert_base",
    paper: {
      title: "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
      authors: "Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova",
      venue: "arXiv / NAACL 2019",
      focus: ["masked language modeling", "bidirectional encoders", "fine-tuning"],
    },
  },
  {
    id: "gpt2",
    label: "GPT-2 small",
    publishedDate: "2019-02-14",
    title: "GPT-2 Small Architecture",
    description: "Trace GPT-2 decoder-only attention blocks with visual architecture, code, and paper context.",
    sourceBaseName: "gpt2_attention",
    paper: {
      title: "Language Models are Unsupervised Multitask Learners",
      authors: "Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever",
      venue: "OpenAI technical report",
      focus: ["decoder-only transformers", "causal language modeling", "zero-shot transfer"],
    },
  },
  {
    id: "efficientnet",
    label: "EfficientNet-B0",
    publishedDate: "2019-05-28",
    title: "EfficientNet-B0 Architecture",
    description: "Inspect EfficientNet MBConv stages, squeeze-excitation blocks, code, and paper context.",
    sourceBaseName: "efficientnet",
    paper: {
      title: "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
      authors: "Mingxing Tan, Quoc V. Le",
      venue: "arXiv / ICML 2019",
      focus: ["compound scaling", "mobile inverted bottlenecks", "squeeze-and-excitation"],
    },
  },
  {
    id: "ddpm",
    label: "DDPM",
    publishedDate: "2020-06-19",
    title: "DDPM Diffusion U-Net Architecture",
    description: "Trace DDPM forward noising, timestep-conditioned U-Net noise prediction, reverse denoising steps, implementation code, and paper context.",
    sourceBaseName: "ddpm",
    paper: {
      title: "Denoising Diffusion Probabilistic Models",
      authors: "Jonathan Ho, Ajay Jain, Pieter Abbeel",
      venue: "arXiv / NeurIPS 2020",
      focus: ["forward diffusion", "noise prediction", "iterative denoising"],
    },
  },
  {
    id: "vit",
    label: "ViT-B/16",
    publishedDate: "2020-10-22",
    title: "Vision Transformer Architecture",
    description: "Explore ViT patch embeddings, transformer encoder blocks, implementation code, and paper context.",
    sourceBaseName: "vit_b16",
    paper: {
      title: "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale",
      authors: "Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, Neil Houlsby",
      venue: "arXiv / ICLR 2021",
      focus: ["image patches as tokens", "class token", "Transformer encoders for vision"],
    },
  },
  {
    id: "clip",
    label: "CLIP",
    publishedDate: "2021-02-26",
    title: "CLIP Architecture",
    description: "Trace CLIP dual image-text encoders, shared embedding projections, contrastive logits, implementation code, and paper context.",
    sourceBaseName: "clip",
    paper: {
      title: "Learning Transferable Visual Models From Natural Language Supervision",
      authors: "Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever",
      venue: "arXiv / ICML 2021",
      focus: ["natural language supervision", "dual encoders", "contrastive image-text pretraining"],
    },
  },
] as const satisfies readonly ModelCatalogEntry[];

export type ModelId = (typeof modelCatalog)[number]["id"];

export function getModelRoute(modelId: string) {
  return modelCatalog.find((model) => model.id === modelId);
}

export function modelRoutePath(modelId: string) {
  return `/models/${modelId}`;
}

export function siteUrl() {
  return new URL(process.env.NEXT_PUBLIC_SITE_URL ?? "https://modelarch.tsilva.eu");
}

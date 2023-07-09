---
layout: post
title: How does Stable Diffusion work
categories:
  - Reading
published: true
---

本文翻译 https://stable-diffusion-art.com/how-stable-diffusion-work/

Stable Diffusion: 稳定扩散

Stable Diffusion是一种深度学习模型。

# 稳定扩散能做什么

最简单的形式，稳定扩散是一种文本到图像模型（text-to-image）。给它一个文字提示（text prompt），将返回与文本匹配的图像。
![Stable diffusion将文字提示转变成图像](/assets/images/Stable Diffusion/Pasted%20image%2020230703233013.png)

# 扩散模型

Stable Diffusion属于一类称为扩散模型（diffusion models）的深度学习模型，是旨在生成与训练中类似数据的生成模型。在稳定Stable Diffusion的情况下，数据是图像。

为什么称为扩散模型？是由于它的数学看起来像物理学中的扩散。

假设只用两张图训练了一个扩散模型：猫和狗。在下图中，左边的两个峰代表猫和狗图像组。

## 前向扩散

![前向扩散将照片变成噪声](/assets/images/Stable Diffusion/Pasted%20image%2020230704130912.png)

前向扩散（Forward diffusion）过程中，向训练图像添加噪声，逐渐将其变成无特征的噪声图像。最终，将无法分辨图片最初是狗或者猫。

就像一滴墨水落入一杯水中，墨滴在水中扩散，几分钟后，随机分布在整个水中，无法判断最初是落在中心还是边缘附近。

![猫图像的前向扩散](/assets/images/Stable Diffusion/Pasted%20image%2020230704131550.png)

## 反向扩散

如何逆转扩散？像倒放视频一样，倒退时间，将会看见墨滴最初滴落的位置。

![反向扩散过程恢复图像](/assets/images/Stable Diffusion/Pasted%20image%2020230704204244.png)

从技术上讲，每个扩散都有两个部分：（1）漂移（drift）、（2）随机运动（random）。反向扩散图像会偏向猫或狗，而不会介于两者中间。

# 训练是如何进行的

反向扩散的想法是巧妙而优雅的。

为了反转扩散，需要知道图像中添加了多少噪声——通过训练神经网络模型来预测添加的噪声。在Stable Diffusion中，被称为噪声预测器——一个U-Net模型。训练过程如下：

1. 选择一张训练图像，比如猫。
2. 生成随机噪声图像。
3. 通过一定数量的步骤，添加噪声图片来破坏训练的图像。
4. 教噪声训练器反馈添加的噪声数量——通过调整权重并且向他展示正确答案。

![按步骤依次添加噪声，噪声预测器估算每一步添加的总噪声](/assets/images/Stable Diffusion/Pasted%20image%2020230704214357.png)

训练后，具备一个能够估计添加到图像中的噪声的噪声预测器。

## Reverse diffusion

如何使用噪声预测器？

首先生成一个完全随机的图像，让噪声预测器告知噪声。然后从原始图像中减去估计的噪声，重复此过程几次，将获得猫或狗的图像。

![反向扩散的工作原理是连续从图像中减去预测的噪声](/assets/images/Stable Diffusion/Pasted%20image%2020230704233511.png)

但是目前无法控制生成猫或狗的图像，图像生成是无条件的。

有关反向扩散采样和采样器的[更多信息](https://stable-diffusion-art.com/samplers/_)。

# 稳定扩散模型

上述扩散过程是在图像空间中进行的，不是稳定扩散的工作原理，无法在单个GPU上运行。

图像空间是巨大的。试想：三个颜色通道（红、蓝、绿）的512×512图像具有786432维的空间。

类似[Imagen](https://imagen.research.google/)和[DALL-E](https://openai.com/dall-e-2/)的扩散模型都在像素空间中，虽然使用了一些技巧来加速模型，但是仍然不够。

## 潜在扩散模型（Latent diffusion model）

Stable Diffusion旨在解决速度问题。

Stable Diffusion是一种潜在扩散模型。它不是在高维图像空间中操作，而是首先将图像压缩到潜在空间中。潜在空间小了48倍，因此降低了处理次数——变快的原因。

## 变分自动编码器（Variational Autoencoder）

上述过程是通过一种变分自动编码器的技术实现。

变分自动编码器神经网络有两部分：（1）编码器、（2）解码器。编码器将图像压缩为潜在空间中的低维表示，解码器从潜在空间中恢复图像。

![变分自动编码器将图像转换到潜在空间或从潜在空间转换图像](/assets/images/Stable Diffusion/Pasted%20image%2020230705184739.png)

稳定扩散模型的潜在空间是4×64×64，比图像像素空间小48倍。所有前向和反向扩散实际上都是在潜在空间中完成的。

因此，在训练过程中，不会生成噪声图像，而是在潜在空间（潜在噪声）中生成随机张量。不是用噪声破坏图像，而是用潜在噪声破坏图像在潜在空间中的表示。由于潜在空间小，所以速度快很多。

## 图像分辨率（Image resolution）

图像分辨率反映在潜在图像张量上，512×512图像的潜在图像大小仅为4×64×6，768×512肖像图的潜在图像为4×96×96。因此需要更长、更多的VRAM才能生成更大的图像。

由于Stable Diffusion v1是在512×512图像上进行了微调（fine-tuned），因此生成大于512×512的图像可能会得到重复的图像，比如两个头。如果是必须的，至少在一侧保留512像素，并用[AI升级器](https://stable-diffusion-art.com/ai-upscaler/)获得更高的分辨率。

## 为什么潜在空间是可能的（Why is latent space possible）

为什么VAE可以将图像压缩到更小的潜在空间而不丢失信息——原因是自然图像不是随机的，它们具有高度的规律性：面部遵循眼睛、鼻子、脸颊和嘴巴之间的特定空间关系。换而言之，图像的高维性是认为的。自然图像可以很容易地压缩到更小的潜在空间中，而不会丢失任何信息——在机器学习中被称为流行假设（manifold hypothesis）。

## 潜在空间中的反向扩散（Reverse diffusion in latent space）

稳定扩散中潜在反向扩散的工作原理：

1. 生成随机潜在空间矩阵。
2. 噪声预测器估算潜在矩阵的噪声。
3. 从潜在矩阵中减去估算的噪声。
4. 重复步骤2&3直至达到特定的采样步骤。
5. VAE解码器将潜在矩阵转换为最终图像。

## VAE文件是什么（What is a VAE file）

在Stable Diffusion v1中，VAE文件被用来改善眼睛和面部，即前面提到的自动编码器的解码器。通过进一步微调解码器，模型可以绘制更精细的细节。

将图像压缩到潜在空间会丢失信息，是因为原始VAE无法恢复精细细节，但是，VAE解码器能够负责绘制精细的细节。

# 调理（Conditioning）

目前我们的理解尚不完整：文字提示如何进入图片？没有文本，stable diffusion就不是text-to-image 模型，将会得到无法控制的猫或狗的图像。

所以需要用到调理——引导噪声预测器，以便在从图像中减去预测的噪声后，能够提供预想的图像。

## 文本调节（Text conditioning「text-to-image」）

Tokenizer首先将提示中的每个单词转换为称为词元（token）的数字，然后每个token被转换为768个值的向量——嵌入（embedding）。这些embeddings随后由文本转换器（text transformer）进行处理，提供给噪声预测器使用。

![如何处理文本提示并将其输入噪声预测器以引导图像生成](/assets/images/Stable Diffusion/Pasted%20image%2020230705232627.png)

### 分词器（Tokenizer）

![分词器](/assets/images/Stable Diffusion/Pasted%20image%2020230705233056.png)

词元化（tokenization）是计算机理解单词的方式。人类可以读取文字，但是计算机只能读取数字，所以需要将文本提示中的单词转换为数字。

分词器只能对训练期间的单词进行分词。例如，CLIP模型中有“dream”和“beach”，但没有“dreambeanch”。分词器会将“dreambeach”一词分解为两个词元，“dream”和“beach”，所以一个词并不总意味一个词元。

另外空格字符也是词元的一部分。上述例子中，“dream beach”和“dreambeach”生成的词元不同。

Stable Diffusion模型仅限于在提示中使用75个词元。

### 嵌入（Embedding）

![Embedding](/assets/images/Stable Diffusion/Pasted%20image%2020230706234815.png)

Stable diffusion v1使用Open AI的ViT-L/14 Clip模型，其中Embedding是一个768个值的向量，每一个词元都有自己独特的嵌入向量。Embedding由在训练过程中学习的CLIP模型固定。

为什么需要embedding？因为有些词彼此之间密切相关，而我们想要利用这些信息。例如，man、gentleman、guy的嵌入几乎相同，因为它们可以互相转换使用。Monet、Manet、Degas都是用不同的方式以印象派风格作画，这些名称具有接近但不相同的嵌入。

这与使用关键词触发的嵌入相同，嵌入可以发挥魔法。科学家已经证实找到正确的嵌入可以触发任意对象和样式——一种称为文本反转（textual inversion）的微调技术。

### 将嵌入提供给噪声预测器（Feeding embeddings to noise predictor）

![从嵌入到噪声预测器](/assets/images/Stable Diffusion/Pasted%20image%2020230706235738.png)

在输入噪声预测器之前，嵌入需要由文本转换器（text transformer）进一步处理。transformer像一个通用适配器，输入文本嵌入向量，也可以像类标签、图像、深度图之类的其它东西。transformer不仅进一步处理数据，也提供提供一种包含不同调节模式/训练方式（conditioning modalities）的机制。

### 交叉注意力（Cross-attention）

整个U-Net中的噪声预测器多次使用文本转换器的输出，U-Net通过交叉注意力机制来消耗它，即提示（prompt）和图像（image）的结合处。

以提示“A man with blue eyes”为例，stable diffusion将“blue”、“eyes”两个词配对在一起（提示中的自我关注），这样即会生成一个蓝眼睛的男人，而不是一个穿蓝色衬衫的男人。然后，使用此信息引导反向扩散至包含蓝眼睛的图像（提示和图像之间的交叉注意力）。

旁注：超网络（Hypernetwork），一种调优稳定扩散模型的技术，通过劫持交叉注意力网络来插入样式。LoRA models修改交叉注意力模块的权重来改变风格，单独修改这个模块就可以调优Stable Diffusion模型。

## 其他条件（Other conditionings）

文本提示并不是调节Stable Diffusion模型的唯一方法。文本提示（text prompt）和深度图像（depth image）都用于调节深度到图像（depth-to-image）模型。

ControlNet使用检测到的轮廓、人体姿势等来调节噪声预测器，并实现对图像生成的出色控制。

# 逐步稳定扩散（Stable Diffusion step-by-step）

## 文本转图像（Text-to-image）

在文本转图像中，给stable diffusion提供文本提示，将返回图像。

Step 1：Stable Diffusion在潜在空间中生成随机张量，可以通过设置随机数生成器种子在控制该张量。如果将种子设置为某个值，将始终获得相同的随机张量——即潜在空间中的图像，目前全是噪音。

![在潜在空间中生成随机张量](/assets/images/Stable Diffusion/Pasted%20image%2020230707185212.png)

Step 2：噪声预测器U-Net将潜在噪声图像和文本提示作为输入，并在潜在空间（一个4×64×64张量）中预测噪声。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707185345.png)

Step 3：从潜在空间中减去潜在噪声，得到新的潜像。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707185428.png)

重复步骤2&步骤3一定数量的采样步骤。

Step 4：最后，VAE的解码器将潜在图像转换回像素空间，得到运行稳定扩散后的图像。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707185557.png)

下面是图像在每个采样步骤中的演变方式。

![每个采样步骤的图像](/assets/images/Stable%20Diffusion/cat_euler_15.webp)

## 噪音表（Noise schedule）

图像从模糊变得清晰（noisy to clean），试图得到每个采样步骤中获得预期的噪声——噪声表（noise schedule）。

![15个采样步骤的噪声表](/assets/images/Stable Diffusion/Pasted%20image%2020230707190337.png)

噪声序列表是我们定义的，可以选择在每一步减去相同数量的噪声，或者在开始时减去更多。采样器在每一步中减去足够的噪声，以在下一步中达到预期噪声。

## 图像到图像（Image-to-image）

Image-to-image是[SDEdit](https://arxiv.org/abs/2108.01073)方法中首次提出的方法，可应用于任何扩散模型。因此，我们有图像到图像的稳定扩散（潜在扩散模型）。

输入图像和文本提示被输入到image-to-image，生成的图像将受到输入图像和文本提示的限制。例如，使用这张业余绘画和提示“photo of perfect green apple with stem, water droplets, dramatic lighting”作为输入，图像到图像可以将其变成专业绘画。

![图像到图像](/assets/images/Stable Diffusion/Pasted%20image%2020230707222138.png)

下面是分布过程：

Step 1：输入图像被编码到潜在空间。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707222634.png)

Step 2：将噪声添加到潜像中，去噪强度（denoising strength）控制添加的噪声量。如果为0，不添加噪声；如果为1，添加最大量的噪声，使潜像称为完全随机的张量。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707223309.png)

Step 3：噪声预测器U-Net将潜在噪声图像和文本提示作为输入，并预测潜在空间（4×64×64张量）中的噪声。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707223512.png)

Step 4：从潜在图像中减去潜在噪声，变成新的潜像。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707224025.png)

重复步骤3&步骤4至采样步骤的一定数量。

Step 5：最后，VAE的解码器将潜像转换回像素空间，得到运行image-to-image后的图像。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707224502.png)

所以，图像到图像——设置带有一点噪声和一点输入图像的初始潜在图像，设置去噪强度为1相当于文本转图像，因为初始潜像完全是随机噪声。

## 修复

修复只是图像到图像的一种特殊情况，噪声会被添加到想要修复的图像部分，噪声量同样有降噪强度控制。

## 深度到图像（depth-to-image）

深度到图像是图像到图像的增强，使用深度图（depth map）生成带有附加条件的新图像。

Step 1：输入图像被编码为潜在状态。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707225704.png)

Step 2：[MiDaS](https://github.com/isl-org/MiDaS)（一种AI深度模型）根据输入图像估算深度图。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707225815.png)

Step 3：将噪声添加到潜像中，去噪强度控制被添加的噪声量。如果去噪强度为0，不添加噪声；去噪强度为1，添加最大噪声，使得潜像变成随机张量。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707230140.png)

Step 4：噪声预测器根据文本提示和深度图估算潜在空间的噪声。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707230247.png)

Step 5：从潜像中减去潜在噪声，得到新的潜像。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707230315.png)

重复采样步骤数的步骤4&5。

Step 6：VAE的解码器对潜像进行解码，将获得从深度到图像的最终图像。

![](/assets/images/Stable Diffusion/Pasted%20image%2020230707230522.png)

# CFG值是什么（What is CFG value）

无分类器指导（Classifier-Free Guidance）（CFG），前身：分类指导（classifier guidance）。

## 分类器指导（classifier guidance）

[分类器](https://arxiv.org/abs/2105.05233)指导是一种将图像标签（image labels）合并到扩散模型中的方法，可以使用标签来指导扩散过程。例如，标签“cat”引导反向扩散过程生成猫的照片。

分类器指导尺度（classifier guidance scale）是控制扩散过程遵循标签程度的参数。

例如，假设有3组图像，标签为“cat”、“dog”、“human”。若扩散是无引导的，模型将从每组总的群体抽取样本，但是有时候可能会抽取适合两个标签的图像，比如一个男孩在抚摸一只狗。

![分类器指导，左：无引导，中：小指导程度，有：大指导程度](/assets/images/Stable Diffusion/Pasted%20image%2020230708175140.png)

在高分类器指导下，扩散模型生成的图像将偏向极端或明确的示例。如果向模型寻求一只猫，它会返回一张明确是猫的图像，除此之外别无他法。

分类器指导尺度控制指导的遵循程度。上图中，右边的采样比中间的采样具有更高的分类器指导尺度。实际上，该比例值只是带有该标签数据的漂移项的乘数。

## 无分类器指导（classifier-free guidance）

尽管分类器指导实现了破纪录的性能，但它需要一个额外的模型来提供该指导，这给训练带来了一些困难。

无分类指导（[classifier-free guidance](**[Classifier-free guidance](https://arxiv.org/abs/2207.12598)**)）是一种实现“没有分类器的分类器指导”的方法，没有使用类标签和单独的模型进行指导，而是使用图像标题训练条件扩散模型（conditional diffusion model），就像上述文本到图像的模型一样。

将分类器部分作为调节噪声预测器U-Net，实现图像生成中的“无分类器”指导。

### CFG值（CFG value）

通过调节获得了一个无分类器扩散过程，如何控制遵循指导的量？

无分类器指导比例是一个控制文本提示对扩散过程影响程度的值，当值为0时，图像生成是无条件的（无提示），较高的值将引导扩散至提示。

# Stable Diffusion v1 vs v2

## 模型差异（Model difference）

Stable Diffusion v2使用[OpenClip](https://stability.ai/blog/stable-diffusion-v2-release)进行文本嵌入，Stable Diffusion使用Open AI的CLIP [ViT-L/14](https://github.com/CompVis/stable-diffusion)进行文本嵌入：

- OpenClip扩大了5倍更大的文本编码器模型可以提高图像质量。
- 尽管Open AI的CLIP模型是开源的，但是这些模型使用专有数据进行训练。切换到OpenClip模型使研究和优化模型更透明，利于长远发展。

## 训练数据差异（Training data difference）

Stable Diffusion v1.4训练：

- laion2B-en数据集上以256×256的分辨率进行237k步（237k steps at resolution 256×256 on [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) dataset.）
- laion-high-resolution上以512×512分辨率进行194k步（194k steps at resolution 512×512 on [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution).）
- 文本调节降低10%，laion-aesthetics v2 5+上以512×512进行225k步（225k steps at 512×512 on “[laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/)“, with 10% dropping of text conditioning.）

Stable Diffusion v2训练：

- 550k steps at the resolution `256x256` on a subset of [LAION-5B](https://laion.ai/blog/laion-5b/) filtered for explicit pornographic material, using the [LAION-NSFW classifier](https://github.com/LAION-AI/CLIP-based-NSFW-Detector) with `punsafe=0.1` and an [aesthetic score](https://github.com/christophschuhmann/improved-aesthetic-predictor) >= `4.5`.
- 850k steps at the resolution `512x512` on the same dataset on images with resolution `>= 512x512`.
- 150k steps using a [v-objective](https://arxiv.org/abs/2202.00512) on the same dataset.
- Resumed for another 140k steps on `768x768` images.

Stable Diffusion v2.1在v2.0上进行了微调：

- additional 55k steps on the same dataset (with `punsafe=0.1`)
- another 155k extra steps with `punsafe=0.98`

所以基本上，都在最后的训练步骤中关闭了NSFW过滤器。

## 结果差异（Outcome difference）

用户通常发现使用Stable Diffusion v2来控制风格和生成名人更困难，尽管Stability AI没有明确过滤艺术家和名人的名字，但是它们的效果在v2中很弱，可能是由于训练数据的差异造成的。Open AI的专有数据可能有更多艺术品和名人照片，数据可经过贵都过滤，因此看起来都更好。
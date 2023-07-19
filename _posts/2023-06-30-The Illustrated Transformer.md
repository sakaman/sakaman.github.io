---
layout: post
title: What Is a Transformer Model
categories:
  - Reading
published: true
---

本文翻译自： 
- https://blogs.nvidia.com/blog/2022/03/25/what-is-a-transformer-model/
- http://jalammar.github.io/illustrated-transformer/

# Transformer模型是什么？

Transformer模型是一种神经网络，通过跟踪连续数据中的关系来学习上下文并从理解含义。

Transformer模型应用一组称为注意或者自注意（attention or self-attention）的数学技术，来检测疏远数据元素（distant data elements）之间不明显的一系列相互影响、依赖的方式。

Google 2017年的一篇[论文](https://arxiv.org/abs/1706.03762)首次描述了[Transformer]([guide annotating the paper with PyTorch implementation](http://nlp.seas.harvard.edu/2018/04/03/attention.html))，它是迄今为止被发明的模型中最新、最强大的类别之一。推动着机器学习领域的进步，有些被称为“变形人工智能”（transformer AI）。

斯坦福大学的研究人员在2021年8月的一篇[论文]([August 2021 paper](https://arxiv.org/pdf/2108.07258.pdf))中将Transformer称为“基础模型”（foundation models）。

# Transformer模型能做什么？

Transformer可以近实时地翻译文本和语音，向多元化和听力障碍的与会人员开放会议和课堂。

帮助研究人员了解DNA中的基因链和蛋白质中的氨基酸，从而加快药物设计。

![Transformer, sometimes called foundation models, are already being used with many data sources for a host of applications](/assets/images/2023-06/Transformer%20Application.png)

# 高层级视角（A High-Level Look）

首先将该模型视为一个黑盒，在机器翻译应用程序中，输入一种语言，返回另一种语言。

![](/assets/images/2023-06/Pasted%20image%2020230709232455.png)

解剖中间，可以得到一个编码组件、一个解码组件以及它们间的链接。

![](/assets/images/2023-06/Pasted%20image%2020230709232826.png)

编码组件是一堆编码器，解码组件是相同数量解码器的堆栈。

![](/assets/images/2023-06/Pasted%20image%2020230710235303.png)

这些解码器的结构是相同的（不共享权重），每一层分为两个子层：

![](/assets/images/2023-06/Pasted%20image%2020230711000038.png)

编码器的输入首先流经自注意层——帮助编码器在对特定单词编码时查看输入句子中的其他单词。自注意层的输出被馈送到前馈神经网络（feed-forward neural network），独立应用完全相同的前馈神经网络到每个位置。

解码器具有这两层，但中间有一个帮助解码器关注输入句子的相关部分的注意层。

![](/assets/images/2023-06/Pasted%20image%2020230711000603.png)

# 张量代入图片（Bringing The Tensors Into The Picture）

接下来了解各种向量/张量以及他们如何在这些组件之间流动，将训练模型的输入转换为输出。

与NLP应用中的一般情况一样，首先使用嵌入算法（[embedding algorithm](https://medium.com/deeper-learning/glossary-of-deep-learning-word-embedding-f90c3cec34ca)）将每个单词转换为向量。

![每个单词都嵌入到大小为512的向量中](/assets/images/2023-06/Pasted%20image%2020230711130619.png)

嵌入（embedding）仅发生在最底部的编码器中。所有编码器共有的抽象是，接收每个大小为512的向量列表——在底部编码器中，会是单词嵌入，但在其他编码器中，将是下面编码器的输出。列表的大小可以设置——基本上是训练数据集中最长句子的长度。将单词嵌入到输入序列后，每个单词都回流经编码器的两层。

![](/assets/images/2023-06/Pasted%20image%2020230711131100.png)

这里有Transformer的一个关键属性，即每个位置的单词在编码器中都流经自有路径。自注意层在这些路径中存在依赖关系。然而，前馈层不具有这些依赖性，因此各种路径可以在流经前馈层时并行处理。

# 编码（Encoding）

编码器接收向量列表作为输入，通过将这些向量传递到“自注意”层（self-attention layer），然后传递到前馈神经网络，将输出向上发送到下一个编码器来处理。

![每个位置的单词都会经过一个自注意过程，然后通过前馈神经网络——每个向量分别流经的完全相同的网络](/assets/images/2023-06/Pasted%20image%2020230711131515.png)

# 高层次自注意（Self-attention at a High Level）

当模型处理每个单词（输入序列中的每个位置）时，自注意允许它查看输入序列中的其他位置来寻找有助于更好地编码该单词的线索。

自注意是Transformer用来将其他相关单词的“理解”融入正在处理的单词中的方法。

![](/assets/images/2023-06/Pasted%20image%2020230711232519.png)
<center>当在编码器#5（堆栈中的顶部编码器）中对单词“it”进行编码时，注意机制的一部分集中在“The Animal”上，并将其表示的一部分聚集到“it”的编码中</center>

# 自注意细节（Self-Attention in Detail）

下文分析如何使用向量计算自注意，与如何使用矩阵实现。

计算自注意力的第一步是从每个编码器的输入向量创建三个向量，因此，对于每个单词，创建一个查询向量（Query vector）、一个键向量（Key vector）、一个值向量（Value vector）。通过将嵌入（embedding）乘以在训练过程中训练的三个矩阵创建这些向量。这些新向量的维度小于嵌入向量——维度为64，而嵌入与编码器的输入、输出向量的维度为512——这是一种架构选择，可以使多头注意力的计算（大部分）保持恒定。

![](/assets/images/2023-06/Pasted%20image%2020230717124508.png)
<center>将x1乘以WQ权重矩阵得到q1，即与该单词关联的“查询”向量。最终为输入句子中的每个单词创建一个“查询”、“键”、“值”投影</center>

第二步是计算分数，假设计算单词“Thinking”的自注意力，需要根据输入句子的每个单词对这个单词进行评分。当在某个位置对单词进行编码时，分数决定了对输入句子的其他部分的关注程度。分数是通过计算查询向量（query vector）与需要评分的单词的键向量（key vector）的点积（dot product）得到。因此，如果计算处理位置#1中单词的自注意力，第一个分数将是q1和k1的点积，第二个分数是q1和k2的点积。

![](/assets/images/2023-06/Pasted%20image%2020230717125946.png)

第三步和第四步将这些分数除以8（键向量维度的平方根是64（默认值），使梯度更稳定，也可以是其他值），然后将结果传递给softmax运算（softmax operation）。Softmax对分数进行归一化，使它们全部为正数并且总和为1。

![](/assets/images/2023-06/Pasted%20image%2020230717130347.png)

softmax分数决定了每个单词在这个位置上的表述量。显然，单词所在位置具有最高的softmax分数，但有时关注与当前单词相关的另一个单词是有用的。

第五步，将每个值向量（value vector）乘以softmax分数，很直观地保持关注单词的完整性，并覆盖不相关的单词（比如将它们乘以0.001这样的小数字）。

第六步，对加权值向量（weighted value）求和，产生该位置自注意力层的输出。

![](/assets/images/2023-06/Pasted%20image%2020230717131014.png){: #calculate6}

自注意力层计算得到的向量可以发送到前馈神经网络。在实际中，可以通过矩阵形式计算，更快的处理。

# 自注意的矩阵计算（Matrix Calculation of Self-Attention）

第一步，计算查询、键、值矩阵。将嵌入（embedding）打包到矩阵X中，并乘以已训练过的权重矩阵（WQ、WK、WV）。

![](/assets/images/2023-06/Pasted%20image%2020230717192208.png)
<center>X矩阵中的每一行对应于输入句子中的一个单词</center>

最后，由于处理的是矩阵，可以将第二步到第六步压缩为一个公式来计算自注意层的输出。

![](/assets/images/2023-06/Pasted%20image%2020230717192425.png)
<center>矩阵形式的self-attention计算</center>

# The Beast With Many Heads

论文通过“多头”注意力机制进一步细化自注意层，通过下面两种方式提高注意层的性能：

1. 扩展模型关注不同位置的能力。[z1](#calculate6)包含一些其他编码，但它可能由实际单词本身主导。
2. 为注意层提供了多个“表示子空间”（representation subspaces）。通过多头注意，不只拥有一组查询/键/值权重矩阵，而是拥有多组查询/键/值权重矩阵（Transformer使用八个注意头，因此最终为每个编码器/解码器提供八组）。这些集合中的每一个都是随机初始化的。然后，在训练之后，每个集合用于将输入嵌入（或来自较低编码器/解码器的向量）投影到不同的表示子空间中。

![](/assets/images/2023-06/Pasted%20image%2020230717234818.png)
<center>通过多头注意，为每个头维护单独的Q/K/V权重矩阵，从而产生不同的Q/K/V矩阵</center>

如果进行上图相同的自注意计算，只需用不同的权重矩阵进行八次不同的计算，最终得到八个不同的Z矩阵。

![](/assets/images/2023-06/Pasted%20image%2020230717235537.png)

而前馈层不需要八个矩阵——只需要一个矩阵（每个单词一个向量），所以需要一种方法将这八个压缩乘一个矩阵。将矩阵连接起来，然后乘以一个附加权重矩阵WO。

![](/assets/images/2023-06/Pasted%20image%2020230717235732.png)

这几乎便是多头注意的全部内容。

![](/assets/images/2023-06/Pasted%20image%2020230717235854.png)

对示例句子的“it”进行编码时，不同注意头聚焦的位置。

![](/assets/images/2023-06/Pasted%20image%2020230718000315.png)
<center>对“it”进行编码时，一个注意头主要关注“animal”，而另一个注意头关注“tired”。从某种意义上，模型对“it”的表征会包含一些“animal”和“tired”的表征</center>

如果将所有注意头添加到图片中，将会更难解释：

![](/assets/images/2023-06/Pasted%20image%2020230718000707.png)

# 使用位置编码表示序列的顺序（Representing The Order of The Sequence Using Positional Encoding）

到目前为止，模型缺少一种解释输入序列中单词顺序的方法。为解决这个问题，transformer向每个输入嵌入添加一个向量。这些向量遵循模型学习的特定模式，有助于确认每个单词的位置，或序列中不同单词之间的距离。

在嵌入向量投影到Q/K/V向量和点积注意时，将这些值添加到嵌入向量中，就可以提供有意义的距离。

![](/assets/images/2023-06/Pasted%20image%2020230718232535.png)
<center>为了让模型理解单词的顺序，添加位置编码向量——值遵循特定的模式</center>

假设嵌入维数为4，则实际的位置编码如下：

![](/assets/images/2023-06/Pasted%20image%2020230718232958.png)

下图中，每一行对应一个向量的位置编码。因此，第一行将是输入序列中添加到第一个单词嵌入的向量。每一行包含512个值，每个值介入1和-1之间。

![](/assets/images/2023-06/Pasted%20image%2020230718233305.png)
<center>嵌入大小为512（列）的20个单词（行）的位置编码。左半部分的值由正弦函数生成，右半部分由余弦函数生成，连接起来行程每个位置编码向量</center>

# 残差（The Residuals）

每个编码器中的每个子层（自注意，ffnn）周围都有一个残差连接，并且后面是层归一化步骤。

![](/assets/images/2023-06/Pasted%20image%2020230718233744.png)

可视化自注意相关的向量和层范数操作（layer-norm operation）：

![](/assets/images/2023-06/Pasted%20image%2020230718233839.png)

同样适用于解码器的子层——一个由2个堆叠编码器和解码器组成的Transformer：

![](/assets/images/2023-06/Pasted%20image%2020230718234339.png)

# 解码器端（The Decoder Side）

编码器首先处理输入序列，然后顶部编码器的输出被转换为一组注意向量K和V。每个解码器在其“编码器-解码器注意（encoder-decoder attention）”使用这些向量，有助于解码器关注输入序列中的适当位置。

![](/assets/images/2023-06/transformer_decoding_1.gif)

<center>完成编码阶段后，开始解码阶段。解码阶段的每个步骤都会输出输出序列中的一个元素</center>

接下来重复这一过程，直到出现一个特殊符号，表明transfomer解码器已完成输出。每个步骤的输出都会在下一时间点的步骤馈送到底部解码器，解码器会像编码器一样向上传递结果。

像对编码器输入所做的那样，在解码器输入中嵌入并添加位置编码，以指示每个单词的位置。

![](/assets/images/2023-06/transformer_decoding_2.gif)

解码器中的自关注层运行方式与编码器中的运行方式略有不同：

在解码器中，自注意层只允许关注输出序列中较早的位置——通过在自注意计算的softmax步骤前屏蔽未来位置（设置为`-inf`）。

“Encoder-Decoder Attention”层的工作方式与multiheaded self-attention类似，只不过它是从其下面层创建查询矩阵，并从编码器堆栈的输出中获取键和值矩阵。

# 最后的线性和Softmax层（The Final Linear and Softmax Layer）

解码器堆栈输出浮点数向量，最后的Linear层将其变成单词，跟随在后面的是softmax层。

线性层是一个简单的全连接神经网络，它将解码器堆栈产生的向量投影到一个更大的向量中，称为logits向量。假设模型知道从训练数据集中学习10000个不同的英文单词（模型的“输出词汇”），将使logits向量有10000个单元格宽度——每个单元格对应一个唯一的分数。然后，softmax层将这些分数转换为概率（全部为正数，加总为1.0）。选择概率最高的单元格，与之相联单词作为该时间步骤的输出结果。

![](/assets/images/2023-06/Pasted%20image%2020230719130302.png)
<center>从底部开始，生成作为解码器堆栈输出的向量，然后转换为输出单词</center>

# 回顾训练（Recap Of Training）

在训练模型期间，未经训练的模型将经历完全相同的前向传递，但由于是在标记的训练数据集上进行训练，可以将其输出与实际的证券输出进行比较。

形象化这一点，假设输出词汇仅包含六个单词（"a", "am", "i", "thanks", "student", "<\eos>"('end of sentence')）：

![](/assets/images/2023-06/Pasted%20image%2020230719130950.png)
<center>模型的输出词汇是在开始训练之前的预处理阶段创建的</center>

一旦定义了输出词汇表，可以使用相同宽度的向量来表示词汇表中的每个单词，称为one-shot编码：

![](/assets/images/2023-06/Pasted%20image%2020230719131207.png)
<center>输出词汇的one-shot编码</center>

# 损失函数（The Loss Function）

![](/assets/images/2023-06/Pasted%20image%2020230719131335.png)
<center>由于模型的参数（权重）都是随机初始化的，（未经训练的）模型会生成每个单元格/单词的任意值概率分布。可以将其与实际输出进行比较，然后使用反向传播调整所有模型的权重，使输出更接近所需的输出</center>

如何比较两个概率分布？只需将其中一个减去另一个即可，可参考[cross-entropy](https://colah.github.io/posts/2015-09-Visual-Information/) 和 [Kullback–Leibler divergence](https://www.countbayesie.com/blog/2017/5/9/kullback-leibler-divergence-explained)。

更现实的是，使用比一个单词长的句子。例如，输入：“je suis étudiant”，预期输出：“i am a student”。意味着希望模型能够连续输出概率分布：
- 每个概率宽度由宽度为vocab_size的向量表示
- 第一个概率分布在与单词“i”相关的单元格中具有最高的概率
- 第二个概率分布在与单词“am”相关的单元格中具有更高的概率
- 以此类推，直到第五个输出输出分布“end of sentences”符号，在10000个元素的词汇表中也有一个与之关联的单元格

![](/assets/images/2023-06/Pasted%20image%2020230719132305.png)

在足够大的数据集上训练模型足够长的时间后，期望生成的概率分布：

![](/assets/images/2023-06/Pasted%20image%2020230719233634.png)

<center>经过训练后，期望模型能够输出期望的翻译。每个位置都有概率，即使不太可能是该时间步的输出——非常有用的softmax的一个属性，有助于训练过程</center>

此时，模型一次产生一个输出，可以假设模型从该概率分布中选择概率最高的单词，并丢弃其余的单词——贪婪解码方法（greedy decoding）。

另一种方法是保留最上面的两个单词（‘I’，‘a’），在下一步中运行模型两次：一次假设第一个输出位置是‘I’，另一次假设是‘a’，并且考虑位置#1、#2，保留产生较少错误的版本。重复#2、#3等位置——束搜索（beam search）。这些都是可以实验的超参数（hyperparameters）。

# 延伸阅读

I hope you’ve found this a useful place to start to break the ice with the major concepts of the Transformer. If you want to go deeper, I’d suggest these next steps:

- Read the [Attention Is All You Need](https://arxiv.org/abs/1706.03762) paper, the Transformer blog post ([Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)), and the [Tensor2Tensor announcement](https://ai.googleblog.com/2017/06/accelerating-deep-learning-research.html).
- Watch [Łukasz Kaiser’s talk](https://www.youtube.com/watch?v=rBCqOTEfxvg) walking through the model and its details
- Play with the [Jupyter Notebook provided as part of the Tensor2Tensor repo](https://colab.research.google.com/github/tensorflow/tensor2tensor/blob/master/tensor2tensor/notebooks/hello_t2t.ipynb)
- Explore the [Tensor2Tensor repo](https://github.com/tensorflow/tensor2tensor).

Follow-up works:

- [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)
- [One Model To Learn Them All](https://arxiv.org/abs/1706.05137)
- [Discrete Autoencoders for Sequence Models](https://arxiv.org/abs/1801.09797)
- [Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/abs/1801.10198)
- [Image Transformer](https://arxiv.org/abs/1802.05751)
- [Training Tips for the Transformer Model](https://arxiv.org/abs/1804.00247)
- [Self-Attention with Relative Position Representations](https://arxiv.org/abs/1803.02155)
- [Fast Decoding in Sequence Models using Discrete Latent Variables](https://arxiv.org/abs/1803.03382)
- [Adafactor: Adaptive Learning Rates with Sublinear Memory Cost](https://arxiv.org/abs/1804.04235)

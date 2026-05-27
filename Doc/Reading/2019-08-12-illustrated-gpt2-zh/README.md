# 图解 GPT-2（可视化 Transformer 语言模型）

原文：[The Illustrated GPT-2 (Visualizing Transformer Language Models)](https://jalammar.github.io/illustrated-gpt2/)

---

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/openAI-GPT-2-3.png"/>
  <br />
</div>


今年（2019），我们见证了机器学习的一项令人惊叹的应用。[OpenAI GPT-2](https://openai.com/blog/better-language-models/) 展示了令人印象深刻的能力——它能够撰写连贯且富有感情的文章，超出了我们对当前语言模型所能达到水平的预期。GPT-2 并不是一个特别新颖的架构——它的架构与 decoder-only transformer 非常相似。然而，GPT-2 是一个非常大的、基于 transformer 的语言模型，在海量数据集上进行了训练。在这篇文章中，我们将研究使该模型能够产生这些结果的架构。我们将深入探讨其 self-attention 层的内部机制。然后我们还将了解 decoder-only transformer 在语言建模之外的应用。

我的目标也是为了补充我之前的文章 [图解 Transformer](/illustrated-transformer/)，用更多可视化内容来解释 transformer 的内部工作原理，以及它们自原始论文以来的演变。我希望这种可视化语言能够使后续基于 Transformer 的模型更容易解释，因为它们的内部机制还在持续演变。

<!--more-->

<div style="font-size:75%; background-color:#eee; border: 1px solid #bbb; display: table; padding: 7px" markdown="1">

<div style="text-align:center" markdown="1">  

**目录**

</div>

* **第一部分：GPT2 与语言建模**
  * 什么是语言模型
  * 用于语言建模的 Transformer
  * 与 BERT 的一个区别
  * Transformer Block 的演变
  * 脑外科速成课：深入 GPT-2 内部
  * 更深入的探索
  * 第一部分结束：GPT-2，隆重登场
* **第二部分：图解 Self-Attention**
  * Self-Attention（无 masking）
  * 1- 创建 Query、Key 和 Value 向量
  * 2- 打分
  * 3- 求和
  * 图解 Masked Self-Attention
  * GPT-2 的 Masked Self-Attention
  * 超越语言建模
  * 恭喜完成！
* **第三部分：超越语言建模**
  * 机器翻译
  * 摘要生成
  * 迁移学习
  * 音乐生成

</div>

## 第一部分：GPT2 与语言建模 

那么语言模型到底是什么？

### 什么是语言模型
在 [图解 Word2vec](/illustrated-word2vec/) 中，我们了解了什么是语言模型——本质上就是一个能够查看句子的一部分并预测下一个词的机器学习模型。最著名的语言模型就是智能手机键盘，它会根据你当前输入的内容来建议下一个词。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/swiftkey-keyboard.png"/>
  <br />
</div>




从这个意义上说，我们可以说 GPT-2 基本上就是手机键盘应用的下一个词预测功能，只不过它比手机上的大得多、也复杂得多。GPT-2 是在一个名为 WebText 的 40GB 海量数据集上训练的，这是 OpenAI 研究人员在研究过程中从互联网上抓取的。从存储大小来比较，我使用的键盘应用 SwiftKey 占用 78MB 空间。经过训练的最小版本 GPT-2 需要 500MB 存储空间来保存其所有参数。最大版本的 GPT-2 是其 13 倍大小，因此可能占用超过 6.5GB 的存储空间。


<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-sizes.png"/>
  <br />
</div>


体验 GPT-2 的一个好方法是使用 [AllenAI GPT-2 Explorer](https://gpt2.apps.allenai.org/?text=Joel%20is)。它使用 GPT-2 来显示下一个词的十个可能预测（以及它们的概率分数）。你可以选择一个词，然后查看下一组预测列表，从而继续撰写段落。


### 用于语言建模的 Transformer

正如我们在 [图解 Transformer](/illustrated-transformer/) 中所看到的，原始 transformer 模型由 encoder 和 decoder 组成——每个都是一堆我们可以称之为 transformer block 的组件。这种架构很适合机器翻译问题，因为 encoder-decoder 架构在过去的翻译任务中已经被证明是成功的。


<div class="img-div" markdown="0">
  <image src="../images/xlnet/transformer-encoder-decoder.png"/>
  <br />
</div>

后续许多研究工作去掉了 encoder 或 decoder 中的一个，只使用一组 transformer block——尽可能多地堆叠，输入海量训练文本，并投入大量算力（训练某些语言模型花费数十万美元，在 [AlphaStar](https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/) 的案例中可能花费数百万美元）。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt-2-transformer-xl-bert-3.png"/>
  <br />
</div>

我们可以把这些 block 堆多高？事实证明，这是不同 GPT2 模型大小之间的主要区分因素之一：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-sizes-hyperparameters-3.png"/>
  <br />
</div>


### 与 BERT 的一个区别
<blockquote class='subtle'>
<strong>机器人第一定律</strong><br />
机器人不得伤害人类，或因不作为而使人类受到伤害。
</blockquote>


GPT-2 使用 transformer decoder block 构建。而 BERT 使用 transformer encoder block。我们将在后续章节中研究两者的区别。但它们之间的一个关键区别是：GPT2 和传统语言模型一样，每次输出一个 token。让我们以提示一个训练好的 GPT-2 背诵机器人第一定律为例：

<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/gpt-2-output.gif"/>
  <br />
</div>

这些模型实际工作的方式是：每产生一个 token 后，该 token 就被添加到输入序列中。这个新序列成为模型下一步的输入。这个思想叫做"自回归"（auto-regression）。这也是 [让 RNN 变得异常有效](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) 的思想之一。


<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/gpt-2-autoregression-2.gif"/>
  <br />
</div>

GPT2 以及后来的一些模型如 TransformerXL 和 XLNet 本质上都是自回归的。BERT 不是。这是一种权衡。失去自回归能力后，BERT 获得了将词两侧的上下文都纳入考量的能力，从而获得更好的结果。XLNet 恢复了自回归，同时找到了一种替代方式来整合两侧上下文。

### Transformer Block 的演变

[最初的 transformer 论文](https://arxiv.org/abs/1706.03762) 引入了两种类型的 transformer block：

#### Encoder Block

首先是 encoder block：

<div class="img-div" markdown="0">
  <image src="../images/xlnet/transformer-encoder-block-2.png"/>
  <br />
  原始 transformer 论文中的 encoder block 可以接受最长特定序列长度的输入（例如 512 个 token）。如果输入序列比这个限制短也没关系，我们可以对序列的其余部分进行填充（padding）。
</div>


#### Decoder Block
其次是 decoder block，它与 encoder block 有一个小的架构差异——增加了一个层，允许它关注 encoder 的特定输出片段：

<div class="img-div" markdown="0">
  <image src="../images/xlnet/transformer-decoder-block-2.png"/>
  <br />
</div>

这里 self-attention 层的一个关键区别是，它会遮蔽未来的 token——不是像 BERT 那样把词变成 [mask]，而是通过在 self-attention 计算中进行干预，阻止当前计算位置右侧 token 的信息流入。

例如，如果我们高亮位置 #4 的路径，可以看到它只被允许关注当前和之前的 token：

<div class="img-div" markdown="0">
  <image src="../images/xlnet/transformer-decoder-block-self-attention-2.png"/>
  <br />
</div>

重要的是要清楚 self-attention（BERT 使用的）和 masked self-attention（GPT-2 使用的）之间的区别。普通的 self-attention block 允许一个位置去查看其右侧的 token。Masked self-attention 则阻止这种情况发生：

<div class="img-div-any-size" markdown="0">
  <image src="../images/gpt2/self-attention-and-masked-self-attention.png"/>
  <br />
</div>

#### Decoder-Only Block
在原始论文之后，[Generating Wikipedia by Summarizing Long Sequences](https://arxiv.org/pdf/1801.10198.pdf) 提出了另一种 transformer block 的排列方式，能够进行语言建模。这个模型去掉了 Transformer encoder。因此，我们称该模型为 "Transformer-Decoder"。这个早期的基于 transformer 的语言模型由六个 transformer decoder block 堆叠而成：

<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/transformer-decoder-intro.png"/>
  <br />
  这些 decoder block 是相同的。我展开了第一个，可以看到它的 self-attention 层是 masked 变体。注意该模型现在可以在某个片段中处理多达 4,000 个 token——相比原始 transformer 的 512 是一个巨大的升级。
</div>

这些 block 与原始 decoder block 非常相似，只是去掉了第二个 self-attention 层。[Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/pdf/1808.04444.pdf) 中研究了一个类似的架构，用来创建一个逐字母/字符预测的语言模型。


OpenAI GPT-2 模型使用的就是这些 decoder-only block。

### 脑外科速成课：深入 GPT-2 内部

<blockquote class='subtle' markdown="block">
Look inside and you will see,
The words are cutting deep inside my brain.
Thunder burning, quickly burning,
Knife of words is driving me insane, insane yeah.
~<strong>[Budgie](https://en.wikipedia.org/wiki/Budgie_(band))</strong>

</blockquote>

让我们把一个训练好的 GPT-2 放在手术台上，看看它是如何工作的。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt-2-layers-2.png"/>
  <br />
  GPT-2 可以处理 1024 个 token。每个 token 沿着自己的路径流经所有 decoder block。
</div>

运行训练好的 GPT-2 最简单的方式是让它自己漫谈（技术上叫做*生成无条件样本*）——或者，我们可以给它一个提示让它谈论某个特定话题（即生成*交互式条件样本*）。在漫谈的情况下，我们只需给它起始 token，让它开始生成词（训练好的模型使用 ```<|endoftext|>``` 作为起始 token。我们改称它为 ```<s>```）。

<div class="img-div-any-width" markdown="1">
  <image src="../images/gpt2/gpt2-simple-output-2.gif"/>
  <br />
</div>

模型只有一个输入 token，所以只有那条路径是活跃的。token 依次通过所有层处理，然后沿该路径产生一个向量。该向量可以与模型的词表（模型认识的所有词，GPT-2 的情况下是 50,000 个词）进行打分。在这种情况下我们选择了概率最高的 token "the"。但我们当然可以混合一下——你知道如果你不断点击键盘应用建议的词，有时会陷入重复循环，唯一的出路就是点击第二个或第三个建议词。这里也会发生同样的情况。GPT-2 有一个叫 top-k 的参数，我们可以用它让模型考虑采样除了最高分词之外的其他词（当 top-k = 1 时只选最高分词）。

在下一步中，我们把第一步的输出添加到输入序列中，让模型进行下一次预测：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt-2-simple-output-3.gif"/>
  <br />
</div>

注意在这次计算中只有第二条路径是活跃的。GPT-2 的每一层都保留了对第一个 token 的自身解读，并会在处理第二个 token 时使用它（我们将在后续关于 self-attention 的章节中详细介绍）。GPT-2 不会根据第二个 token 重新解读第一个 token。

### 更深入的探索

#### 输入编码
让我们看看更多细节以更深入地了解模型。从输入开始。与我们之前讨论过的其他 NLP 模型一样，模型在其 embedding 矩阵中查找输入词的 embedding——这是训练好的模型的组成部分之一。

<div class="img-div" markdown="0">
  <image src="../images/gpt2/gpt2-token-embeddings-wte-2.png"/>
  <br />
  每一行是一个词嵌入：一个表示某个词并捕捉其部分含义的数字列表。这个列表的大小在不同的 GPT2 模型尺寸中是不同的。最小的模型每个词/token 使用 768 维的 embedding。
</div>

所以一开始，我们在 embedding 矩阵中查找起始 token ```<s>``` 的 embedding。在将其交给模型中的第一个 block 之前，我们需要加入位置编码——一个用于向 transformer block 指示词在序列中顺序的信号。训练好的模型中包含一个矩阵，其中包含输入中 1024 个位置中每个位置的位置编码向量。

<div class="img-div" markdown="0">
  <image src="../images/gpt2/gpt2-positional-encoding.png"/>
  <br />
</div>

至此，我们已经介绍了输入词在被送入第一个 transformer block 之前的处理方式。我们也了解了构成训练好的 GPT-2 的两个权重矩阵。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-input-embedding-positional-encoding-3.png"/>
  <br />
  将一个词送入第一个 transformer block 意味着查找其 embedding 并加上位置 #1 的位置编码向量。
</div>

#### 沿堆栈向上的旅程

第一个 block 现在可以处理这个 token 了：先通过 self-attention 过程，然后通过其神经网络层。一旦第一个 transformer block 处理完 token，它会将结果向量向上发送到堆栈中由下一个 block 处理。每个 block 中的处理过程是相同的，但每个 block 在 self-attention 和神经网络子层中都有各自的权重。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-transformer-block-vectors-2.png"/>
  <br />
</div>



#### Self-Attention 回顾

语言高度依赖上下文。例如，看看第二定律：

<blockquote class='subtle'>

<strong>机器人第二定律</strong><br />
A robot must obey the orders given <strong style="color:#D81B60">it</strong> by human beings except where <strong style="color:#689F38">such orders</strong> would conflict with the <strong style="color:#6D4C41">First Law</strong>.

</blockquote>

我在句中高亮了三处词语引用其他词语的地方。如果不结合它们所引用的上下文，就无法理解或处理这些词。当模型处理这个句子时，它必须能够知道：
* <strong style="color:#D81B60">it</strong> 指的是 robot
* <strong style="color:#689F38">such orders</strong> 指的是该定律前面的部分，即 "the orders given it by human beings"
* <strong style="color:#6D4C41">The First Law</strong> 指的是完整的第一定律

这就是 self-attention 所做的事情。它在处理某个词（将其通过神经网络）之前，将模型对相关和关联词的理解融入该词的表示中。它通过为片段中每个词的相关性打分，然后将它们的向量表示加权求和来实现这一点。

举个例子，顶层 block 中的 self-attention 层在处理 "it" 这个词时关注了 "a robot"。它将传递给其神经网络的向量是三个词的向量各自乘以其分数后的总和。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-example-2.png"/>
  <br />
</div>

#### Self-Attention 过程
Self-attention 沿着片段中每个 token 的路径进行处理。重要的组成部分是三个向量：

* <span class="decoder">Query</span>：Query 是当前词的表示，用于与所有其他词（使用它们的 key）进行打分。我们只关心当前正在处理的 token 的 query。
* <span class="context">Key</span>：Key 向量就像片段中所有词的标签。它们是我们在搜索相关词时进行匹配的对象。
* <span class="step_no">Value</span>：Value 向量是实际的词表示，一旦我们对每个词的相关性打了分，这些就是我们加总起来表示当前词的值。


<div class="img-div-any-size" markdown="0">
  <image src="../images/gpt2/self-attention-example-folders-3.png"/>
  <br />
</div>

一个粗略的类比是把它想象成在文件柜中搜索。Query 就像是一张写着你研究主题的便利贴。Key 就像是文件柜里文件夹上的标签。当你将便利贴与标签匹配时，我们取出该文件夹的内容，这些内容就是 value 向量。只不过你不是只在找一个值，而是从多个文件夹中混合多个值。

将 query 向量与每个 key 向量相乘会为每个文件夹产生一个分数（技术上：先做点积再做 softmax）。


<div class="img-div-any-size" markdown="0">
  <image src="../images/gpt2/self-attention-example-folders-scores-3.png"/>
  <br />
</div>


我们将每个 value 乘以其分数然后求和——得到我们的 self-attention 输出结果。

<div class="img-div-any-size" markdown="0">
  <image src="../images/gpt2/gpt2-value-vector-sum.png"/>
  <br />
</div>

这种 value 向量的加权混合产生了一个向量，它将 50% 的"注意力"放在了词 ```robot``` 上，30% 放在了词 ```a``` 上，19% 放在了词 ```it``` 上。在文章后面，我们会更深入地研究 self-attention。但首先，让我们继续沿堆栈向上走向模型的输出。

#### 模型输出

当模型顶层 block 产生其输出向量（其自身 self-attention 加上自身神经网络的结果）时，模型将该向量与 embedding 矩阵相乘。

<div class="img-div-any-size" markdown="0">
  <image src="../images/gpt2/gpt2-output-projection-2.png"/>
  <br />
</div>

回忆一下，embedding 矩阵中的每一行对应模型词表中一个词的 embedding。这次乘法的结果被解释为模型词表中每个词的分数。

<div class="img-div-any-size" markdown="0">
  <image src="../images/gpt2/gpt2-output-scores-2.png"/>
  <br />
</div>

我们可以简单地选择分数最高的 token（top_k = 1）。但如果模型也考虑其他词，会获得更好的结果。所以更好的策略是使用分数作为选择概率，从整个列表中采样一个词（这样分数更高的词被选中的机会更大）。一个折中方案是将 top_k 设为 40，让模型考虑分数最高的 40 个词。


<div class="img-div-any-size" markdown="0">
  <image src="../images/gpt2/gpt2-output.png"/>
  <br />
</div>

这样，模型完成了一次迭代，输出了一个词。模型持续迭代直到生成完整上下文（1024 个 token）或产生一个序列结束 token。

### 第一部分结束：GPT-2，隆重登场

以上就是 GPT2 工作原理的全面解析。如果你想确切知道 self-attention 层内部发生了什么，那么下面的附加章节就是为你准备的。我创建它是为了引入更多可视化语言来描述 self-attention，以便后续更容易研究和描述其他 transformer 模型（说的就是你们，TransformerXL 和 XLNet）。

我想指出这篇文章中的一些简化：
* 我将 "words"（词）和 "tokens"（token）互换使用。但实际上，GPT2 使用 Byte Pair Encoding 来创建其词表中的 token。这意味着 token 通常是词的一部分。
* 我们展示的例子是在 GPT2 的推理/评估模式下运行的。这就是为什么它一次只处理一个词。在训练时，模型会针对更长的文本序列进行训练，并一次处理多个 token。同样在训练时，模型处理更大的 batch size（512），而评估时使用的 batch size 为 1。
* 我在旋转/转置向量方面做了一些自由处理，以更好地管理图像中的空间。在实现时需要更加精确。
* Transformer 使用了大量的 layer normalization，这非常重要。我们在 图解 Transformer 中已经注意到了其中一些，但在这篇文章中更侧重于 self-attention。
* 有时我需要展示更多方框来表示一个向量。我将其标注为"放大"。例如：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/zoom-in.png"/>
  <br />
</div>


## 第二部分：图解 Self-Attention 

在文章前面我们展示了这张图来说明在处理词 ```it``` 的层中应用 self-attention 的情况：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-1-2.png"/>
  <br />
</div>

在这一节中，我们将详细看看它是如何完成的。注意我们将尝试以理解单个词发生了什么的方式来看待它。这就是为什么我们会展示很多单独的向量。实际的实现是通过将巨大的矩阵相乘来完成的。但我想在这里聚焦于词层面上的直觉。

### Self-Attention（无 masking）
让我们从原始 self-attention 开始看起，即在 encoder block 中计算的方式。看一个一次只能处理四个 token 的玩具 transformer block。


Self-attention 通过三个主要步骤完成：

1. 为每条路径创建 Query、Key 和 Value 向量。
2. 对每个输入 token，使用其 query 向量与所有其他 key 向量打分。
3. 将 value 向量乘以对应的分数后求和。

<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/self-attention-summary.png"/>
  <br />
</div>

### 1- 创建 Query、Key 和 Value 向量
让我们聚焦于第一条路径。我们将取其 query，并与所有 key 进行比较。这为每个 key 产生一个分数。self-attention 的第一步是为每条 token 路径计算三个向量（现在先忽略 attention head）：

<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/self-attention-1.png"/>
  <br />
</div>

### 2- 打分
现在我们有了这些向量，在第 2 步中只使用 query 和 key 向量。由于我们聚焦于第一个 token，我们将其 query 与所有其他 key 向量相乘，得到四个 token 各自的分数。
<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/self-attention-2.png"/>
  <br />
</div>


### 3- 求和

现在我们可以用分数乘以 value 向量了。分数高的 value 在我们求和后的结果向量中会占更大比重。

<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/self-attention-3-2.png"/>
  <br />
  分数越低，我们就把 value 向量显示得越透明。这是为了表明乘以小数会稀释向量的值。
</div>

如果我们对每条路径做同样的操作，最终会得到一个包含该 token 适当上下文的向量来表示每个 token。然后这些向量被呈递给 transformer block 中的下一个子层（前馈神经网络）：


<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/self-attention-summary.png"/>
  <br />
</div>

### 图解 Masked Self-Attention

现在我们已经了解了 transformer 内部的 self-attention 步骤，让我们继续看 masked self-attention。Masked self-attention 与 self-attention 完全相同，只是在第 2 步上有所不同。假设模型只有两个 token 作为输入，我们正在观察第二个 token。在这种情况下，最后两个 token 被遮蔽了。所以模型在打分步骤中进行了干预。它基本上总是将未来 token 的分数设为 0，这样模型就无法窥视未来的词：

<div class="img-div-any-width" markdown="0">
  <image src="../images/xlnet/masked-self-attention-2.png"/>
  <br />
</div>

这种遮蔽通常通过一个称为 attention mask 的矩阵来实现。想象一个四个词的序列（"robot must obey orders"）。在语言建模场景中，这个序列分四步吸收——每个词一步（暂时假设每个词是一个 token）。由于这些模型按 batch 工作，我们可以假设这个玩具模型的 batch size 为 4，它将整个序列（及其四个步骤）作为一个 batch 来处理。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/transformer-decoder-attention-mask-dataset.png"/>
  <br />
</div>

在矩阵形式中，我们通过将 queries 矩阵乘以 keys 矩阵来计算分数。让我们如下可视化它，只是单元格中放的不是词本身，而是与该单元格中的词关联的 query（或 key）向量：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/queries-keys-attention-mask.png"/>
  <br />
</div>

乘法之后，我们叠加上 attention mask 三角形。它将我们想要遮蔽的单元格设为 -infinity 或一个非常大的负数（例如 GPT2 中的 -10 亿）：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/transformer-attention-mask.png"/>
  <br />
</div>

然后，对每一行应用 softmax 产生我们用于 self-attention 的实际分数：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/transformer-attention-masked-scores-softmax.png"/>
  <br />
</div>

这个分数表的含义如下：
* 当模型处理数据集中的第一个样本（第 1 行），其中只包含一个词（"robot"），100% 的注意力会放在那个词上。
* 当模型处理数据集中的第二个样本（第 2 行），其中包含词（"robot must"），当它处理词 "must" 时，48% 的注意力放在 "robot" 上，52% 的注意力放在 "must" 上。
* 以此类推


### GPT-2 的 Masked Self-Attention
让我们更详细地了解 GPT-2 的 masked attention。

#### 评估时：一次处理一个 Token
我们可以让 GPT-2 完全按照 masked self-attention 的方式运行。但在评估时，当我们的模型在每次迭代后只添加一个新词时，为已经处理过的 token 重新计算沿早期路径的 self-attention 是低效的。

在这种情况下，我们处理第一个 token（暂时忽略 `<s>`）。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-qkv-1-2.png"/>
  <br />
</div>

GPT-2 保存了 ```a``` token 的 key 和 value 向量。每个 self-attention 层都保存该 token 对应的 key 和 value 向量：


<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-qkv-2-2.png"/>
  <br />
</div>

现在在下一次迭代中，当模型处理词 ```robot``` 时，它不需要为 ```a``` token 生成 query、key 和 value。它直接复用第一次迭代中保存的那些：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-qkv-3-2.png"/>
  <br />
</div>

#### GPT-2 Self-attention：1- 创建 queries、keys 和 values

假设模型正在处理词 ```it```。如果我们讨论的是底层 block，那么该 token 的输入将是 `it` 的 embedding 加上位置 #9 的位置编码：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-1.png"/>
  <br />
</div>


transformer 中每个 block 都有自己的权重（后文会详细分解）。我们首先遇到的是用于创建 queries、keys 和 values 的权重矩阵。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-2.png"/>
  <br />
  Self-attention 将其输入乘以权重矩阵（并加上一个偏置向量，此处未图示）。
</div>

这次乘法的结果是一个向量，基本上是词 `it` 的 query、key 和 value 向量的拼接。


<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-3.png"/>
  <br />
  将输入向量乘以 attention 权重向量（并在之后加上偏置向量）得到该 token 的 key、value 和 query 向量。
</div>

#### GPT-2 Self-attention：1.5- 拆分为 attention head

在前面的例子中，我们直接深入了 self-attention 而忽略了 "multi-head" 部分。现在有必要对这个概念做一些说明。Self-attention 在 Q、K、V 向量的不同部分上进行多次。"拆分" attention head 只是将长向量 reshape 为矩阵。小型 GPT2 有 12 个 attention head，所以这将是 reshape 后矩阵的第一个维度：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-split-attention-heads-1.png"/>
  <br />
</div>

在前面的例子中，我们看了一个 attention head 内部发生的事情。多 attention head 的一种理解方式是这样的（如果我们只可视化 12 个 attention head 中的 3 个）：


<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-split-attention-heads-2.png"/>
  <br />
</div>

#### GPT-2 Self-attention：2- 打分
现在我们可以继续打分了——注意我们只看一个 attention head（其他所有 head 都在进行类似的操作）：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-scoring.png"/>
  <br />
</div>

现在这个 token 可以与其他所有 token 的 key（在之前迭代中 attention head #1 计算得到的）进行打分：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-scoring-2.png"/>
  <br />
</div>


#### GPT-2 Self-attention：3- 求和
如我们之前所见，现在我们将每个 value 乘以其分数，然后求和，得到 attention head #1 的 self-attention 结果：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-multihead-sum-1.png"/>
  <br />
</div>


#### GPT-2 Self-attention：3.5- 合并 attention head

我们处理各个 attention head 的方式是先将它们拼接成一个向量：
<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-merge-heads-1.png"/>
  <br />
</div>

但这个向量还不能直接发送到下一个子层。我们需要先将这个科学怪人式的隐藏状态拼接体转变为同质的表示。

#### GPT-2 Self-attention：4- 投影

我们让模型学习如何将拼接后的 self-attention 结果最好地映射为一个前馈神经网络能处理的向量。这里用到了我们的第二个大型权重矩阵，它将 attention head 的结果投影到 self-attention 子层的输出向量中：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-project-1.png"/>
  <br />
</div>

这样，我们就产生了可以发送到下一层的向量：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-self-attention-project-2.png"/>
  <br />
</div>

#### GPT-2 全连接神经网络：第 1 层

全连接神经网络是 block 在 self-attention 将适当上下文纳入表示后处理其输入 token 的地方。它由两层组成。第一层是模型大小的四倍（由于 GPT2 small 是 768，这个网络会有 768*4 = 3072 个单元）。为什么是四倍？这只是原始 transformer 选定的大小（模型维度是 512，该模型中的第一层是 2048）。这似乎为 transformer 模型提供了足够的表示容量来处理迄今为止遇到的任务。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-mlp1.gif"/>
  <br />
  （未显示：偏置向量）
</div>



#### GPT-2 全连接神经网络：第 2 层 - 投影回模型维度

第二层将第一层的结果投影回模型维度（小型 GPT2 为 768）。这次乘法的结果就是该 token 的 transformer block 输出。

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-mlp-2.gif"/>
  <br />
  （未显示：偏置向量）
</div>

### 恭喜完成 <span style="color:#7F34AA">It</span>！
这是我们将要深入的最详细版本的 transformer block！你现在已经基本掌握了 transformer 语言模型内部发生的绝大部分情况。回顾一下，我们勇敢的输入向量会遇到这些权重矩阵：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-transformer-block-weights-2.png"/>
  <br />
</div>

每个 block 都有自己的一组这些权重。另一方面，模型只有一个 token embedding 矩阵和一个位置编码矩阵：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-weights-2.png"/>
  <br />
</div>

如果你想查看模型的所有参数，我在这里做了统计：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/gpt2-117-parameters.png"/>
  <br />
</div>

它们加起来是 124M 个参数而不是 117M，不知道为什么。但这就是已发布代码中看起来的数量（如果我有误请纠正我）。

## 第三部分：超越语言建模 
Decoder-only transformer 在语言建模之外持续展现潜力。它在许多应用中取得了成功，这些应用可以用与上面类似的可视化来描述。让我们通过查看其中一些应用来结束这篇文章。

### 机器翻译
进行翻译并不需要 encoder。同样的任务可以由 decoder-only transformer 来完成：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/decoder-only-transformer-translation.png"/>
  <br />
</div>


### 摘要生成

这是第一个 decoder-only transformer 被训练的任务。具体来说，它被训练来阅读维基百科文章（不包含目录之前的开头部分），并对其进行摘要。文章实际的开头部分被用作训练数据集中的标签：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/wikipedia-summarization.png"/>
  <br />
</div>

该论文针对维基百科文章训练了模型，因此训练好的模型能够生成文章摘要：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/decoder-only-summarization.png"/>
  <br />
</div>

### 迁移学习
在 [Sample Efficient Text Summarization Using a Single Pre-Trained Transformer](https://arxiv.org/abs/1905.08836) 中，一个 decoder-only transformer 先在语言建模上进行预训练，然后微调来做摘要生成。结果表明，在有限数据设置下，它比预训练的 encoder-decoder transformer 取得了更好的结果。

GPT2 的论文也展示了在语言建模预训练后进行摘要生成的结果。

### 音乐生成
[Music Transformer](https://magenta.tensorflow.org/music-transformer) 使用 decoder-only transformer 来生成具有表现力的时序和动态变化的音乐。"音乐建模"就像语言建模一样——只需让模型以无监督方式学习音乐，然后让它采样输出（我们之前称之为"漫谈"）。

你可能会好奇在这种场景中音乐是如何表示的。请记住，语言建模可以通过字符、词或词片段 token 的向量表示来完成。对于音乐演奏（我们先考虑钢琴），我们必须表示音符，还要表示力度（velocity）——衡量钢琴键被按下力度的指标。


<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/music-transformer-performance-encoding-3.png"/>
  <br />
</div>

一段演奏就是一系列这样的 one-hot 向量。一个 MIDI 文件可以转换为这种格式。论文中有如下示例输入序列：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/music-representation-example.png"/>
  <br />
</div>

这个输入序列的 one-hot 向量表示看起来像这样：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/music-transformer-input-representation-2.png"/>
  <br />
</div>

我喜欢论文中展示 Music Transformer 中 self-attention 的一个可视化。我在这里添加了一些注释：

<div class="img-div-any-width" markdown="0">
  <image src="../images/gpt2/music-transformer-self-attention-2.png"/>
  <br />
  "图 8：这段曲子有一个重复的三角形轮廓。query 位于后面的一个峰值上，它关注了之前所有峰值上的高音符，一直到曲子的开头。"..."[该]图显示了一个 query（所有注意力线的来源）和被关注的先前记忆（接收到更多 softmax 概率的音符被高亮显示）。注意力线的颜色对应不同的 head，宽度对应 softmax 概率的权重。"
</div>

如果你对音符的这种表示方式不清楚，[请看这个视频](https://www.youtube.com/watch?v=ipzR9bhei_o)。

## 结论
至此结束了我们的 GPT2 之旅，以及对其父模型 decoder-only transformer 的探索。希望你读完这篇文章后对 self-attention 有了更好的理解，并且对 transformer 内部发生的事情更加从容。

## 资源
* OpenAI 的 [GPT2 实现](https://github.com/openai/gpt-2)
* 查看 [Hugging Face](https://huggingface.co/) 的 [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) 库，除了 GPT2 之外，它还实现了 BERT、Transformer-XL、XLNet 和其他前沿 transformer 模型。

## 致谢
感谢 [Lukasz Kaiser](https://twitter.com/lukaszkaiser)、[Mathias Müller](https://www.cl.uzh.ch/de/people/team/compling/mmueller.html)、[Peter J. Liu](https://twitter.com/peterjliu)、[Ryan Sepassi](https://twitter.com/rsepassi) 和 [Mohammad Saleh](https://www.linkedin.com/in/mohammad-saleh-39614224/) 对本文早期版本的反馈。

欢迎评论或纠正，请在 Twitter 上联系我 [@JayAlammar](https://twitter.com/JayAlammar)


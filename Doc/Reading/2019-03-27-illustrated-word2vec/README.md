# 图解 Word2vec

原文：[The Illustrated Word2vec](https://jalammar.github.io/illustrated-word2vec/)

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/word2vec.png"/>
  <br />
</div>


<blockquote class='subtle'>
  “<strong>There is in all things a pattern that is part of our universe. It has symmetry, elegance, and grace</strong> - those qualities you find always in that which the true artist captures. You can find it in the turning of the seasons, in the way sand trails along a ridge, in the branch clusters of the creosote
  bush or the pattern of its leaves. <br /><br />

  We try to copy these patterns in our lives and our society,
  seeking the rhythms, the dances, the forms that comfort.
  Yet, it is possible to see peril in the finding of
  ultimate perfection. It is clear that the ultimate
  pattern contains it own fixity. In such
  perfection, all things move toward death.”
  ~ Dune (1965)
</blockquote>


我认为嵌入（embedding）是机器学习中最迷人的思想之一。如果你用过 Siri、Google Assistant、Alexa、Google 翻译，甚至带有下一词预测功能的智能手机键盘，那么你很可能已经从这个思想中获益了——它已经成为自然语言处理模型的核心。在过去的几十年里，将嵌入用于神经网络模型方面有了相当大的发展（最近的进展包括上下文相关的词嵌入，催生了像 [BERT](https://jalammar.github.io/illustrated-bert/) 和 GPT2 这样的前沿模型）。

<iframe width="560" height="315" src="https://www.youtube.com/embed/ISPId9Lhc1g" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"  style="
width: 100%;
max-width: 560px;" allowfullscreen></iframe>

Word2vec 是一种高效创建词嵌入的方法，自 2013 年以来就一直存在。但除了作为词嵌入方法的用途之外，它的一些概念已经被证明在创建推荐引擎、以及理解序列数据（甚至在商业的、非语言的任务中）方面同样有效。像 [Airbnb](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb)、[Alibaba](https://www.kdd.org/kdd2018/accepted-papers/view/billion-scale-commodity-embedding-for-e-commerce-recommendation-in-alibaba)、[Spotify](https://www.slideshare.net/AndySloane/machine-learning-spotify-madison-big-data-meetup) 和 [Anghami](https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484) 这样的公司，都从 NLP 世界中剥离出这件精巧的工具，并将其用于生产环境，赋能了新一代推荐引擎。

在本文中，我们会讲解嵌入的概念，以及用 word2vec 生成嵌入的机制。但我们先从一个例子开始，熟悉一下用向量来表示事物。你知道吗，一组五个数字（一个向量）就能表达出关于你性格的很多东西？

<!--more-->

## 性格嵌入：你是个什么样的人？
<blockquote class='subtle'>
“I give you the desert chameleon, whose ability to blend itself into the background tells you all you need to know about the roots of ecology and the foundations of a personal identity” ~Children of Dune
</blockquote>

在 0 到 100 的范围内，你有多内向/外向（其中 0 表示最内向，100 表示最外向）？你做过像 MBTI 这样的性格测试吗——或者更好的，[大五人格特质（Big Five Personality Traits）](https://en.wikipedia.org/wiki/Big_Five_personality_traits)测试？如果没有，这些测试会问你一系列问题，然后在若干个维度上给你打分，内向/外向就是其中之一。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/big-five-personality-traits-score.png"/>
  <br />
  Example of the result of a Big Five Personality Trait test. It can really tell you a lot about yourself and is shown to have predictive ability in <a href="http://psychology.okstate.edu/faculty/jgrice/psyc4333/FiveFactor_GPAPaper.pdf">academic</a>, <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1744-6570.1999.tb00174.x">personal</a>, and <a href="https://www.massgeneral.org/psychiatry/assets/published_papers/soldz-1999.pdf">professional success</a>. <a href="https://projects.fivethirtyeight.com/personality-quiz/">This</a> is one place to find your results.
</div>


假设我的内向/外向得分是 38/100。我们可以这样把它画出来：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/introversion-extraversion-100.png"/>
</div>

我们把范围换成 -1 到 1：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/introversion-extraversion-1.png"/>
</div>

仅凭这一条信息，你觉得自己对一个人了解得有多深？并不多。人是复杂的。所以让我们再加一个维度——测试中另一项特质的得分。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/two-traits-vector.png"/>
  <br />
  我们可以把这两个维度表示为图上的一个点，或者更好地，表示为从原点指向该点的一个向量。我们有出色的工具来处理向量，它们很快就会派上用场。
</div>

我隐藏了我们正在绘制的是哪些特质，只是为了让你习惯不知道每个维度代表什么——但仍然能从一个人性格的向量表示中获得很多价值。

我们现在可以说，这个向量部分地代表了我的性格。当你想把另外两个人和我作比较时，这种表示的用处就体现出来了。假设我被一辆 ```bus``` 撞了，需要找一个性格相似的人来替代我。在下图中，这两个人里哪一个和我更相似？


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/personality-two-persons.png"/>
</div>

在处理向量时，一种常用的相似度计算方式是 [cosine_similarity](https://en.wikipedia.org/wiki/Cosine_similarity)（余弦相似度）：


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/cosine-similarity.png"/>
  <br />
  <span class="encoder">Person #1</span> 在性格上和我更相似。指向相同方向的向量（长度也起一定作用）会有更高的余弦相似度得分。
</div>


然而，两个维度仍然不足以捕捉到关于人与人之间差异的足够信息。几十年的心理学研究归纳出了五大特质（以及大量的子特质）。所以让我们在比较中使用全部五个维度：


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/big-five-vectors.png"/>
  <br />
</div>

五个维度的问题在于，我们失去了在二维平面上画出整齐小箭头的能力。这是机器学习中常见的挑战，我们常常不得不在更高维的空间里思考。不过好消息是，cosine_similarity 仍然有效。它适用于任意数量的维度：


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/embeddings-cosine-personality.png"/>
  <br />
  cosine_similarity 适用于任意数量的维度。这些得分要好得多，因为它们是基于被比较事物的更高分辨率的表示计算出来的。
</div>

在本节的结尾，我希望我们能得出两个核心思想：

1. 我们可以把人（以及事物）表示为数字向量（这对机器来说太棒了！）。
2. 我们可以很容易地计算这些向量彼此之间有多相似。


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/section-1-takeaway-vectors-cosine.png"/>
  <br />
</div>


## 词嵌入

<blockquote class='subtle'>
“The gift of words is the gift of deception and illusion” ~Children of Dune
</blockquote>

有了这层理解，我们就可以来看一些训练好的词向量（word-vector）例子（也称为词嵌入），并开始观察它们的一些有趣性质。

这是单词 "king" 的词嵌入（在 Wikipedia 上训练的 GloVe 向量）：

<code class="small_code">
[ 0.50451 ,  0.68607 , -0.59517 , -0.022801,  0.60046 , -0.13498 ,
 -0.08813 ,  0.47377 , -0.61798 , -0.31012 , -0.076666,  1.493   ,
 -0.034189, -0.98173 ,  0.68229 ,  0.81722 , -0.51874 , -0.31503 ,
 -0.55809 ,  0.66421 ,  0.1961  , -0.13495 , -0.11476 , -0.30344 ,
  0.41177 , -2.223   , -1.0756  , -1.0783  , -0.34354 ,  0.33505 ,
  1.9927  , -0.04234 , -0.64319 ,  0.71125 ,  0.49159 ,  0.16754 ,
  0.34344 , -0.25663 , -0.8523  ,  0.1661  ,  0.40102 ,  1.1685  ,
 -1.0137  , -0.21585 , -0.15155 ,  0.78321 , -0.91241 , -1.6106  ,
 -0.64426 , -0.51042 ]
 </code>

这是一组 50 个数字。光看数值我们看不出太多东西。但让我们把它稍微可视化一下，以便和其他词向量做比较。我们把所有这些数字放在一行里：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/king-white-embedding.png"/>
  <br />
</div>

让我们根据数值给这些单元格上色（接近 2 的为红色，接近 0 的为白色，接近 -2 的为蓝色）：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/king-colored-embedding.png"/>
  <br />
</div>

接下来我们忽略数字，只看颜色来表示单元格的值。现在让我们把 "King" 和其他单词作对比：


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/king-man-woman-embedding.png"/>
  <br />
</div>

看到 "Man" 和 "Woman" 彼此之间要比它们各自和 "king" 之间相似得多了吗？这说明了一些东西。这些向量表示捕捉到了这些单词相当多的信息／含义／关联。

这里还有另一组例子（通过纵向扫描各列、寻找颜色相似的列来比较）：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/queen-woman-girl-embeddings.png"/>
  <br />
</div>

有几点值得指出：

1. 贯穿所有这些不同单词的有一条笔直的红色列。它们在那个维度上是相似的（而我们并不知道每个维度编码的是什么）。
2. 你可以看到 "woman" 和 "girl" 在很多地方彼此相似。"man" 和 "boy" 也是如此。
3. "boy" 和 "girl" 也有彼此相似、但又不同于 "woman" 或 "man" 的地方。这些会不会在编码某种模糊的"青年"概念？有可能。
4. 除了最后一个单词，其余都是表示人的单词。我加入了一个物体（water）来展示不同类别之间的差异。比如你可以看到那条蓝色的列一路向下，在 "water" 的嵌入之前就停止了。
5. 有些地方明显地显示出 "king" 和 "queen" 彼此相似、又区别于其他所有单词。这些会不会在编码某种模糊的"皇室"概念？

### 类比

<blockquote class='subtle'>
"Words can carry any burden we wish. All that's required is agreement and a tradition upon which to build." ~God Emperor of Dune
</blockquote>

展示嵌入这一惊人性质的著名例子，就是类比（analogies）的概念。我们可以对词嵌入做加减运算，并得到有趣的结果。最著名的例子是这个公式："king" - "man" + "woman"：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/king-man+woman-gensim.png"/>
  <br />
  使用 python 中的 <a href="https://radimrehurek.com/gensim/">Gensim</a> 库，我们可以对词向量做加减运算，它会找出与结果向量最相似的单词。图中展示了一组最相似的单词，每个都附有其余弦相似度。
</div>

我们可以像之前那样把这个类比可视化：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/king-analogy-viz.png"/>
  <br />
  "king-man+woman" 得到的结果向量并不严格等于 "queen"，但在我们这个集合中的 400,000 个词嵌入里，"queen" 是离它最近的单词。
</div>

现在我们已经看过了训练好的词嵌入，让我们进一步了解训练过程。但在我们讲到 word2vec 之前，需要先看看词嵌入在概念上的"父辈"：神经语言模型（neural language model）。

## 语言建模

<blockquote class='subtle'>
  “The prophet is not diverted by illusions of past, present and future. <strong>The fixity of language determines such linear distinctions.</strong> Prophets hold a key to the lock in a language. <br /> <br />

  This is not a mechanical universe. The linear progression of events is imposed by the observer. Cause and effect? That's not it at all. <strong>The prophet utters fateful words.</strong> You glimpse a thing "destined to occur." But the prophetic instant releases something of infinite portent and power. The universe undergoes a ghostly shift.” ~God Emperor of Dune
</blockquote>

如果要举一个 NLP 应用的例子，最好的例子之一就是智能手机键盘的下一词预测功能。这是数十亿人每天使用数百次的功能。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/swiftkey-keyboard.png"/>
  <br />
</div>

下一词预测是一个可以由*语言模型（language model）*来完成的任务。语言模型可以接收一组单词（比如说两个单词），并尝试预测紧随其后的那个单词。

在上面的截图中，我们可以把模型看作这样一个东西：它接收了这两个绿色的单词（<code class="plain_code mdc-text-green-600">thou shalt</code>），并返回了一组建议（"not" 是概率最高的那个）：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/thou-shalt-_.png"/>
  <br />
</div>

<br />

我们可以把这个模型想象成这样一个黑箱：

<br />

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/language_model_blackbox.png"/>
  <br />
</div>

<br />

但实际上，模型输出的不只是一个单词。它实际上为它所知道的所有单词（模型的"词汇表"，规模可以从几千到超过一百万个单词）输出一个概率得分。键盘应用随后必须找出得分最高的那些单词，并把它们呈现给用户。

<br />

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/language_model_blackbox_output_vector.png"/>
  <br />
  神经语言模型的输出是模型所知道的所有单词的概率得分。我们这里把概率说成百分比，但 40% 实际上在输出向量中会表示为 0.4。
</div>

<br />

训练之后，早期的神经语言模型（[Bengio 2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)）会用三个步骤来计算一次预测：

<br />

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/neural-language-model-prediction.png"/>
  <br />
</div>

<br />

第一步对我们讨论嵌入来说最为相关。训练过程的结果之一，就是这个矩阵，它包含了我们词汇表中每个单词的嵌入。在预测时，我们只需查找输入单词的嵌入，并用它们来计算预测结果：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/neural-language-model-embedding.png"/>
  <br />
</div>

现在让我们转向训练过程，更多地了解这个嵌入矩阵是如何形成的。


## 语言模型训练

<blockquote class='subtle'>
“A process cannot be understood by stopping it. Understanding must move with the flow of the process, must join it and flow with it.” ~Dune
</blockquote>

相比大多数其他机器学习模型，语言模型有一个巨大的优势。这个优势在于，我们能够用连续的文本来训练它们——而这种文本我们有的是。想想我们身边所有的书籍、文章、Wikipedia 内容以及其他形式的文本数据吧。与之相对的是很多其他机器学习模型，它们需要人工设计的特征和专门收集的数据。

> "You shall know a word by the company it keeps" J.R. Firth

单词的嵌入是这样得来的：我们观察它们倾向于和哪些其他单词一起出现。其机制是：

1. 我们获取大量文本数据（比如说，所有 Wikipedia 文章）。然后
2. 我们用一个窗口（比如说三个单词的窗口）在所有这些文本上滑动。
3. 这个滑动窗口为我们的模型生成训练样本。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/wikipedia-sliding-window.png"/>
  <br />
</div>

随着这个窗口在文本上滑动，我们（虚拟地）生成了一个用来训练模型的数据集。为了准确看清这是怎么做到的，让我们看看滑动窗口如何处理这个短语：

> “Thou shalt not make a machine in the likeness of a human mind” ~Dune

开始时，窗口落在句子的前三个单词上：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/lm-sliding-window.png"/>
  <br />
</div>

<br />  

我们把前两个单词作为特征（features），把第三个单词作为标签（label）：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/lm-sliding-window-2.png"/>
  <br />
  我们现在已经生成了数据集中的第一个样本，之后可以用它来训练语言模型。
</div>

<br />  

接着我们把窗口滑动到下一个位置，并创建第二个样本：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/lm-sliding-window-3.png"/>
  <br />
  现在第二个样本也生成了。
</div>

<br />  

很快我们就有了一个更大的数据集，记录着在不同的单词对之后倾向于出现哪些单词：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/lm-sliding-window-4.png"/>
  <br />  
</div>

<br />  

实际上，模型往往是在我们滑动窗口的同时进行训练的。但我觉得在逻辑上把"数据集生成"阶段和训练阶段分开会更清晰。除了基于神经网络的语言建模方法之外，一种叫做 N-grams 的技术也曾被普遍用于训练语言模型（见：[Speech and Language Processing](http://web.stanford.edu/~jurafsky/slp3/) 第 3 章）。要了解从 N-grams 到神经模型的这种转变如何体现在真实产品上，[这里有一篇 2015 年来自 Swiftkey 的博客文章](https://blog.swiftkey.com/neural-networks-a-meaningful-leap-for-mobile-typing/)（Swiftkey 是我最喜欢的 Android 键盘），介绍了他们的神经语言模型，并将其与之前的 N-gram 模型作了比较。我喜欢这个例子，因为它向你展示了嵌入的算法性质如何能用营销话术来描述。

### 同时看两个方向

<blockquote class='subtle'>
"Paradox is a pointer telling you to look beyond it. If paradoxes bother you, that betrays your deep desire for absolutes. The relativist treats a paradox merely as interesting, perhaps amusing or even, dreadful thought, educational." ~God Emperor of Dune
</blockquote>

根据你在本文前面学到的知识，填空：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/jay_was_hit_by_a_.png"/>
  <br />  
</div>

我在这里给你的上下文是空白单词之前的五个单词（以及前面提到过的 "bus"）。我相信大多数人都会猜 ```bus``` 这个词填进空白处。但如果我再给你一条信息——空白之后的一个单词，那会改变你的答案吗？


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/jay_was_hit_by_a_bus.png"/>
  <br />  
</div>

这彻底改变了应该填进空白处的内容。现在 ```red``` 这个词最有可能填入空白。我们从中学到的是：一个特定单词前面和后面的单词都携带着信息价值。事实证明，同时考虑两个方向（我们正在猜测的单词左侧和右侧的单词）能带来更好的词嵌入。让我们看看如何调整训练模型的方式来兼顾这一点。


## Skipgram

<blockquote class='subtle'>
  “Intelligence takes chance with limited data in an arena where mistakes are not only possible but also necessary.” ~Chapterhouse: Dune
</blockquote>

我们不只看目标单词前面的两个单词，还可以看它后面的两个单词。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/continuous-bag-of-words-example.png"/>
  <br />  
</div>

如果这样做，我们虚拟构建并用来训练模型的数据集会是这个样子：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/continuous-bag-of-words-dataset.png"/>
  <br />  
</div>

这被称为 **Continuous Bag of Words** 架构，在 [word2vec 的一篇论文](https://arxiv.org/pdf/1301.3781.pdf) [pdf] 中有描述。另一种同样倾向于展现出优秀结果的架构，做法则略有不同。

这另一种架构不是根据上下文（一个单词前后的单词）来猜测该单词，而是尝试用当前单词来猜测相邻的单词。我们可以把它在训练文本上滑动的窗口想象成这样：


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/skipgram-sliding-window.png"/>
  <br />  
  绿色格子里的单词是输入单词，每个粉色框是一个可能的输出。
</div>

这些粉色框的深浅不同，因为这个滑动窗口实际上在我们的训练数据集中创建了四个独立的样本：

<br />  

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/skipgram-sliding-window-samples.png"/>
  <br />  
</div>

<br />  

这种方法被称为 **skipgram** 架构。我们可以把这个滑动窗口的工作过程可视化如下：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-sliding-window-1.png"/>
  <br />  
</div>

<br />  

这会把这四个样本加入我们的训练数据集：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-sliding-window-2.png"/>
  <br />  
</div>

接着我们把窗口滑动到下一个位置：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-sliding-window-3.png"/>
  <br />  
</div>
<br />  

这会生成我们接下来的四个样本：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-sliding-window-4.png"/>
  <br />  
</div>

再过几个位置，我们就有了多得多的样本：

<br />  

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/skipgram-sliding-window-5.png"/>
  <br />  
</div>


### 重新审视训练过程

<blockquote class="subtle">
  "Muad'Dib learned rapidly because his first training was in how to learn. And the first lesson of all was the basic trust that he could learn. It's shocking to find how many people do not believe they can learn, and how many more believe learning to be difficult." ~Dune
</blockquote>

现在我们有了从现有连续文本中提取出来的 skipgram 训练数据集，让我们看一眼如何用它来训练一个基础的、预测相邻单词的神经语言模型。



<br />  

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/skipgram-language-model-training.png"/>
  <br />  
</div>

我们从数据集中的第一个样本开始。我们取出特征，喂给未训练的模型，请它预测一个合适的相邻单词。

<br />  


<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-language-model-training-2.png"/>
  <br />  
</div>

模型执行那三个步骤，并输出一个预测向量（为词汇表中的每个单词分配一个概率）。由于模型尚未训练，它在此阶段的预测肯定是错的。但没关系。我们知道它本应猜出哪个单词——也就是我们当前用来训练模型的这一行中的标签／输出单元格：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-language-model-training-3.png"/>
  <br />  
  "目标向量（target vector）"是这样一个向量：目标单词的概率为 1，其余所有单词的概率都为 0。
</div>

<br />  

模型偏差有多大？我们把这两个向量相减，得到一个误差向量（error vector）：

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-language-model-training-4.png"/>
  <br />  
</div>

<br />  

现在可以用这个误差向量来更新模型，使得下一次当它得到 <code class="plain_code mdc-text-green-500">not</code> 作为输入时，更有可能猜出 <code class="plain_code mdc-text-pink-500">thou</code>。

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-language-model-training-5.png"/>
  <br />  
</div>

<br />  

这样就完成了训练的第一步。我们继续对数据集中的下一个样本做同样的处理，然后是再下一个，直到我们覆盖了数据集中的所有样本。这样就完成了一个训练 *epoch*（轮次）。我们把这个过程重复若干个 epoch，然后就得到了训练好的模型，可以从中提取出嵌入矩阵，用于任何其他应用。

虽然这加深了我们对这个过程的理解，但这仍然不是 word2vec 实际的训练方式。我们还缺了几个关键的思想。

## 负采样

<blockquote class='subtle'>
“To attempt an understanding of Muad'Dib without understanding his mortal enemies, the Harkonnens, is to attempt seeing Truth without knowing Falsehood. It is the attempt to see the Light without knowing Darkness. It cannot be.” ~Dune
</blockquote>

回想一下这个神经语言模型计算预测的三个步骤：
<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/language-model-expensive.png"/>
  <br />  
</div>

<br />  

从计算的角度看，第三步代价非常高昂——尤其是考虑到我们会对数据集中的每个训练样本都做一次（轻轻松松就达到数千万次）。我们需要做点什么来提升性能。

一种办法是把我们的目标拆成两步：

1. 生成高质量的词嵌入（先不管下一词预测）。
2. 用这些高质量的嵌入来训练一个语言模型（去做下一词预测）。

在本文中我们将聚焦于第 1 步，因为我们关注的是嵌入。为了用一个高性能的模型来生成高质量的嵌入，我们可以把模型的任务从预测相邻单词：

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/predict-neighboring-word.png "/>
  <br />  
</div>

换成这样一个模型：它接收输入单词和输出单词，并输出一个分数，表明它们是不是相邻（0 表示"不相邻"，1 表示"相邻"）。


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/are-the-words-neighbors.png "/>
  <br />  
</div>

这个简单的转换，把我们所需的模型从一个神经网络变成了一个逻辑回归（logistic regression）模型——因此它变得简单得多、计算也快得多。

这个转换要求我们改变数据集的结构——标签现在是一个新的列，取值为 0 或 1。它们将全部是 1，因为我们添加的所有单词都是相邻的。

<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-training-dataset.png "/>
  <br />  
</div>

<br />  

现在这个计算可以以惊人的速度完成了——几分钟内处理数百万个样本。但有一个漏洞我们需要堵上。如果我们所有的样本都是正样本（目标：1），我们就给了一个"耍小聪明"的模型可乘之机——它总是返回 1，从而达到 100% 的准确率，但什么也没学到，生成的是垃圾嵌入。


<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-smartass-model.png "/>
  <br />  
</div>

为了解决这个问题，我们需要向数据集中引入*负样本（negative samples）*——也就是那些不相邻的单词样本。对于这些样本，我们的模型需要返回 0。这下就成了一个模型必须努力才能解决的挑战——但仍然是以惊人的速度。


<br />  

<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-negative-sampling.png "/>
  <br />  
  对于数据集中的每个样本，我们都添加一些<strong>负样本（negative examples）</strong>。它们有着相同的输入单词，以及一个为 0 的标签。
</div>

但我们用什么来填充输出单词呢？我们从词汇表中随机抽取单词。

<br />  


<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-negative-sampling-2.png "/>
  <br />  
</div>

这个思想的灵感来自 [Noise-contrastive estimation](http://proceedings.mlr.press/v9/gutmann10a/gutmann10a.pdf)（噪声对比估计）[pdf]。我们把真实信号（相邻单词的正样本）与噪声（随机选取的、不相邻的单词）作对比。这带来了计算效率和统计效率之间一种很好的权衡。

## 带负采样的 Skipgram（SGNS）

我们现在已经讲解了 word2vec 中的两个核心思想：作为一对，它们合称为带负采样的 skipgram（skipgram with negative sampling）。

<div class="img-div" markdown="0">
  <image src="../images/word2vec/skipgram-with-negative-sampling.png "/>
  <br />  
</div>



## Word2vec 训练过程

<blockquote class="subtle">
"The machine cannot anticipate every problem of importance to humans. It is the difference between serial bits and an unbroken continuum. We have the one; machines are confined to the other." ~God Emperor of Dune
</blockquote>

现在我们已经确立了 skipgram 和负采样这两个核心思想，可以进一步细看 word2vec 实际的训练过程了。

在训练过程开始之前，我们会对要用来训练模型的文本做预处理。在这一步中，我们确定词汇表的大小（我们称之为 <code class="plain_code mdc-text-amber-700">vocab_size</code>，可以想成是，比如说，10,000）以及哪些单词属于它。

在训练阶段开始时，我们创建两个矩阵——一个 <code class="plain_code mdc-text-green-500">Embedding</code> 矩阵和一个 <code class="plain_code mdc-text-purple-500">Context</code> 矩阵。这两个矩阵为词汇表中的每个单词都保存了一个嵌入（所以 <code class="plain_code mdc-text-amber-700">vocab_size</code> 是它们的维度之一）。第二个维度是我们希望每个嵌入有多长（<code class="plain_code mdc-text-amber-900">embedding_size</code>——300 是一个常见值，但我们在本文前面看过一个 50 的例子）。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/word2vec-embedding-context-matrix.png "/>
  <br />  
</div>

在训练过程开始时，我们用随机值初始化这些矩阵。然后我们开始训练过程。在每个训练步骤中，我们取一个正样本及其关联的负样本。让我们取出第一组：


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/word2vec-training-example.png "/>
  <br />  
</div>

现在我们有四个单词：输入单词 <code class="plain_code mdc-text-green-500">not</code> 和输出／上下文单词：<code class="plain_code mdc-text-purple-500">thou</code>（真正的相邻词）、<code class="plain_code mdc-text-purple-500">aaron</code> 和 <code class="plain_code mdc-text-purple-500">taco</code>（负样本）。我们接着查找它们的嵌入——对于输入单词，我们在 <code class="plain_code mdc-text-green-500">Embedding</code> 矩阵中查找。对于上下文单词，我们在 <code class="plain_code mdc-text-purple-500">Context</code> 矩阵中查找（尽管这两个矩阵都为词汇表中的每个单词保存了一个嵌入）。

<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/word2vec-lookup-embeddings.png "/>
  <br />  
</div>

然后，我们取输入嵌入与每个上下文嵌入的点积（dot product）。在每种情况下，这都会得到一个数字，这个数字表明输入嵌入和上下文嵌入的相似度。


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/word2vec-training-dot-product.png "/>
  <br />  
</div>

现在我们需要一种办法，把这些分数变成看起来像概率的东西——我们需要它们全部为正、并且取值在 0 到 1 之间。这正是 [sigmoid](https://jalammar.github.io/feedforward-neural-networks-visual-interactive/#sigmoid-visualization)（[逻辑斯谛运算](https://en.wikipedia.org/wiki/Logistic_function)）大显身手的任务。


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/word2vec-training-dot-product-sigmoid.png "/>
  <br />  
</div>

现在我们可以把 sigmoid 运算的输出当作模型对这些样本的输出了。你可以看到，无论是在 sigmoid 运算之前还是之后，<code class="plain_code mdc-text-purple-500">taco</code> 的分数都最高，而 <code class="plain_code mdc-text-purple-500">aaron</code> 的分数仍然最低。

既然未训练的模型已经做出了预测，而我们又有一个实际的目标标签可以拿来比较，那就让我们计算一下模型的预测中有多少误差。为此，我们只需用目标标签减去 sigmoid 分数。


<div class="img-div-any-width" markdown="0">
  <image src="../images/word2vec/word2vec-training-error.png "/>
  <br />  
  <code class="plain_code mdc-text-yellow-800">error</code> = <code class="plain_code mdc-text-pink-400">target</code> - <code class="plain_code mdc-text-grey-900">sigmoid_scores</code>
</div>

<br />

"机器学习"中"学习"的部分来了。我们现在可以用这个误差分数来调整 <code class="plain_code mdc-text-green-500">not</code>、<code class="plain_code mdc-text-purple-500">thou</code>、<code class="plain_code mdc-text-purple-500">aaron</code> 和 <code class="plain_code mdc-text-purple-500">taco</code> 的嵌入，使得下一次我们做这个计算时，结果会更接近目标分数。

<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-training-update.png "/>
  <br />  
</div>

这样就完成了一个训练步骤。我们从中得到了这一步所涉及单词（<code class="plain_code mdc-text-green-500">not</code>、<code class="plain_code mdc-text-purple-500">thou</code>、<code class="plain_code mdc-text-purple-500">aaron</code> 和 <code class="plain_code mdc-text-purple-500">taco</code>）的略微更好一些的嵌入。现在我们继续进行下一步（下一个正样本及其关联的负样本），再次执行同样的过程。

<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-training-example-2.png "/>
  <br />  
</div>

随着我们把整个数据集循环若干遍，这些嵌入会持续得到改善。然后我们就可以停止训练过程，丢弃 <code class="plain_code mdc-text-purple-500">Context</code> 矩阵，把 <code class="plain_code mdc-text-green-500">Embeddings</code> 矩阵用作我们为下一个任务准备的预训练嵌入。

## 窗口大小与负样本数量
word2vec 训练过程中有两个关键的超参数，分别是窗口大小（window size）和负样本数量（number of negative samples）。

<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-window-size.png "/>
  <br />  
</div>

不同的任务适合不同的窗口大小。一条[经验法则](https://youtu.be/tAxrlAVw-Tk?t=648)是：较小的窗口大小（2-15）会得到这样的嵌入——两个嵌入之间的高相似度分数表明这两个单词是*可互换的（interchangeable）*（注意，如果我们只看周围的单词，反义词往往是可互换的——例如 *good* 和 *bad* 经常出现在相似的上下文中）。较大的窗口大小（15-50，甚至更大）会得到这样的嵌入——相似度更能体现单词之间的*相关性（relatedness）*。在实践中，你常常得提供一些[标注](https://youtu.be/ao52o9l6KGw?t=287)来引导嵌入过程，从而为你的任务得到有用的相似度含义。Gensim 的默认窗口大小是 5（输入单词前面 5 个单词、后面 5 个单词，再加上输入单词本身）。


<div class="img-div" markdown="0">
  <image src="../images/word2vec/word2vec-negative-samples.png "/>
  <br />  
</div>

负样本的数量是训练过程的另一个因素。原始论文给出的建议是 5-20 是一个不错的负样本数量。它还指出，当你有足够大的数据集时，2-5 似乎就够了。Gensim 的默认值是 5 个负样本。

## 结论

<blockquote class="subtle">
“If it falls outside your yardsticks, then you are engaged with intelligence, not with automation”  ~God Emperor of Dune
</blockquote>

我希望你现在对词嵌入和 word2vec 算法有了一种感觉。我也希望，现在当你读到一篇提及"skip gram with negative sampling"（SGNS）的论文时（就像本文开头那些推荐系统论文），你对这些概念会有更好的理解。一如既往，欢迎任何反馈 <a href="https://twitter.com/JayAlammar">@JayAlammar</a>。

## 参考资料与延伸阅读
* [Distributed Representations of Words and Phrases and their Compositionality](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) [pdf]
* [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf) [pdf]
* [A Neural Probabilistic Language Model](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) [pdf]
* [Speech and Language Processing](https://web.stanford.edu/~jurafsky/slp3/) by Dan Jurafsky and James H. Martin is a leading resource for NLP. Word2vec is tackled in Chapter 6.
* [Neural Network Methods in Natural Language Processing](https://www.amazon.com/Language-Processing-Synthesis-Lectures-Technologies/dp/1627052984) by [Yoav Goldberg](https://twitter.com/yoavgo) is a great read for neural NLP topics.
* [Chris McCormick](http://mccormickml.com/) has written some great blog posts about Word2vec. He also just released [The Inner Workings of word2vec](https://www.preview.nearist.ai/paid-ebook-and-tutorial), an E-book focused on the internals of word2vec.
* Want to read the code? Here are two options:
  * Gensim's [python implementation](https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/models/word2vec.py) of word2vec
  * Mikolov's original [implementation in C](https://github.com/tmikolov/word2vec/blob/master/word2vec.c) -- better yet, this [version with detailed comments](https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c) from Chris McCormick.
* [Evaluating distributional models of compositional semantics](http://sro.sussex.ac.uk/id/eprint/61062/1/Batchkarov,%20Miroslav%20Manov.pdf)
* [On word embeddings](http://ruder.io/word-embeddings-1/index.html), [part 2](http://ruder.io/word-embeddings-softmax/)
* [Dune](https://www.amazon.com/Dune-Frank-Herbert/dp/0441172717/)

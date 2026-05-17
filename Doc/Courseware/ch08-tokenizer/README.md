# ch08 · 分词器

> 模型只认数字。文本到 id 的那座桥就是 **分词器（tokenizer）**。
> 这一章回答三个问题：
> 1. 为什么不直接按字符或按词切？子词（subword）赢在哪
> 2. BPE / WordPiece / Unigram 三个主流算法到底差在哪
> 3. 中英混合场景怎么训一个不烂的分词器
>
> M3 第一章。理论为主 + 一段 50 行的 BPE 训练 demo，为 M4 echo-mini 训自己词表打底。

## 学习目标

1. 能解释字符级、词级、子词级三种粒度的取舍（词表大小 / 序列长度 / OOV（Out-Of-Vocabulary，词表外））
2. 能讲清 BPE 的训练过程（合并最频繁的 byte pair）和它为什么有效
3. 知道 byte-level BPE 在多语言/中英混合下解决了什么问题

## 前置依赖

- 无强依赖。建议先了解过 `nn.Embedding` 把 id 映射成向量这一步（ch03）

---

## 1. 三种粒度

设语料 `"the cat sat on the mat"`，词表大小 `V`，序列长度 `L`。

| 粒度 | 切法 | V | L | OOV |
|---|---|---|---|---|
| 字符 | `t,h,e, ,c,a,t,...` | 极小（英文 ~100，含中文 ~5k+） | 极长 | 几乎不会有 |
| 词 | `the,cat,sat,on,the,mat` | 极大（英文 ~50万，含变形/复合词更多） | 短 | 必然有，新词、拼写错误、罕见专有名词全炸 |
| 子词 | `chat,ting`（"chatting"被切开） | 中等（GPT-2 是 50257） | 中等 | 理论上无（任何词能拆成子串组合） |

**核心矛盾**：V 大 → embedding 矩阵大、softmax 慢；L 长 → attention `O(L²)` 直接炸。子词是这条 V-L 曲线上的甜点。

> 实际数字感受：把 1k 字符的英文段落切成
> - 字符级：~1000 token
> - GPT-2 BPE：~250 token
> - 词级：~180 token，但词表得几十万

---

## 2. BPE（Byte Pair Encoding）

> 1994 年 Gage 发明，原本是数据压缩算法。Sennrich 2016 把它搬到 NMT（Neural Machine Translation，神经机器翻译）里，从此成为 NLP 标配。GPT-2 / GPT-3 / LLaMA 都用 BPE 系。

### 2.1 训练流程

```
输入：原始语料；目标词表大小 V（如 16k）
1. 初始化：把每个词拆成字符序列，结尾加个特殊符号（如 </w>）
   "low" → ["l","o","w","</w>"]
2. 统计语料里所有相邻字符对（pair）的频次
3. 取频次最高的 pair（如 ("e","s") 出现 3000 次），合并为新 token "es"
4. 把语料里所有这个 pair 都替换成新 token，词表 size +1
5. 回到第 2 步，直到词表达到目标大小 V
```

> `</w>` 干嘛的？标记**词尾**。这样 "est</w>"（词尾，如 "best"）和 "est"（词中，如 "estimate"）会被当作不同 token，避免错误共享。byte-level BPE 用空格前缀（GPT-2 的 `Ġ`）达到同样目的——见 §4，二者**不会同时出现**，是两套并列方案。

### 2.2 为什么有效

- **常见词整体编码**：高频词（the, ing, ed）在迭代里被尽早合并成单 token，序列短
- **罕见词降级到子串**：unfortunately 切成 `un|fortunate|ly`，没见过的"trumpily"也能切成 `trump|ily`，没有 OOV
- **形态共享**：`run/runs/running/runner` 共享 `run` 这个 token，参数效率高

### 2.3 推理时怎么用

训完得到一份 `merges.txt`（合并规则按训练时的合并顺序排，先合并的优先级最高）+ `vocab.json`（token→id）。
推理是**按 merge 全局优先级**反复扫描：每轮在序列里找当前可应用的、优先级最高的那条 merge 规则，全局替换，再进入下一轮，直到没规则可用。

```
"running" → ["r","u","n","n","i","n","g","</w>"]
轮 1：序列里能匹配的规则中优先级最高是 ("n","n") → ["r","u","nn","i","n","g","</w>"]
轮 2：能匹配的最高优先级是 ("i","n")             → ["r","u","nn","in","g","</w>"]
... → 直到序列里没有任何 merge 规则能匹配
```

> 注意不是"从左到右扫一遍"。同一序列里若有多条可应用规则，**先应用 merges.txt 里位置靠前（训练时更早合并）的那条**。工程实现常用 priority queue 或缓存加速。

### 自检

1. BPE 的"贪心"贪在哪？
2. 给个英文句子，词表 V 越大，平均 token 数会越多还是越少？

<details markdown="1">
<summary>答案速查</summary>

1. 训练时每一步贪心选**当前频次最高**的 pair 合并，不回溯（不考虑"现在合并 X，未来 Y 频次会变得更高"这种全局最优）

2. 越少。V 大意味着合并次数多，更多常见 n-gram 被打包成单 token，平均序列变短

</details>

---

## 3. WordPiece 与 Unigram（一段话讲完）

### WordPiece（BERT 用）

- 与 BPE 几乎一致，唯一差别在选合并 pair 的 **打分函数**
- BPE 选 `count(xy)` 最高，WordPiece 选 `count(xy) / (count(x) * count(y))` 最高
- 直觉：不只看共现频次，还看"是否互相独立的两个 token 凑巧在一起"——分母惩罚高频通用字符。这本质是 **PMI（Pointwise Mutual Information，点互信息）** 的非对数形式，要求两个 token 共现频次高且各自独立频次低才值得合并
- **推理也不一样**：BPE 按 merge 优先级反复合并；WordPiece 按**最长前缀贪心**从左切（找词表里最长匹配前缀，砍掉，剩下加 `##` 前缀继续）。匹配不上就 `[UNK]` —— 这是 BERT 偶尔吐 `[UNK]` 的根源
- 实际差异不大；BERT 时代选它纯属历史路径

### Unigram（SentencePiece 默认 / T5 / mBART / ALBERT）

- 反向操作：先建一个超大词表（含所有可能子串），再**逐步删掉对似然贡献最小的 token**直到达到目标 V
- 优点：训练时同一个词可能有多种切法，给训练加随机性（subword regularization）
- 缺点：训练慢，工程实现也复杂

### 选哪个

工程上**默认 BPE**。GPT 系全用，LLaMA 系（含 LLaMA-1/2/3）全用 SentencePiece-BPE，HF `tokenizers` 库 BPE 支持也最稳。Unigram 在多语言场景偶尔略好。

---

## 4. Byte-level BPE 与中英混合

### 4.1 普通 BPE 的中文困境

普通 BPE 以**字符**为最小单元。处理英文时字符就是 ASCII 码（256 个），干净；处理中文时每个汉字是一个独立"字符"（基础集 ~6000 常用 + 2 万生僻），起步词表就几万，浪费严重。
更糟的是遇到训练时没见过的 emoji / 古文字 → 整个落入 `<unk>` → OOV。

### 4.2 Byte-level BPE（GPT-2 起的标配）

> 关键 trick：**最小单元不是字符，是 byte**。

UTF-8 编码下任意字符都能拆成 1–4 个 byte，byte 总共只有 256 种。举例：

```
"A"  → UTF-8 → [0x41]                  1 byte（ASCII 兼容）
"é"  → UTF-8 → [0xC3, 0xA9]            2 byte
"你" → UTF-8 → [0xE4, 0xBD, 0xA0]      3 byte（CJK（Chinese-Japanese-Korean，中日韩统一表意文字）通常 3 byte）
"😀" → UTF-8 → [0xF0, 0x9F, 0x98, 0x80] 4 byte（emoji 通常 4 byte）
```

所以：

- 起步词表永远 256，不管处理什么语言
- **绝对无 OOV**：再奇怪的字符串都能落到 byte
- 中英混合天然支持，emoji 也能切（虽然切得碎）

代价：高频中文字符会被拆成 ~3 个 byte，序列略长——但 BPE 训练里这些 byte 组合会迅速被合并成单 token，最终序列长度可接受。

### 4.3 训中文 / 中英混合分词器的实务

- **必用 byte-level BPE**（HF（HuggingFace，开源大模型社区与工具集）`tokenizers` 里 `ByteLevelBPETokenizer` 或 `BPE(..., byte_fallback=True)`）
- 词表 ~16k–32k 适合 echo-mini 这个量级；大模型常 32k–128k
- pre-tokenizer 通常做三件事：按空白切英文词、按 Unicode 类别**把 CJK 字符逐字拆开**、单独抽出标点和数字。GPT-4 / LLaMA-3 / Qwen 的 pre-tokenizer 用一段 regex 同时处理。中文每个字独立进 BPE 后，靠 byte 组合学出"词"或常见短语
- **special token 提前定好**：`<pad>`, `<bos>`, `<eos>`, `<unk>`，对话模型还要 `<|user|>`, `<|assistant|>` 等，训练前注册，避免后续重训

### 自检

1. byte-level BPE 为什么号称"无 OOV"？
2. 训中文分词器时，词表大小定 5000 vs 30000 各自的代价是什么？

<details markdown="1">
<summary>答案速查</summary>

1. 任何字符在 UTF-8 下都能拆成 1–4 个 byte，byte 只有 256 种，全在初始词表里——再奇怪的字符也能用 byte 序列表示，不会落到 `<unk>`

2. 5000 太小：常见汉字都没合并完，平均序列长，attention `O(L²)` 吃亏；30000 偏大：embedding/softmax 矩阵更大，小模型上参数浪费、训练慢。echo-mini 量级 16k–24k 是常见甜点

</details>

---

## 5. 练习

落到 `Playground/ch08-tokenizer/`：

| 脚本 | 内容 |
|---|---|
| `01_tokenize_compare.py` | 字符 / 词 / GPT-2 BPE 三种分词在同一段中英混合文本上的对比（token 数、token 形态） |
| `02_train_bpe.py` | 用 HF `tokenizers` 在 tiny shakespeare 上训一个 ~2k 的 byte-level BPE，编码-解码 round-trip 验证 |

跑法：

```bash
uv run python Playground/ch08-tokenizer/01_tokenize_compare.py
uv run python Playground/ch08-tokenizer/02_train_bpe.py
```

`02` 首次跑会从 HF 下载 GPT-2 tokenizer（~1MB，腾讯镜像不缓存 HF，略慢）。

## 思考题

1. 为什么 LLaMA-3 把词表从 32k 扩到 128k 后多语言性能涨明显？代价是什么？
2. 推理时遇到分词器从没见过的特殊符号（如新版 emoji），byte-level BPE 怎么处理它？输出端又会怎么解码回来？

## 参考资料

- **Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units"** (2016)：BPE 用于 NLP 的开山之作
- **Radford et al., "Language Models are Unsupervised Multitask Learners"** (GPT-2)：byte-level BPE 的经典实现，§2.2 "Input Representation"
- **HuggingFace 课程**：<https://huggingface.co/learn/nlp-course/chapter6>，分词器训练实操最易上手
- **Karpathy, "Let's build the GPT Tokenizer"**：从零手撕 BPE 的视频，比读论文直观

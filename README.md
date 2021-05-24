# CourtviewGeneration

# T5

跟BERT一样，T5也是Google出品的预训练模型，来自论文为[《Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer》](https://arxiv.org/abs/1910.10683)，Github为[text-to-text-transfer-transformer](https://github.com/google-research/text-to-text-transfer-transformer)。T5的理念就是“万事皆可Seq2Seq”，它使用了标准的Encoder-Decoder模型，并且构建了无监督/有监督的文本生成预训练任务，最终将效果推向了一个新高度。

![“万事皆可Seq2Seq”的T5](https://kexue.fm/usr/uploads/2020/11/4211830191.png)

T5的预训练包含无监督和有监督两部分。无监督部分使用的是Google构建的近800G的语料（论文称之为C4），而训练目标则跟BERT类似，只不过改成了Seq2Seq版本，我们可以将它看成一个高级版的完形填空问题：

> **输入：** 明月几时有，[M0]问青天，不知[M1]，今夕是何年？我欲[M2]归去，又恐琼楼玉宇，高处[M3]；起舞[M4]清影，何似在人间。
>
> **输出：** [M0]把酒[M1]天上宫阙[M2]乘风[M3]不胜寒[M4]弄

而有监督部分，则是收集了常见的NLP监督任务数据，并也统一转化为SeqSeq任务来训练。

比如情感分类可以这样转化：

> **输入：** 识别该句子的情感倾向：这趟北京之旅我感觉很不错。
>
> **输出：** 正面

主题分类可以这样转化：

> **输入：** 下面是一则什么新闻？八个月了，终于又能在赛场上看到女排姑娘们了。
>
> **输出：** 体育

阅读理解可以这样转化：

> **输入：** 阅读理解：特朗普与拜登共同竞选下一任美国总统。根据上述信息回答问题：特朗普是哪国人？
>
> **输出：** 美国

这种转化跟GPT2、GPT3、PET的思想都是一致的，都是希望用文字把我们要做的任务表达出来，然后都转化为文字的预测.

# mT5

mT5，即Multilingual T5，T5的多国语言版，出自论文[《mT5: A massively multilingual pre-trained text-to-text transformer》](https://arxiv.org/abs/2010.11934)，Github为[multilingual-t5](https://github.com/google-research/multilingual-t5)，这也是将多语种NLP任务的榜单推到了一个新高度了。

## Config

T5模型的配置文件是gin格式的，这不符合bert4keras的输入，使用者请根据所给的gin和下述模版构建对应的config.json文件。

下面是mT5 small版的参考config.json：

```python
{
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 512,
  "initializer_range": 0.02,
  "intermediate_size": 1024,
  "num_attention_heads": 6,
  "attention_head_size": 64,
  "num_hidden_layers": 8,
  "vocab_size": 250112,
  "hidden_act": ["gelu", "linear"]
}
```

一般要修改的是`hidden_size`、`intermediate_size`、`num_attention_heads`、`attention_head_size`和`num_hidden_layers`这几个参数。

# 训练过程

## 1. 预训练

mT5使用的Tokenizer是[sentencepiece](https://github.com/google/sentencepiece)，这是一个C++所写的分词库，具有高效轻便的特点，但是很遗憾，对于中文来说它并不是特别友好，主要体现为：

> 1、sentencepiece会把某些全角符号强制转化为半角符号，这在某些情况下是难以接受的，而且还可能影响任务的评测结果；
>
> 2、sentencepiece内置的算法虽然有能力分出中文词来，但对于中文分词来说其实还是不够智能的；
>
> 3、sentencepiece用C++写的，虽然开源了，但对于用惯Python的人来说C++就相当于黑箱，难以阅读源码，改起来也不容易。

这些特点需要将Tokenizer切换回BERT的Tokenizer。但直接替换原始版本的中文BERT的Tokenizer是不够的，一来是以词为单位来做生成模型能获得更好的效果，二来哪怕只看字中文BERT的vocab.txt也是很不完善的，漏了一些常见的标点符号（如双引号）和中文字（比如“琊”等）。为此，作者选择给BERT的tokenizer加入分词功能，并进一步完善vocab.txt。

- 具体来说，往原始中文BERT的token_dict里边加入结巴分词的前20万个词
- 然后修改Tokenizer的逻辑，使得它能够切分出词来，这些改动都已经内置在bert4keras中了，直接调用就行。
- 接着，用这个修改后的Tokenizer去遍历切分准备的预训练语料，统计各个token的频数，最后只保留最高频的5万个token，得到一个规模为5万的vocab.txt来构建最终的Tokenizer。

预训练模型直接使用了作者在所选的预训练语料上训练的模型。模型位于

```python
CourtviewGeneration/tokenizer/sentencepiece_cn.model
```

## 2. 精简Embedding

mT5涵盖了101种语言，总词表有25万，而且它采用的T5.1.1结构的Softmax还不共享参数，这就导致了Embedding层占用了相当多的参数量，比如mT5 small的参数量为3亿，其中Embedding相关的就占了2.5亿，关键是里边的大部分参数我们都用不上，纯粹是不必要的浪费。

作者用这个25万token的tokenizer对收集的几十G中文语料分词，统计分词结果，然后按照词频选择前面的部分（最后保留了3万多个token）。

使用的词表也是作者最后生成的词表。词表位于

```python
CourtviewGeneration/tokenizer/sentencepiece_cn_keep_tokens.json
```

**以上这两个文件均复制到mt5文件夹下**

## 3. mT5模型下载

首先，要想办法下载Google开放的权重，最简单的方式，是找一台能科学上网的服务器，在上面安装gsutil，然后执行

```
gsutil cp -r gs://t5-data/pretrained_models/mt5/small .
```

T5使用sentencepiece作为tokenizer，mT5的tokenizer模型下载地址为

```
gsutil cp -r gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model .
```

将下载的模型添加至mt5文件夹下

## 4. 训练

```
python task_autotitle_csl.py
```

**实验结果：**

|           | Rouge-L | Rouge-1 | Rouge-2 | BLEU  |
| :-------: | :-----: | :-----: | :-----: | :---: |
| mT5 small |  76.58  |  74.21  |  63.33  | 51.96 |

## 5. courtView生成

```
python generate_summ.py
```

**输出结果示例：**

本院查明部分：

```
<2>经审理查明，<date>，<name>与<name>等人一起在新宾<name>自治县木奇镇<name>家中吃饭，后赶到的<name>也加入其中。席间，<name>与<name>因琐事发生争吵，<name>进行劝阻并将<name>送离<name>家。而后<name>又持刀返回到<name>家，与正欲出门的<name>相遇，<name>见到<name>持刀再次返回<name>家，遂用劈材棒将<name>头部打伤。经法医鉴定：<name>损伤评定为轻伤。案发后，<name>经公安机关传唤到案。上述事实有案件来源；到案经过；鉴定所法医人体损伤程度鉴定意见书；办案说明；户籍证明；<name>陈述；证人<name>、<name>、<name>、<name>等人证言；<name>的供述与辩解等证据材料证实，足以认定。案发后<name>与<name>达成刑事和解协议，赔偿<name>经济损失，取得<name>谅解，具有认罪、悔罪表现，且本案中<name>有一定过错，对<name>宣告缓刑对所居住社区无重大不良影响。
```

原court view：

```
<name>故意伤害他人身体，致人轻伤，犯罪事实清楚，证据确实充分。
```

生成court view：

```
<name>故意伤害他人身体,致人轻伤,犯罪事实清楚,证据确实充分。
```

---

本院查明部分：

```
<3>经审理查明，<date>，<name>驾驶鲁<name>号<name>托车沿临朐县东红路由北向南行驶至<name>路口<name>，与前方顺行的<name>驾驶的电动三轮车相撞，致使<name>受伤后于<date>死亡，造成道路交通事故。经大队认定书认定，<name>负事故的全部责任。案发后，<name>主动到公安机关投案，并如实供述了交通肇事的犯罪事实。另查明，附带民事诉讼<name>、<name>、<name>、<name>因<name>死亡造成的经济损失：丧葬费<number>元，死亡赔偿金<number>元，护理费<number>元，伙食补助费<number>元，误工费<number>元，交通费<number>元，医疗费<number>元，施救费<number>元，共计<number>元。<name>驾驶的鲁<name>号<name>托车于<date>在东营中心公司加入机动车交通事故责任强制保险摩托车定额保险。保险期间自<date>起至<date>止。在审理过程中，附带民事诉讼<name>、<name>、<name>、<name>与<name>已达成和解，<name>在<name>险限额外赔偿附带民事诉讼<name>因<name>死亡造成的各项经济损失<number>元，并已取得<name>家属的谅解。上述事实，<name>在开庭审理过程中亦无异议，且有书证人口信息、到案经过、道路
```

原court view：

```
<name>违反道路交通安全法规，发生交通事故，致一人死亡，负事故的全部责任。
```

生成court view：

```
<name>违反道路交通安全法规,驾驶机动车发生交通事故,致一人死亡,负事故的全部责任。
```


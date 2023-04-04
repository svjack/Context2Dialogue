<!-- PROJECT LOGO -->
<br />
<p align="center">
  <h3 align="center">Context2Dialogue</h3>

  <p align="center">
   		A Dialogue Context Generator based on Text or Image
    <br />
  </p>
</p>

[中文简介](README.md)

## Introduction
### Project view in the point of dialogue goal
Given the first sentence of a dialogue, one can get the subsequent context by
a seq2seq style model. I have released two repositories called [svjack/Daliy-Dialogue](https://github.com/svjack/Daliy-Dialogue) and [svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue) do this kinds of works. </br>

The difference between them is, the former is simply a GPT or Bloom seq2seq style model that predict the subsequent without knowledge's help or restraint, the latter use General Language Model (GLM) do auxiliary and do some reconstructions. </br>

This project focus on achieve the goal like [deepset](https://huggingface.co/deepset)'s [wikipedia-assistant](https://huggingface.co/spaces/deepset/wikipedia-assistant) in Dialogue domain. [wikipedia-assistant](https://huggingface.co/spaces/deepset/wikipedia-assistant) use a seq2seq model to generate long form answer based on wikipedia context (recall by a faiss index). The answers' validity is guaranteed by the context, which makes them more closed to reality.</br>

In the  [svjack/Daliy-Dialogue](https://github.com/svjack/Daliy-Dialogue) situation, no reality guarantee, In the [svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue) situation, the reality guarantee is rely on pretrained General Language Model (GLM). This project more close to the fact compared with above two
 repositories and have a relative rapid running performance (speed) than  [svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue).</br>
 
Try and compare [svjack/context-dialogue-chinese-sample-search](https://huggingface.co/spaces/svjack/context-dialogue-chinese-sample-search) and [svjack/bloom-gpt-dialogue-chinese-sample-search](https://huggingface.co/spaces/svjack/bloom-gpt-dialogue-chinese-sample-search)
with similar questions as input, you will understand the difference between this project and first sentence guide dialogue prediction models in Chinese domain.

### Project view in the point of generation
[svjack/docvqa-gen](https://github.com/svjack/docvqa-gen) is a project that One can use text or Document Image as Input and get question-answer pairs as Output. In the point of generation is a “context to qa” generator. This project is the counterpart in “context to dialogue” scene.


## HuggingFace demonstration

### Used Related Model demonstration (Name startswith svjack is trained by myself)
|Name |HuggingFace Model link| HuggingFace Space link | Language | Model skeleton |
|---------|--------|-------|-------|------|
| svjack/summary-dialogue-eng | https://huggingface.co/svjack/summary-dialogue-eng | https://huggingface.co/spaces/svjack/English-Context-Dialogue-Generator | English | T5 |
| svjack/summary-dialogue | https://huggingface.co/svjack/summary-dialogue | https://huggingface.co/spaces/svjack/Chinese-Context-Dialogue-Generator | Chinese | T5 |
| daspartho/prompt-extend | https://huggingface.co/daspartho/prompt-extend | https://huggingface.co/spaces/daspartho/prompt-extend | English | GPT2 |
| svjack/prompt-extend-chinese-gpt | https://huggingface.co/svjack/prompt-extend-chinese-gpt | https://huggingface.co/spaces/svjack/prompt-extend-gpt-chinese | Chinese | GPT2 |
| nlpconnect/vit-gpt2-image-captioning | https://huggingface.co/nlpconnect/vit-gpt2-image-captioning |https://huggingface.co/spaces/SRDdev/Image-Caption | English | VIT X GPT2 |
| YeungNLP/ofa-cn-base-muge-v2 |https://huggingface.co/YeungNLP/ofa-cn-base-muge-v2 ||Chinese| OFA |

<br/>
Try to select the "do_sample" option when you do not get the satisfying output when try https://huggingface.co/spaces/svjack/English-Context-Dialogue-Generator and https://huggingface.co/spaces/svjack/Chinese-Context-Dialogue-Generator
You may have a try to wrap this option by a reinforce style reward extention in a reinforcement learning manner to generate samples.

### Dataset generate by above models demonstration
|Name |HuggingFace Dataset link| HuggingFace Space link | Language |
|---------|--------|-------|-------|
| svjack/context-dialogue-generate-ds-zh-v1 | https://huggingface.co/datasets/svjack/context-dialogue-generate-ds-zh-v1 | https://huggingface.co/spaces/svjack/context-dialogue-chinese-sample-search | Chinese |

## Installation and Instructions

Refer to HuggingFace Model cards.

### Installation
```bash
pip install -r requirements.txt
```
### Instructions
#### simply call one times

* 1 English Dialogue Generator in Text 🦅:

```python
from summary_reverse_pred_eng_native import *

en_context = "The Wisconsin Territorial Centennial half dollar was designed by David Parsons and Benjamin Hawkins and minted by the United States Bureau of the Mint in 1936. The obverse (pictured) depicts a pick axe and lead ore, referring to the lead mining in early Wisconsin"

simple_pred(en_context, do_sample = False)
```

will output:
```json
['Have you seen the Wisconsin Territorial Centennial half dollar?',
 'Yeah, it was designed by David Parsons and Benjamin Hawkins.',
 'What is it?',
 "It's a half dollar with a pick axe and lead ore.",
 "That's great!"]
```

</br>

* 2 Chinese Dialogue Generator in Text 🐰:

```python
from summary_reverse_pred_native import *

zh_context = "巴伐利亚号战列舰[a]（德语：SMS Bayern[b]）是德意志帝国海军巴伐利亚级战列舰的主导舰。该舰于1915年2月下水并于1916年7月开始服役，但已来不及参加日德兰海战。它的主炮包括分布在四座双联装炮塔中的八门380毫米口径炮，这比其前身国王级配备的十门305毫米口径炮有了显著改进。[c]舰只连同它的三艘姊妹舰已经形成了公海舰队第四战列分舰队的核心。而这当中仅有一艘，即巴登号完成建造；另外两艘则在第一次世界大战后期，当生产需求被转移至U型潜艇后而撤销。"

simple_pred(zh_context, do_sample = False)
```

will output:
```json
['杰克:巴罗利亚号战列舰是哪个国家?',
 '安娜:德意志帝国海军的。它在1915年2月下水,19167开始服役',
 '杰克:该舰的主要装备是什么?',
 '安娜:主炮包括四座双联装炮塔中的八门380毫米口径。',
 '杰克:这比其前身国王级装备的十门305毫米口径炮有了明显改进。',
 '安娜:但只有三艘姊妹舰已经形成公海舰队第四战列分的核心。',
 '杰克:这是为什么?',
 '安娜:它是二战后建造的。']
```

</br>

#### some bad cases and remedy

```python
from summary_reverse_pred_eng_native import *

en_context = "Cyclone Gabrielle causes widespread damage and flooding across New Zealand."

simple_pred(en_context, do_sample = False)
```

will output:
```json
['file_photo>',
 'Cyclone Gabrielle!',
 "What's that?",
 "It's massive damage and flooding across New Zealand."]
```

</br>

```python
from summary_reverse_pred_native import *

zh_context = '长宗我部本队坚守长濑川，当藤堂军接近时，长宗我部队立即命令部队以铁炮射击[47]，然后进入混战，然后长宗我部进行突击，藤堂高刑、藤堂氏胜和桑名一孝战死[46]，藤堂队先锋被消灭，长宗我部三军进行挟击本队'

simple_pred(zh_context, do_sample = False)
```

will output:
```json
['杰克:长宗我部本队坚守长濑川,当藤堂军接近时,部队立即命令以铁炮射击[47],然后进入混战,进行突击,藤堂高刑、氏胜和桑名一孝战死,藤堂队先锋被消灭,三军挟击。']
```
</br>
</br>
The above two cases show that some sentence will yield dialogue predictions with few rounds. A simple but valid remedy of this problem is set "do_sample" parameter to True and try multiple times.
</br>
</br>
Below is the remedy demonstration.

```python
from summary_reverse_pred_eng_native import *

en_context = "Cyclone Gabrielle causes widespread damage and flooding across New Zealand."

df = sample_pred_wrapper(en_context, i2c_obj = i2c_obj)
df["dialogue"].values.tolist()
```

will output:
```json
[['What is the weather like in New Zealand?',
  'Cyclone Gabrielle, a massive flooding and hailstorm.',
  "I'm not sure what to make of it",
  'You can talk about it',
  "It's really terrible!"],
 ["What's the weather like in New Zealand?",
  'Cyclone Gabrielle is causing massive damage and flooding.',
  "I guess it's not that bad, but it can be worse than other parts of the country.",
  'But we have to go with a bit of care.'],
 ['Cyclone Gabrielle has broken through New Zealand!',
  "It's a disaster!",
  "I hope it won't be so bad.",
  "But it's really terrible!"],
 ['file_photo>',
  "What's this?",
  'Cyclone Gabrielle!',
  'Where are you?',
  'New Zealand, I think',
  'And what happened?',
  "It's very bad. The roads are so muddy",
  "Why doesn't it rain?",
  'We have to go through these terrible floods',
  "So we're going to be forced to stay in the city"],
 ['How is the weather today?',
  "It's ok, but it's not good",
  'What happened?',
  'Cyclone Gabrielle has spread across New Zealand',
  "I hope it won't be too bad",
  'The worst is already behind us',
  'We need to take care of ourselves']]
```

</br>

```python
from summary_reverse_pred_native import *

zh_context = '长宗我部本队坚守长濑川，当藤堂军接近时，长宗我部队立即命令部队以铁炮射击[47]，然后进入混战，然后长宗我部进行突击，藤堂高刑、藤堂氏胜和桑名一孝战死[46]，藤堂队先锋被消灭，长宗我部三军进行挟击本队'

df = sample_pred_wrapper(zh_context, i2c_obj = ofa_obj)
df["dialogue"].values.tolist()
```

will output:
```json
[['杰克:长宗我部本队,你们在哪里?',
  '安娜:我们坚守长濑川,当藤堂军接近时,他们立即命令部队以铁炮射击[47],然后进入混战,进行突击,藤堂高刑、氏胜和桑名一孝战死。',
  '杰克:哦,天哪,这太可怕了!',
  '安娜:还有人在战斗中被切断了线。 长宗我部三军进行挟击本队'],
 ['杰克:长宗我部本队在哪里?',
  '安娜:在长濑川,当藤堂军接近时,我的部队立即命令他们以铁炮射击[47]。然后进入混战,进行突击',
  '杰克:这是什么? 长宗我部:藤堂高刑、氏胜和桑名一孝战死。',
  '安娜:大屠杀是什么时候开始的? 长宗我部三军正在进行挟击本队。'],
 ['杰克:长宗我部三军围攻本队',
  '安娜:为什么?',
  '杰克:我们坚守长濑川,当藤堂军接近时,他们立即命令部队以铁炮射击[47]。然后进入混战,进行突击,藤堂高刑、氏胜和桑名一孝战死',
  '安娜:哦,天哪,这很糟糕!',
  '杰克:大屠杀之后,藤堂队长被解散了。 长宗我部本队:这就是为什么你一直躲在长濑川'],
 ['杰克:长宗我部本队坚守长濑川,当藤堂军接近时,他们将立即命令部队以铁炮射击[47]。然后进入混战,再进行突击,藤堂高刑、氏胜和桑名一孝战死',
  '安娜:长宗我部三军进行挟击本队'],
 ['杰克:你在哪里? 长宗我部三军进行挟击本队',
  '安娜:为什么?',
  '杰克:是的,我们坚守长濑川,当藤堂军接近时,他们立即命令部队以铁炮射击[47]。然后进入混战',
  '安娜:这是哪里?',
  '杰克:藤堂高刑、氏胜和桑名一孝在战斗中死亡。',
  '安娜:这很令人震惊。',
  '杰克:这就是为什么我们的队长被解散原因。',
  '安娜:那我们就等着吧。']]
```

#### Image Context Dialogue Generator

With the help of some excellent Image Caption models. (i.e.
[nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/nlpconnect/vit-gpt2-image-captioning ) in English and [YeungNLP/ofa-cn-base-muge-v2](https://huggingface.co/YeungNLP/ofa-cn-base-muge-v2) in Chinese
   ), we can extend the context modal from text to image.

* 1 English Dialogue Generator in Image 🦅:

<div><img src='pic/black_man.jpeg' width="550" height="450" /></div>

</br>


```python
from summary_reverse_pred_eng_native import *

img_path = "pic/black_man.jpeg"

df = sample_pred_wrapper(img_path, i2c_obj = i2c_obj)
df["dialogue"].values.tolist()
```

will output:
```json
[['file_photo>', 'A man in black and white', 'Ok'],
 ['Man in a black and white photo', 'Good!', "I'm watching it now"],
 ['file_photo>', 'what is that?', 'man in black and white', 'ok'],
 ['file_photo>', 'a man in a black and white photo', "oh, that's what I saw"],
 ['file_photo>', 'This man in black and white', "He's the best!", 'I know.']]
```

</br>

* 2 Chinese Dialogue Generator in Image 🐰:

<div><img src='pic/cat.jpg' width="550" height="450" /></div>

</br>


```python
from summary_reverse_pred_native import *

img_path = "pic/cat.jpg"

df = sample_pred_wrapper(img_path, i2c_obj = ofa_obj)
df["dialogue"].values.tolist()
```

will output:
```json
[['杰克:你见过可爱的猫咪吗?', '安娜:是的,我见过。', '杰克:太可爱了!'],
 ['杰克:你见过可爱的猫吗?', '安娜:是的,我见过。', '杰克:<file_gif>'],
 ['杰克:嘿,宝贝,你见过可爱的猫咪吗?', '安娜:是的,我见过。', '杰克:哦,对了!', '安娜:<file_gif>'],
 ['杰克:<file_photo>。', '安娜:可爱的猫咪,你见过吗?', '杰克:是的,我非常喜欢它!'],
 ['杰克:你见过可爱的猫咪吗?', '安娜:是的,它很可爱。', '杰克:我看了它们的照片,他们看起来非常可爱。', '安娜:谢!']]
```

</br>

</br>
Set "extend_by_diffusion" parameter to True will use prompt extend model that will make the caption more vivid in stable diffusion prompt format. </br>

Some details can be seen in 

[svjack/Stable-Diffusion-Chinese-Extend](https://github.com/svjack/Stable-Diffusion-Chinese-Extend)

</br>
</br>
Below is the the demonstration.

```python
from summary_reverse_pred_eng_native import *

img_path = "pic/black_man.jpeg"

df = sample_pred_wrapper(img_path, i2c_obj = i2c_obj, extend_by_diffusion = True)
df["dialogue"].values.tolist()
```

```json
[['what is this man wearing?',
  'black and white photo by robert mapplethorpe and paolo roversi',
  'file_photo>',
  'hahaha, the color is so vivid that it’s hard to see how dark it really is',
  'yeah, there’s a lot of detail in this picture',
  'I haven’t seen him for ages',
  'just look at his eyes',
  'he looks like he’s having a good day'],
 ['file_photo>',
  "what's that?",
  'a man in a black and white photo style, a black and white photo by ryan church',
  'trending on shutterstock',
  'neo-periptivism',
  'movie still, movie poster, poster art national anthem 3',
  'concert poster, poster art toyism'],
 ['Look at this',
  'file_photo>',
  "What's that?",
  'Lee Jeffries outdoor, 35mm Pentax studio lighting, studio lighting, highly detailed, sharp focus, masterpiece, concept art, trending on artstation, head and shoulders shot, rule of thirds',
  'Wow!'],
 ['file_photo>',
  'What is that?',
  "You're talking about the black and white photo from 1 9 6 8's.",
  'Hahah, how d live in television ransacked at a computer in a basement.',
  "I don't know how to do it.",
  "It's not like you can do anything"],
 ['file_photo>',
  "OMG, that's amazing! I can't believe it's black and white. The colors are so close to the ground...",
  'Yeah, they look really cool in this photo. And what do you think about this?',
  "Well, there's some ultrafine detail, very chiaroscuro lighting, private press, association press photo # film, movie still Proviablize, concert poster, concert poster for the band “Back To The Shielding”",
  'Oh, yeah, nice!']]
```

</br>

```python
from summary_reverse_pred_native import *

img_path = "pic/cat.jpg"

df = sample_pred_wrapper(img_path, i2c_obj = ofa_obj, extend_by_diffusion = True)
df["dialogue"].values.tolist()
```

</br>

will output:
```json
[['杰克:嘿,宝贝们,你们看到可爱的猫咪了吗?',
  '安娜:是的,它太可爱了!',
  '杰克:你看过《猫的艺术》吗?',
  '安娜:是的,它非常精彩。',
  '杰克:这是最精彩的部分。 4k. 高分辨率',
  '安娜:那是什么?',
  '杰克:它们在不同的时间有颜色和类型。',
  '安娜:他们看起来真不错。',
  '杰克:这真是太完美了。',
  '安娜:<file_photo>。'],
 ['杰克:你看到可爱的猫咪了吗?',
  '安娜:是的,我看到了。',
  '杰克:她穿着黑色的衣服,站在森林里。',
  '安娜:有一头白色的长发吗?',
  '杰克:<file_photo>。'],
 ['杰克:你看到可爱的猫咪了吗?',
  '安娜:是的,我看到了。',
  '杰克:<file_photo>。',
  '安娜:它是彩色铅笔艺术吗?',
  '杰克:是的,它太可爱了!',
  '安娜:哦,我喜欢它!。',
  '杰克:真的吗?',
  '安娜:是的,它非常有趣!',
  '杰克:这很完美!',
  '安娜:它们很有趣!',
  '杰克:它们的尺寸真的很大,大约3.5厘米。',
  '安娜:看起来真不错!',
  '杰克:你能想象吗?',
  '安娜:我想这是最昂贵的。',
  '杰克:嗯,不完全是。',
  '安娜:谢你,宝贝们!',
  '杰克:在artstation上也很流行。'],
 ['杰克:你见过可爱的猫咪吗?',
  '安娜:哦,我见过。工作室灯火通红,灰色背景,单一身体,没有阴影,混合器,在artstation上趋势,高度详细)彩色',
  '杰克:你喜欢它们吗?',
  '安娜:不,它看起来像一只狗,但是如此可爱。',
  '杰克:是的,我喜欢它们。'],
 ['杰克:嘿,你看到可爱的猫咪了吗?',
  '安娜:是的,我看到了。',
  '杰克:你的数字艺术是什么?',
  '安娜:<file_photo>。',
  '杰克:它看起来真不错!',
  '安娜:它是如此的超现实主义。',
  '杰克:高细节、。',
  '安娜:哇哦!这真是太神奇了',
  '杰克:我们一直在谈论这个系列。',
  '安娜:我想把它变成一个非常有趣的东西。',
  '杰克:但它似乎很有趣。',
  '安娜:但它们很性感。',
  '杰克:就像真正的猫吗?',
  '安娜:是的。',
  '杰克:它的主题是什么?',
  '安娜:文艺复兴风格。',
  '杰克:典型的现代主义。',
  '安娜:像什么?']]
```

## More Info and Disscussion
Set "do_sample" to True can be a simple method to do augmentation on the outputs and prevent from get trivial outputs. One can do this multi times and find some good samples by setting simple rules.

### Self trained other Related Model demonstration 
|Name |HuggingFace Model link| Task | Language | Model skeleton |
|---------|--------|-------|-------|-------|
| svjack/dialogue-summary | https://huggingface.co/svjack/dialogue-summary | Generate summary of a dialogue context | Chinese | T5 |
| svjack/dialogue-summary-fill-characters | https://huggingface.co/svjack/dialogue-summary-fill-characters | Map dialogue character to summary position | Chinese | T5 |
| svjack/vit-gpt-diffusion-zh | https://huggingface.co/svjack/vit-gpt-diffusion-zh | Generate stable diffusion style caption of Image | Chinese | VIT X GPT2 |

<!-- CONTACT -->
## Contact

<!--
Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com
-->
svjack - https://huggingface.co/svjack - svjackbt@gmail.com - ehangzhou@outlook.com

<!--
Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)
-->
Project Link:[https://github.com/svjack/Context2Dialogue](https://github.com/svjack/Context2Dialogue)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
<!--
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)
* [Animate.css](https://daneden.github.io/animate.css)
* [Loaders.css](https://connoratherton.com/loaders)
* [Slick Carousel](https://kenwheeler.github.io/slick)
* [Smooth Scroll](https://github.com/cferdinandi/smooth-scroll)
* [Sticky Kit](http://leafo.net/sticky-kit)
* [JVectorMap](http://jvectormap.com)
* [Font Awesome](https://fontawesome.com)
* [Bigscience](https://bigscience.huggingface.co)
* [TextBox](https://github.com/RUCAIBox/TextBox)
* [Langboat](https://huggingface.co/Langboat)
* [uer](https://huggingface.co/uer)
-->
* [deepset-wikipedia-assistant](https://huggingface.co/spaces/deepset/wikipedia-assistant)
* [ClueAI](https://huggingface.co/ClueAI)
* [nlpconnect/vit-gpt2-image-captioning](https://huggingface.co/svjack/prompt-extend-chinese-gpt)
* [YeungNLP/ofa-cn-base-muge-v2](https://huggingface.co/YeungNLP/ofa-cn-base-muge-v2)
* [daspartho/prompt-extend](https://huggingface.co/daspartho/prompt-extend )
* [svjack/Daliy-Dialogue](https://github.com/svjack/Daliy-Dialogue)
* [svjack/GLM-Open-Dialogue](https://github.com/svjack/GLM-Open-Dialogue)
* [svjack/docvqa-gen](https://github.com/svjack/docvqa-gen)
* [svjack/Stable-Diffusion-Chinese-Extend](https://github.com/svjack/Stable-Diffusion-Chinese-Extend)
* [svjack](https://huggingface.co/svjack)

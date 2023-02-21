#### Chinese scope
device = "cuda:0"
#device = "cpu"
assert device.startswith("cpu") or device.startswith("cuda")

import sys
from predict import *

from transformers import (
    T5ForConditionalGeneration,
    MT5ForConditionalGeneration,
    ByT5Tokenizer,
    PreTrainedTokenizer,
    T5TokenizerFast as T5Tokenizer,
    MT5TokenizerFast as MT5Tokenizer,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BertTokenizer,
    GPT2LMHeadModel,
)

import pandas as pd
import numpy as np
import re
from rapidfuzz import fuzz
from tqdm import tqdm
import numpy as np
import os

import jieba
def repeat_to_one_f(x):
    req = None
    for token in jieba.lcut(x):
        #print("req :", req)

        if len(set(token)) == 1:
            token = token[0]
        if req is None:
            req = token
        else:

            if (token in req and token not in [',', '，', '、', ' ']) or (req and token in [',', '，', '、', ' '] and req[-1] in [',', '，', '、', ' ']):
                continue
            else:
                while req.endswith(token[0]):
                    token = token[1:]
                req = req + token
    if req is None:
        return ""
    return req.strip()

def shorten_exists(l, sim_threshold = 80, slice_size = 5):
    req = []
    for ele in l:
        if not req:
            req.append(ele)
        else:
            if max(map(lambda x: fuzz.ratio(x[:slice_size], ele[:slice_size]), req)) < sim_threshold:
                req.append(ele)
    return req

model_path = "svjack/summary-dialogue"
tokenizer0 = T5Tokenizer.from_pretrained(model_path)
model0 = T5ForConditionalGeneration.from_pretrained(model_path)

if device.startswith("cuda"):
    model = Obj(model0, tokenizer0, device = "cuda:0")
else:
    model = Obj(model0, tokenizer0, device = "cpu")

def loop_add(l, names = ["杰克", "安娜"]):
    req = []
    for i in range(len(l)):
        ii = int(i % len(names))
        req.append(
            "{}:{}".format(names[ii], l[i])
        )
    return req

#### need some names drop in context(may not have ":")
#### '艾米-亚当斯在《沉睡的空洞》中，全身，双色大眼睛，咬牙切齿，恐怖，复杂的细节，电影，史诗，现实，解剖，汤姆-哈努卡，上光，艺术站，逼真，可怕'
def guess_name_candidates(context, cnt_threshold = 1):
    from copy import deepcopy
    assert type(context) == type("")
    import re
    l = re.findall(r"[\u4e00-\u9fa5a-zA-Z]+:", context)
    l = list(filter(lambda x: x.strip(), l))
    ori_l = deepcopy(l)
    if not l:
        return []
    s = pd.Series(l).value_counts()
    l = pd.Series(s[s > cnt_threshold].index.values.tolist()).map(lambda x: x[:-1]).values.tolist()
    for ele in ori_l:
        if len(ele[:-1]) not in l and (len(ele[:-1]) <= 3 or (
            sum(map(len ,re.findall(r"[a-zA-Z]+:", ele))) == len(ele)
        )):
            l.append(ele[:-1])
    l = list(set(l))
    return l

def simple_pred(summary, candidates = ["杰克", "安娜"],
    shorten_it = False, do_sample = True):
    pred_text = model.predict(
    "摘要：{} 候选集：{}".format(summary, " ".join(candidates)),
    do_sample = do_sample
    )[0]
    candidates_ = guess_name_candidates(pred_text)
    l = re.split("{}".format("|".join(map(lambda x: "{}:".format(x), candidates_))) ,pred_text)
    l = list(filter(lambda x: x.strip(), l))
    if shorten_it:
        l = shorten_exists(l)
    l = list(map(repeat_to_one_f, l))
    l = loop_add(l, candidates)
    return l

def percentile_sort(df, perc_num = 101):
    score_tuple_s = df["score_tuple"]
    score_array = np.asarray(score_tuple_s.values.tolist())
    perc_list = np.linspace(0, 100, perc_num).tolist()
    low_to_high_perc_array = np.stack(list(map(lambda p: np.percentile(score_array, p, axis = 0), perc_list)))

    def get_rank(array_):
        lookup_list = pd.DataFrame(array_ - low_to_high_perc_array[::-1]).apply(lambda s: min(s) >= 0, axis = 1).tolist()
        if True not in lookup_list:
            return len(lookup_list)
        return lookup_list.index(True)

    rank_list = []
    for i in range(score_array.shape[0]):
        rank_list.append(get_rank(score_array[i, :]))

    rank_s = pd.Series(rank_list)
    return df.iloc[np.argsort(rank_s.values)]

def repeat_score(l, slice_size = 200 ,sim_threshold = 70):
    from copy import deepcopy
    assert type(l) == type([])
    l = deepcopy(l)
    l = sorted(l)
    cnt_num = 0
    set0 = set([])
    for ele in l:
        if ":" in ele:
            ele = "".join(ele.split(":")[1:])
        if set0 and max(map(lambda x: fuzz.ratio(x[:slice_size], ele[:slice_size]), set0)) > sim_threshold:
            #if ele in set0:
            cnt_num += 1
        set0.add(ele)
    return cnt_num

#### "svjack/prompt-extend-chinese-gpt"
#model_path = "/home/featurize/zh_p_extend_outputs/simplet5-epoch-3-train-loss-1.2628-val-loss-1.6293"
model_path = "svjack/prompt-extend-chinese-gpt"
tokenizer1 = BertTokenizer.from_pretrained(model_path)
model1 = GPT2LMHeadModel.from_pretrained(model_path)

if device.startswith("cuda"):
    zh_pe_model = Obj(model1, tokenizer1, device = "cuda:0")
else:
    zh_pe_model = Obj(model1, tokenizer1, device = "cpu")

def one_ele_trans(x):
    x = x.strip()
    x = x[1:] if x.startswith("'") else x
    x = x[:-1] if x.endswith("'") else x
    x = x[1:] if x.startswith('"') else x
    x = x[:-1] if x.endswith('"') else x
    return x

def stdf_prompt_expander(x):
    assert type(x) == type("")
    return zh_pe_model.predict(
    one_ele_trans(x.strip()).strip(),
    max_length = 128
    )[0].replace(" ", "").strip()

def sample_pred(context, times = 5, stdf_prompt_expander = lambda _: _):
    df_req = []
    for i in tqdm(range(times)):
        ele = stdf_prompt_expander(context)
        #ele = context
        l = simple_pred(ele, do_sample = True)
        df_req.append(
            [ele, l]
        )
    df = pd.DataFrame(df_req)
    df.columns = ["context", "dialogue"]
    df["fuzz"] = df["dialogue"].map(
        lambda x: fuzz.ratio(context, " ".join(x))
    )
    df["max_fuzz"] = df["dialogue"].map(
        lambda x: max(map(lambda y: fuzz.ratio(y, context), x))
    )
    df["length"] = df["dialogue"].map(len)
    df["rpt_score"] = df["dialogue"].map(repeat_score)
    df["score_tuple"] = df.apply(
        lambda x: (x["fuzz"], -1 * x["max_fuzz"], x["length"], -1 * x["rpt_score"]), axis = 1
    )
    df = percentile_sort(df)
    return df

def sample_pred_wrapper(context, i2c_obj, times = 5, extend_by_diffusion = False):
    assert type(context) == type("")
    if any(map(lambda x: context.endswith(x), [".jpg", ".png", ".jpeg"])):
        img_path = context
        i2c_df = i2c_obj.predict_to_df([img_path])
        assert i2c_df.size > 0
        context = i2c_df["caption"].iloc[0]
    else:
        pass
    assert type(context) == type("")
    if extend_by_diffusion:
        req_df = sample_pred(context, times = times, stdf_prompt_expander = stdf_prompt_expander)
    else:
        req_df = sample_pred(context, times = times, stdf_prompt_expander = lambda _:_)
    return req_df

from ofa import *
ofa_obj = OFA()

if __name__ == "__main__":
    '''
    from image2caption import *
    i2c_tiny_zh_obj = Image2Caption("svjack/vit-gpt-diffusion-zh",
        overwrite_encoder_checkpoint_path = "google/vit-base-patch16-224",
        overwrite_token_model_path = "IDEA-CCNL/Wenzhong-GPT2-110M",
        device = device
    )
    '''
    from ofa import *
    ofa_obj = OFA()

    img_path = "../pic/bug.jpg"
    img_path = "../pic/baobao.jpeg"
    img_path = "../pic/cat0.jpg"
    img_path = "../pic/cat.jpg"
    os.path.exists(img_path)

    df = sample_pred_wrapper(img_path, i2c_obj = ofa_obj)
    df["dialogue"].values.tolist()

    img_url = "https://datasets-server.huggingface.co/assets/metashift/--/metashift/train/2/image/image.jpg"
    img_url = "https://datasets-server.huggingface.co/assets/metashift/--/metashift/train/6/image/image.jpg"

    #### diffusion model, general model
    df = sample_pred_wrapper(img_url, i2c_obj = ofa_obj)
    df["dialogue"].values.tolist()

    ds_en_zh_df = pd.read_csv("../ds_en_zh_df.csv")

    idx = 3
    ds_en_zh_df.iloc[:, -1].iloc[idx]

    df = sample_pred(ds_en_zh_df.iloc[:, -1].iloc[idx])
    df["dialogue"].values.tolist()

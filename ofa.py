#### https://github.com/yangjianxin1/OFA-Chinese

from component.ofa.modeling_ofa import OFAModelForCaption
from torchvision import transforms
from PIL import Image
from transformers import BertTokenizerFast
import torch
import pathlib
import pandas as pd
import numpy as np
from IPython.core.display import HTML
import os
import requests

# 定义图片预处理逻辑
mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
resolution = 256
patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

class OFA(object):
    def __init__(self ,model_path = 'YeungNLP/ofa-cn-base-muge-v2',
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        self.device = device
        self.model = OFAModelForCaption.from_pretrained(model_path)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_path)
        self.model = self.model.to(self.device)

    def predict_to_df(self, image_paths):
        img_caption_pred = self.predict_step(image_paths)
        img_cation_df = pd.DataFrame(list(zip(image_paths, img_caption_pred)))
        img_cation_df.columns = ["img", "caption"]
        return img_cation_df
        #img_cation_df.to_html(escape=False, formatters=dict(Country=path_to_image_html))

    def predict_step(self ,image_paths):
        images = []
        for image_path in image_paths:
            #i_image = Image.open(image_path)
            if image_path.startswith("http"):
                i_image = Image.open(
                    requests.get(image_path, stream=True).raw
                    )
            else:
                i_image = Image.open(image_path)

            if i_image.mode != "RGB":
                i_image = i_image.convert(mode="RGB")
            patch_img = patch_resize_transform(i_image).unsqueeze(0)
            images.append(patch_img)

        txt = '图片描述了什么?'
        inputs = self.tokenizer([txt], return_tensors="pt").input_ids
        inputs = inputs.to(self.device)
        req = []
        for patch_img in images:
            # 生成caption
            patch_img = patch_img.to(self.device)
            gen = self.model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
            gen = self.tokenizer.batch_decode(gen, skip_special_tokens=True)[0]
            gen = gen.replace(" ", "").strip()
            req.append(gen)
        return req

def path_to_image_html(path):
    return '<img src="'+ path + '" width="60" >'

if __name__ == "__main__":
    #### build too slow
    ofa_obj = OFA()

    img_path_l = pd.Series(list(pathlib.Path("../../pic").rglob("*"))).map(
        lambda x: x.__fspath__()
    ).map(str).map(lambda x: np.nan if "._" in x else x).dropna().values.tolist()
    img_path_l

    img_caption_ofa_df = ofa_obj.predict_to_df(img_path_l)

    HTML(img_caption_ofa_df.to_html(escape=False, formatters=dict(img=path_to_image_html)))

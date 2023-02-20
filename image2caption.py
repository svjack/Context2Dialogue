##### image pred
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import pathlib
import pandas as pd
import numpy as np
from IPython.core.display import HTML
import os
import requests

class Image2Caption(object):
    def __init__(self ,model_path = "nlpconnect/vit-gpt2-image-captioning",
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                overwrite_encoder_checkpoint_path = None,
                overwrite_token_model_path = None
    ):
        assert type(overwrite_token_model_path) == type("") or overwrite_token_model_path is None
        assert type(overwrite_encoder_checkpoint_path) == type("") or overwrite_encoder_checkpoint_path is None
        if overwrite_token_model_path is None:
            overwrite_token_model_path = model_path
        if overwrite_encoder_checkpoint_path is None:
            overwrite_encoder_checkpoint_path = model_path
        self.device = device
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(overwrite_encoder_checkpoint_path)
        self.tokenizer = AutoTokenizer.from_pretrained(overwrite_token_model_path)
        self.model = self.model.to(self.device)

    def predict_to_df(self, image_paths):
        img_caption_pred = self.predict_step(image_paths)
        img_cation_df = pd.DataFrame(list(zip(image_paths, img_caption_pred)))
        img_cation_df.columns = ["img", "caption"]
        return img_cation_df
        #img_cation_df.to_html(escape=False, formatters=dict(Country=path_to_image_html))

    def predict_step(self ,image_paths, max_length = 128, num_beams = 4):
        gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
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
            images.append(i_image)

        pixel_values = self.feature_extractor(images=images, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(self.device)

        output_ids = self.model.generate(pixel_values, **gen_kwargs)

        preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        preds = [pred.strip() for pred in preds]
        return preds

def path_to_image_html(path):
    return '<img src="'+ path + '" width="60" >'

if __name__ == "__main__":
    i2c_obj = Image2Caption()
    i2c_tiny_zh_obj = Image2Caption("svjack/vit-gpt-diffusion-zh",
        overwrite_encoder_checkpoint_path = "google/vit-base-patch16-224",
        overwrite_token_model_path = "IDEA-CCNL/Wenzhong-GPT2-110M"
    )

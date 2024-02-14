from IPython.display import clear_output, Image, display
import PIL.Image
import io
import json
import torch
import numpy as np
from module.processing_image import Preprocess
from module.visualizing_image import SingleImageViz
from module.modeling_frcnn import GeneralizedRCNN
from module.utils import Config
import module.utils as utils
from transformers import LxmertForQuestionAnswering, LxmertTokenizer
import wget
import pickle
import os



# URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
OBJ_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/objects_vocab.txt"
ATTR_URL = "https://raw.githubusercontent.com/airsplay/py-bottom-up-attention/master/demo/data/genome/1600-400-20/attributes_vocab.txt"
GQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/gqa/trainval_label2ans.json"
VQA_URL = "https://raw.githubusercontent.com/airsplay/lxmert/master/data/vqa/trainval_label2ans.json"


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = io.BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    display(Image(data=f.getvalue()))

objids = utils.get_data(OBJ_URL)
attrids = utils.get_data(ATTR_URL)
gqa_answers = utils.get_data(GQA_URL)
vqa_answers = utils.get_data(VQA_URL)

frcnn_cfg = Config.from_pretrained("unc-nlp/frcnn-vg-finetuned")
frcnn = GeneralizedRCNN.from_pretrained("unc-nlp/frcnn-vg-finetuned", config=frcnn_cfg)
image_preprocess = Preprocess(frcnn_cfg)
lxmert_tokenizer = LxmertTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
lxmert_gqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-gqa-uncased")
lxmert_vqa = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-vqa-uncased")



# image viz
# frcnn_visualizer = SingleImageViz(URL, id2obj=objids, id2attr=attrids)
# run frcnn

# add boxes and labels to the image

# frcnn_visualizer.draw_boxes(
#     output_dict.get("boxes"),
#     output_dict.pop("obj_ids"),
#     output_dict.pop("obj_probs"),
#     output_dict.pop("attr_ids"),
#     output_dict.pop("attr_probs"),
# )
# showarray(frcnn_visualizer._get_buffer())



test_questions_for_url1 = [
    "Where is this scene?",
    "what is the man riding?",
    "What is the man wearing?",
    "What is the color of the horse?",
]
# test_questions_for_url2 = [
#     "Where is the cat?",
#     "What is near the disk?",
#     "What is the color of the table?",
#     "What is the color of the cat?",
#     "What is the shape of the monitor?",
# ]

test_questions_for_url2 = [
    "Describe the image",
    "What is the color of horse?",
    "How far is the horse?",
    "What are the objects in the image?",
    "who is riding the horse?",
    "is there any dolphin in the image?",
    "There is a umbrella",
    "Is there any umbrella?",
    "what is color of the dolphin?"
]

# Very important that the boxes are normalized


# for test_question in test_questions_for_url2:
def get_result(img_path, question):
    URL = img_path
    images, sizes, scales_yx = image_preprocess(URL)
    output_dict = frcnn(
        images,
        sizes,
        scales_yx=scales_yx,
        padding="max_detections",
        max_detections=frcnn_cfg.max_detections,
        return_tensors="pt",
    )
    normalized_boxes = output_dict.get("normalized_boxes")
    features = output_dict.get("roi_features")
    # run lxmert
    test_question = [question]

    inputs = lxmert_tokenizer(
        test_question,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_token_type_ids=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt",
    )

    # run lxmert(s)
    output_gqa = lxmert_gqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    output_vqa = lxmert_vqa(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        visual_feats=features,
        visual_pos=normalized_boxes,
        token_type_ids=inputs.token_type_ids,
        output_attentions=False,
    )
    # get prediction
    pred_vqa = output_vqa["question_answering_score"].argmax(-1)
    pred_gqa = output_gqa["question_answering_score"].argmax(-1)
    print("Question:", test_question)
    print("prediction from LXMERT GQA:", gqa_answers[pred_gqa])
    print("prediction from LXMERT VQA:", vqa_answers[pred_vqa])
    return gqa_answers[pred_gqa]
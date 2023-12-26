
from transformers import DPTFeatureExtractor, DPTForDepthEstimation
from PIL import Image
import torch
import numpy as np
import requests

def formatter_depth1(formatted):
  return formatted
def formatter_depth(formatted):
  form = []
  for ind,i in enumerate(formatted):
    temp=[]
    for indj,j in enumerate(i):
      if(j>200):
        temp.append(255)
      elif(j>150):
        temp.append(150)
      elif(j>100):
        temp.append(100)
      elif(j>50):
        temp.append(50)
      else:
        temp.append(0)
    form.append(temp)
  return form

feature_extractor = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")
model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large")

def depth_model(url):
  image = Image.open(url)
  pixel_values = feature_extractor(image, return_tensors="pt").pixel_values
  with torch.no_grad():
    outputs = model(pixel_values)
    predicted_depth = outputs.predicted_depth

  prediction = torch.nn.functional.interpolate(
                      predicted_depth.unsqueeze(1),
                      size=image.size[::-1],
                      mode="bicubic",
                      align_corners=False,
                ).squeeze()
  output = prediction.cpu().numpy()
  formatted = (output * 255 / np.max(output)).astype('uint8')
  form = formatter_depth1(formatted)
  form = np.array(form).astype('uint8')
  depth = Image.fromarray(form)
  return form
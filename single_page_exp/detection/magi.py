from transformers import AutoModel
import numpy as np
from PIL import Image
import torch
import os

images = [
        "/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images/1afa34a7/000.jpg",
        "/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images/1afa34a7/003.jpg",
        "/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images/1afa34a7/005.jpg",
        "/home-local/mserrao/PSSComics/multimodal-comic-pss/datasets.unify/DCM/images/1afa34a7/026.jpg"
    ]
out_dir = "/home-local/mserrao/PSSComics/multimodal-comic-pss/Statistics/magi_out"

def read_image_as_np_array(image_path):
    with open(image_path, "rb") as file:
        image = Image.open(file).convert("L").convert("RGB")
        image = np.array(image)
    return image

images = [read_image_as_np_array(image) for image in images]

model = AutoModel.from_pretrained("ragavsachdeva/magi", trust_remote_code=True).cuda()
with torch.no_grad():
    results = model.predict_detections_and_associations(images)
    text_bboxes_for_all_images = [x["texts"] for x in results]

for i in range(len(images)):
    model.visualise_single_image_prediction(images[i], results[i], filename=f"{out_dir}/image_{i}.png")

import sys
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

from src.model import OrigSegformerForSemanticSegmentation, get_model


def get_images(samples_dir):
    samples_dir = Path(samples_dir)

    def _iter_pil_images():
        sample_images = sorted(list(samples_dir.glob("*.jpg")))
        for p in sample_images:
            image = Image.open(p)
            image = image.resize((224, 224))
            yield image

    return tuple(_iter_pil_images())


images = get_images("data/samples")


def draw_dot(ax, point):
    print(point)
    x, y = tuple(point)
    c = patches.Circle(xy=(x, y), radius=4, color="red")
    ax.add_patch(c)

def get_model():
    model_dir = "models/custom_segsem_08/"
    model = OrigSegformerForSemanticSegmentation.from_pretrained(model_dir)
    model.eval()

    feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)
    inputs = feature_extractor_inference(images=images, return_tensors="pt")

def infer():
    with torch.no_grad():
        _, points = model(**inputs)
        points = points.numpy()
        points = points * 112 + 112
        print(points)

    fig, axes = plt.subplots(2, 1, figsize=(7, 8))
    for i, (ax, image) in enumerate(zip(axes, images)):
        ax.imshow(image)
        # ax.imshow(masks.numpy().transpose(1, 2, 0))
        point = points[i]
        # print(p)
        draw_dot(ax, point[:2])
        draw_dot(ax, point[2:])

    plt.show()


def main():
    pass


if __name__ == "__main__":
    main()


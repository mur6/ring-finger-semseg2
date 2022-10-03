import json
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from skimage import measure
from torch import nn
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import transforms as transforms
from torchvision.transforms import Resize, ToTensor
from transformers import SegformerDecodeHead, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerMLP, SegformerModel

from src.dataset import RingFingerDataset
from src.model import get_model

base_data_dir = Path("../blender-for-finger-segmentation/data2/")


feature_extractor_inference = SegformerFeatureExtractor(do_random_crop=False, do_pad=False)


train_dataset = RingFingerDataset(
    base_data_dir / "training",
    "data/datasets/contour_checked_numbers_training.json",
    feature_extractor=feature_extractor_inference,
    transform=None,
)
valid_dataset = RingFingerDataset(
    base_data_dir / "validation",
    "data/datasets/contour_checked_numbers_validation.json",
    feature_extractor=feature_extractor_inference,
    transform=None,
)


from torch import nn
from torch.utils.data import DataLoader

train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=8)
model = get_model()

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AdamW

criterion = nn.MSELoss()
optimizer = AdamW(model.parameters(), lr=0.00006)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Model Initialized!")

# pbar = tqdm(train_dataloader)

for epoch in range(1, 1 + 1):
    print("Epoch:", epoch)
    pbar = tqdm(train_dataloader)
    accuracies = []
    losses = []
    val_accuracies = []
    val_losses = []
    model.train()
    train_loss = 0.0
    for idx, batch in enumerate(pbar):
        pixel_values = batch["pixel_values"].to(device)
        points = batch["points"].to(device)
        print(pixel_values.shape, points.shape)
        optimizer.zero_grad()

        outputs = model(pixel_values=pixel_values)
        print("outputs.shape", outputs.shape)
        print("outputs.dtype", outputs.dtype)

        # evaluate
        points = ((points - (224 / 2.0)) / 112.0).float()

        loss = criterion(outputs, points)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    else:
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for idx, batch in enumerate(valid_dataloader):
                pixel_values = batch["pixel_values"].to(device)
                points = batch["points"].to(device)
                points = (points - (224 / 2.0)) / 112.0
                outputs = model(pixel_values=pixel_values)
                loss = criterion(outputs, points)
                val_loss += loss.item()

    train_count = len(train_dataloader)
    val_count = len(valid_dataloader)
    s1 = f"Training: Mean Squared Error: {train_loss/train_count}"
    s2 = f"Validation: Mean Squared Error: {val_loss/val_count}"
    print(s1 + " " + s2)

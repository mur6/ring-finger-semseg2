import json
import math
from pathlib import Path
from typing import Optional, Tuple, Union

import albumentations as A
import hydra
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from skimage import measure
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision import transforms as transforms
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm
from transformers import AdamW, SegformerDecodeHead, SegformerFeatureExtractor, SegformerForSemanticSegmentation
from transformers.models.segformer.modeling_segformer import SegformerMLP, SegformerModel

from src.dataset import RingFingerDataset
from src.loader import make_dataloaders
from src.model import get_model

# base_data_dir = Path("../blender-for-finger-segmentation/data2/")
# pbar = tqdm(train_dataloader)


def train(*, max_epoch, train_dataloader, valid_dataloader):
    model = get_model()
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=0.00006)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("Model Initialized!")

    for epoch in range(1, max_epoch + 1):
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


@hydra.main(version_base=None, config_name="config/local")
def main(cfg):
    max_epoch = int(cfg.config.max_epoch)
    base_data_dir = Path(cfg.config.base_data_dir)
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = make_dataloaders(base_data_dir)
    train(
        max_epoch=max_epoch,
    )


if __name__ == "__main__":
    main()

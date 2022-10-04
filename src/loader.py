from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import SegformerFeatureExtractor
from transformers.models.segformer.modeling_segformer import SegformerMLP, SegformerModel

from src.dataset import RingFingerDataset


def make_dataloaders(base_data_dir, batch_size=8):
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

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
    return train_dataset, valid_dataset, train_dataloader, valid_dataloader


@hydra.main(version_base=None, config_name="config/local")
def main(cfg):
    base_data_dir = Path(cfg.config.base_data_dir)
    print(type(base_data_dir))
    train_dataset, valid_dataset, train_dataloader, valid_dataloader = make_dataloaders(base_data_dir)


if __name__ == "__main__":
    main()

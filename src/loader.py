import pathlib
from dataclasses import dataclass

import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import SegformerFeatureExtractor
from transformers.models.segformer.modeling_segformer import SegformerMLP, SegformerModel

from src.dataset import RingFingerDataset


@dataclass
class Config:
    base_data_dir: pathlib.Path


cs = ConfigStore.instance()
cs.store(name="config/local", node=Config)


def make_dataloaders(base_data_dir):
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

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=8)
    return train_dataloader, valid_dataloader


@hydra.main(version_base=None, config_name="config/local")
def main(cfg):
    print(f"Orig working directory:  {hydra.utils.get_original_cwd()}")
    print(OmegaConf.to_yaml(cfg))
    base_data_dir = cfg.config.base_data_dir
    print(type(base_data_dir))
    # make_dataloaders


if __name__ == "__main__":
    main()

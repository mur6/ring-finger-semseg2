from distutils.command.config import config
from omegaconf import DictConfig, OmegaConf
import hydra

@hydra.main(version_base=None, config_name="config/local")
def my_app(cfg):
    print(f"Orig working directory:  {hydra.utils.get_original_cwd()}")
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    my_app()

"""
Model Factory.
"""

from typing import Optional
from torch.nn import Module
from common.trainpipeline.arcvein import ArcVein
from common.trainpipeline.config import ModelConfig
from common.trainpipeline.model.custom_model import CustomModel
from common.trainpipeline.model.fine_tune_model import FineTuneModel


def get_model(
    config: ModelConfig,
    pretrained_model_path: Optional[str] = None,
    pretrained_predictor_classes: Optional[int] = None,
) -> Module:
    """
    Gives back appropriate models.
    """
    if pretrained_model_path:
        return FineTuneModel(
            config,
            pretrained_model_path,
            pretrained_predictor_classes if pretrained_predictor_classes else 301,
        )
    if config.arcvein:
        return ArcVein()
    return CustomModel(config=config)

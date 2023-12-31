"""
Factory for backbones.
"""


from common.trainpipeline.backbone.isotropic_backbone import IsotropicBackBone
from common.trainpipeline.backbone.pyramid_backbone import PyramidBackbone
from common.trainpipeline.config import BackboneConfig


def get_backbone(config: BackboneConfig):
    """
    Calls appropriate backbone build.
    """
    if config.backbone_type == "isotropic_backbone":
        return IsotropicBackBone(config.blocks)
    if config.backbone_type == "pyramid_backbone":
        return PyramidBackbone(
            config.blocks,
            requires_grad=config.requires_grad,
        )
    raise NotImplementedError("Wrong backbone type")

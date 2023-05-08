# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .dota import DOTADataset


@ROTATED_DATASETS.register_module()
class FusarshipDataset(DOTADataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('cargo','fishing','tanker' )
    PALETTE = [
        (165, 42, 42), (189, 183, 107), (0, 255, 0),
    ]

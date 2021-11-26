# Copyright (c) OpenMMLab. All rights reserved.
import copy
from collections import defaultdict

import numpy as np
import pytest
import torch
from mmcv import Config



@pytest.mark.parametrize('cfg_file', [
    './tests/data/configs_mmtrack/tracktor_faster-rcnn_r50_fpn_4e.py',
])
def test_tracktor_forward(cfg_file):
    config = Config.fromfile(cfg_file)
    model = copy.deepcopy(config.model)
    model.pretrains = None
    model.detector.pretrained = None

    from mmtrack.models import build_model
    mot = build_model(model)
    mot.eval()

    input_shape = (1, 3, 256, 256)
    mm_inputs = _demo_mm_inputs(input_shape, num_items=[10], with_track=True)
    imgs = mm_inputs.pop('imgs')
    img_metas = mm_inputs.pop('img_metas')
    with torch.no_grad():
        imgs = torch.cat([imgs, imgs.clone()], dim=0)
        img_list = [g[None, :] for g in imgs]
        img2_metas = copy.deepcopy(img_metas)
        img2_metas[0]['frame_id'] = 1
        img_metas.extend(img2_metas)
        results = defaultdict(list)
        for one_img, one_meta in zip(img_list, img_metas):
            result = mot.forward([one_img], [[one_meta]], return_loss=False)
            for k, v in result.items():
                results[k].append(v)


def _demo_mm_inputs(
        input_shape=(1, 3, 300, 300),
        num_items=None,
        num_classes=10,
        with_track=False):
    """Create a superset of inputs needed to run test or train batches.

    Args:
        input_shape (tuple):
            input batch dimensions

        num_items (None | List[int]):
            specifies the number of boxes in each batch item

        num_classes (int):
            number of different labels a box might have
    """
    from mmdet.core import BitmapMasks

    (N, C, H, W) = input_shape

    rng = np.random.RandomState(0)

    imgs = rng.rand(*input_shape)

    img_metas = [{
        'img_shape': (H, W, C),
        'ori_shape': (H, W, C),
        'pad_shape': (H, W, C),
        'filename': '<demo>.png',
        'scale_factor': 1.0,
        'flip': False,
        'frame_id': 0,
        'img_norm_cfg': {
            'mean': (128.0, 128.0, 128.0),
            'std': (10.0, 10.0, 10.0)
        }
    } for i in range(N)]

    gt_bboxes = []
    gt_labels = []
    gt_masks = []
    gt_match_indices = []

    for batch_idx in range(N):
        if num_items is None:
            num_boxes = rng.randint(1, 10)
        else:
            num_boxes = num_items[batch_idx]

        cx, cy, bw, bh = rng.rand(num_boxes, 4).T

        tl_x = ((cx * W) - (W * bw / 2)).clip(0, W)
        tl_y = ((cy * H) - (H * bh / 2)).clip(0, H)
        br_x = ((cx * W) + (W * bw / 2)).clip(0, W)
        br_y = ((cy * H) + (H * bh / 2)).clip(0, H)

        boxes = np.vstack([tl_x, tl_y, br_x, br_y]).T
        class_idxs = rng.randint(1, num_classes, size=num_boxes)

        gt_bboxes.append(torch.FloatTensor(boxes))
        gt_labels.append(torch.LongTensor(class_idxs))
        if with_track:
            gt_match_indices.append(torch.arange(boxes.shape[0]))

    mask = np.random.randint(0, 2, (len(boxes), H, W), dtype=np.uint8)
    gt_masks.append(BitmapMasks(mask, H, W))

    mm_inputs = {
        'imgs': torch.FloatTensor(imgs).requires_grad_(True),
        'img_metas': img_metas,
        'gt_bboxes': gt_bboxes,
        'gt_labels': gt_labels,
        'gt_bboxes_ignore': None,
        'gt_masks': gt_masks,
    }
    if with_track:
        mm_inputs['gt_match_indices'] = gt_match_indices
    return mm_inputs

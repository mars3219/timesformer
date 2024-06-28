from pathlib import Path
from einops import rearrange, repeat, reduce
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
# import cv2

import torch
import torchvision.transforms as transforms
from timesformer.config.defaults import get_cfg
from timesformer.models import MODEL_REGISTRY
import visualize_attn_util as vau

if __name__ == "__main__":
    model_file = '/workspace/TimeSformer/TimeSformer_divST_8_224_SSv2.pyth'
    Path(model_file).exists()

    cfg = get_cfg()
    cfg.merge_from_file('/workspace/TimeSformer/configs/SSv2/TimeSformer_divST_8_224.yaml')
    cfg.TRAIN.ENABLE = False
    cfg.TIMESFORMER.PRETRAINED_MODEL = model_file
    model = MODEL_REGISTRY.get('vit_base_patch16_224')(cfg)

    with open('/workspace/TimeSformer/example_data/labels.json') as f:
        ssv2_labels = json.load(f)
    ssv2_labels = list(ssv2_labels.keys())

    path_to_video = Path('/workspace/TimeSformer/example_data/74225/')
    path_to_video.exists()

    with torch.set_grad_enabled(False):
        np.random.seed(cfg.RNG_SEED)
        torch.manual_seed(cfg.RNG_SEED)
        model.eval()
        pred = model(vau.create_video_input(path_to_video)).cpu().detach()

    topk_scores, topk_label = torch.topk(pred, k=5, dim=-1)
    for i in range(5):
        pred_name = ssv2_labels[topk_label.squeeze()[i].item()]
        print(f"Prediction index {i}: {pred_name:<25}, score: {topk_scores.squeeze()[i].item():.3f}")


    att_roll = vau.DividedAttentionRollout(model)
    masks = att_roll(path_to_video)

    np_imgs = [vau.transform_plot(p) for p in vau.get_frames(path_to_video)]
    masks = vau.create_masks(list(rearrange(masks, 'h w t -> t h w')),np_imgs)
    stacked_image = np.hstack(np_imgs)

    cv2.imwrite('stacked_img.jpg', stacked_image)
    cv2.imwrite('stacked_mask.jpg', np.hstack(masks))
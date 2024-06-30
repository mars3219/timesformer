from pathlib import Path
import json, time
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import threading
import queue
from timesformer.config.defaults import get_cfg
from timesformer.models import MODEL_REGISTRY
import visualize_attn_util as vau
from einops import rearrange, repeat, reduce

DEFAULT_MEAN = [0.45, 0.45, 0.45]
DEFAULT_STD = [0.225, 0.225, 0.225]
# convert video path to input tensor for model
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(DEFAULT_MEAN,DEFAULT_STD),
    transforms.Resize(224),
    transforms.CenterCrop(224),
])

# convert the video path to input for cv2_imshow()
transform_plot = transforms.Compose([
    # lambda p: cv2.imread(str(p),cv2.IMREAD_COLOR),
    transforms.ToTensor(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    lambda x: rearrange(x*255, 'c h w -> h w c').numpy()
])


TOTAL_FRAMES = 0

def load_video_frames(video_path, frame_queue, num_frames=8, img_size=224):
    """
    Load video from the given path and preprocess it to the required format
    """
    cap = cv2.VideoCapture(str(video_path))
    TOTAL_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f'Total frames in the video: {TOTAL_FRAMES}')

    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, (img_size, img_size))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
       
        if frame_queue.full():
            try:
                frame_queue.get_nowait()
            except queue.Empty:
                pass

        frame_queue.put(frame)
        time.sleep(0.03)

def analyze_frames(model, frame_queue, ssv2_labels):
    att_roll = vau.DividedAttentionRollout(model)  
    with torch.set_grad_enabled(False):
        frames = []
        while True:
            try:
                frames.append(frame_queue.get())
                if len(frames) == num_frames:
                    frames_o = [transform(frame) for frame in frames]
                    frames_t = torch.stack(frames_o, dim=0)
                    frames_t = rearrange(frames_t, 't c h w -> c t h w')
                    frames_t = frames_t.unsqueeze(dim=0)  # Shape: (1, C, T, H, W)
                    model.eval()
                    pred = model((frames_t)).cpu().detach()
                    topk_scores, topk_label = torch.topk(pred, k=5, dim=-1)
                    print(f"Prediction index {topk_label[0,0]}")
                    
                    pred_name = ssv2_labels[topk_label.squeeze()[0].item()]
                    print(f"Prediction index {0}: {pred_name:<25}, score: {topk_scores.squeeze()[0].item():.3f}")

                    masks = att_roll(frames_t, source_type="video")
                    np_imgs = [transform_plot(frame) for frame in frames]
                    masks = vau.create_masks(list(rearrange(masks, 'h w t -> t h w')), np_imgs)
                    
                    cv2.imwrite('stacked_mask.jpg', np.hstack(masks))

                    frames.clear()
                    frames_o.clear()
                    np_imgs.clear()
                else:
                    continue
            except queue.Empty:
                continue


if __name__ == "__main__":
    model_file = '/workspace/timesformer/TimeSformer_divST_8x32_224_K600.pyth'
    assert Path(model_file).exists(), "Model file does not exist."

    cfg = get_cfg()
    cfg.merge_from_file('/workspace/timesformer/configs/Kinetics/TimeSformer_divST_8x32_224.yaml')
    cfg.TRAIN.ENABLE = False
    cfg.TIMESFORMER.PRETRAINED_MODEL = model_file
    model = MODEL_REGISTRY.get('vit_base_patch16_224')(cfg)

    with open('/workspace/timesformer/kinetics600-label.json') as f:
        ssv2_labels = json.load(f)
    ssv2_labels = list(ssv2_labels.values())

    path_to_video = Path('/workspace/timesformer/mfalldown.mp4')
    assert path_to_video.exists(), "Video file does not exist."

    num_frames = 8
    frame_queue = queue.Queue(maxsize=num_frames)

    # Create threads for loading video frames and analyzing them
    video_thread = threading.Thread(target=load_video_frames, args=(path_to_video, frame_queue, num_frames))
    analysis_thread = threading.Thread(target=analyze_frames, args=(model, frame_queue, ssv2_labels))

    video_thread.start()
    analysis_thread.start()

    video_thread.join()
    analysis_thread.join()

    # # Attention rollout and visualization (optional)
    # att_roll = vau.DividedAttentionRollout(model)
    # masks = att_roll(video_input)

    # np_imgs = [vau.transform_plot(frame) for frame in video_input.squeeze().permute(1, 2, 3, 0).numpy()]
    # masks = vau.create_masks(list(rearrange(masks, 'h w t -> t h w')), np_imgs)
    # stacked_image = np.hstack(np_imgs)
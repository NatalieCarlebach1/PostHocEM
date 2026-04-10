import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from medpy import metric


def calculate_metric_percase(pred, gt):
    """Dice, Jaccard, HD95, ASD for one binary case."""
    if pred.sum() == 0:
        return 0.0, 0.0, 200.0, 200.0
    if gt.sum() == 0:
        return 0.0, 0.0, 200.0, 200.0
    dice = metric.binary.dc(pred, gt)
    jc   = metric.binary.jc(pred, gt)
    hd   = metric.binary.hd95(pred, gt)
    asd  = metric.binary.asd(pred, gt)
    return dice, jc, hd, asd


def sliding_window_inference(net, image, patch_size, stride_xy, stride_z, n_classes):
    """
    Patch-based sliding window inference on a single 3-D volume.
    image: numpy array (W, H, D), already normalised float32.
    Returns label_map (W, H, D) and score_map (n_classes, W, H, D).
    """
    w, h, d = image.shape
    p = patch_size if isinstance(patch_size, tuple) else (patch_size,) * 3

    pw = max((p[0] - w) // 2 + 1, 0)
    ph = max((p[1] - h) // 2 + 1, 0)
    pd = max((p[2] - d) // 2 + 1, 0)
    if pw or ph or pd:
        image = np.pad(image, [(pw, pw), (ph, ph), (pd, pd)], mode='constant')
    ww, hh, dd = image.shape

    sx = math.ceil((ww - p[0]) / stride_xy) + 1
    sy = math.ceil((hh - p[1]) / stride_xy) + 1
    sz = math.ceil((dd - p[2]) / stride_z)  + 1

    score_map = np.zeros((n_classes,) + (ww, hh, dd), dtype=np.float32)
    cnt       = np.zeros((ww, hh, dd), dtype=np.float32)

    net.eval()
    with torch.no_grad():
        for x in range(sx):
            xs = min(stride_xy * x, ww - p[0])
            for y in range(sy):
                ys = min(stride_xy * y, hh - p[1])
                for z in range(sz):
                    zs = min(stride_z * z, dd - p[2])
                    patch   = image[xs:xs+p[0], ys:ys+p[1], zs:zs+p[2]]
                    patch_t = torch.from_numpy(
                        patch[None, None].astype(np.float32)).cuda()
                    logits = net(patch_t)[0]
                    probs  = F.softmax(logits, dim=1).cpu().numpy()[0]
                    score_map[:, xs:xs+p[0], ys:ys+p[1], zs:zs+p[2]] += probs
                    cnt[xs:xs+p[0], ys:ys+p[1], zs:zs+p[2]] += 1

    score_map /= np.expand_dims(cnt + 1e-8, 0)
    label_map  = np.argmax(score_map, axis=0)

    if pw or ph or pd:
        label_map = label_map[pw:pw+w, ph:ph+h, pd:pd+d]
        score_map = score_map[:, pw:pw+w, ph:ph+h, pd:pd+d]
    return label_map, score_map


@torch.no_grad()
def evaluate(net, test_dataset, patch_size, stride_xy=16, stride_z=4, n_classes=2):
    """
    Sliding-window inference on full volumes.
    test_dataset: FullVolumeDataset — yields (image_np, label_np, case_name).
    Returns mean (Dice, Jc, HD95, ASD).
    """
    dice_list, jc_list, hd_list, asd_list = [], [], [], []
    net.eval()

    for i in range(len(test_dataset)):
        image_np, label_np, case_name = test_dataset[i]

        pred, _ = sliding_window_inference(
            net, image_np, patch_size, stride_xy, stride_z, n_classes)
        pred = pred.astype(np.uint8)

        d, jc, hd, asd = calculate_metric_percase(pred, label_np)
        dice_list.append(d)
        jc_list.append(jc)
        hd_list.append(hd)
        asd_list.append(asd)

    return (float(np.mean(dice_list)), float(np.mean(jc_list)),
            float(np.mean(hd_list)),  float(np.mean(asd_list)))

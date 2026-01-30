# infer_onnx.py
"""
滑窗推理脚本：输入一张大图，使用 ONNX 模型做 1024x1024 滑窗推理，overlap 融合输出 mask。
"""

import numpy as np
import cv2
import onnxruntime as ort
import argparse
import os
import json
import re
import time

from utils.label_mapping import aggregate_mask


def _parse_roi_id_from_image_path(image_path: str) -> str:
    """从文件名解析 ROI id。

    例：620_door_000210.jpg -> "620"
    """
    base = os.path.basename(image_path)
    m = re.match(r'^(\d+)_', base)
    if not m:
        raise ValueError(f'无法从文件名解析 ROI id（需要形如 620_xxx.jpg）：{base}')
    return m.group(1)


def _load_labelme_rectangles(roi_json_path: str) -> list:
    """读取 labelme 格式 roi_merged/*.json 中的 rectangle shapes，返回 [(x1,y1,x2,y2), ...]。"""
    with open(roi_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    rects = []
    for shape in data.get('shapes', []):
        if shape.get('shape_type') != 'rectangle':
            continue
        points = shape.get('points')
        if not points or len(points) != 2:
            continue
        (x1, y1), (x2, y2) = points
        x_min = float(min(x1, x2))
        y_min = float(min(y1, y2))
        x_max = float(max(x1, x2))
        y_max = float(max(y1, y2))
        rects.append((x_min, y_min, x_max, y_max))
    return rects


def _clip_and_int_rect(rect, w: int, h: int):
    """将 float rect 裁剪到图像边界并转 int，返回 (x1,y1,x2,y2)（右下为开区间）。"""
    x1, y1, x2, y2 = rect
    x1 = int(np.floor(x1))
    y1 = int(np.floor(y1))
    x2 = int(np.ceil(x2))
    y2 = int(np.ceil(y2))
    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _union_rects(rects):
    x1 = min(r[0] for r in rects)
    y1 = min(r[1] for r in rects)
    x2 = max(r[2] for r in rects)
    y2 = max(r[3] for r in rects)
    return (x1, y1, x2, y2)


def roi_sliding_window_inference(
    full_img,
    session,
    roi_json_path: str,
    window_size=1024,
    stride=896,
    num_classes=4,
    roi_mode: str = 'multi',
    progress: bool = False,
    downscale: float = 1.0,
    apply_softmax: bool = True,
):
    """按 roi_json 的矩形区域裁剪推理，再贴回全图输出。

    roi_mode:
      - 'multi': 对每个矩形分别推理并贴回（推荐；与 roi_merged 的多框兼容）
      - 'union': 取所有矩形的并集框，只裁一次推理
    """
    h, w = full_img.shape
    rects = _load_labelme_rectangles(roi_json_path)
    if not rects:
        raise ValueError(f'ROI 配置中未找到 rectangle shapes: {roi_json_path}')

    if roi_mode == 'union':
        rects = [_union_rects(rects)]
    elif roi_mode != 'multi':
        raise ValueError(f'未知 roi_mode: {roi_mode}，可选 multi/union')

    full_pred = np.zeros((h, w), dtype=np.uint8)
    # 注意：不要在整图上分配 (C,H,W) 的 prob_map，大图会直接爆内存。
    # bbox 的 score 在每个 ROI crop 内用 crop_prob_map 计算，再映射回整图坐标。
    full_bboxes = {str(cls): [] for cls in range(1, num_classes)}

    print(f'ROI rectangles: {len(rects)} (mode={roi_mode})')

    for idx, rect in enumerate(rects, start=1):
        clipped = _clip_and_int_rect(rect, w=w, h=h)
        if clipped is None:
            continue
        x1, y1, x2, y2 = clipped
        crop = full_img[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_h, crop_w = crop.shape
        n_y = (max(crop_h - 1, 0) // stride) + 1
        n_x = (max(crop_w - 1, 0) // stride) + 1
        print(f'[{idx}/{len(rects)}] crop=({crop_w}x{crop_h}), windows≈{n_x*n_y}')
        crop_pred, crop_prob_map = sliding_window_inference(
            crop,
            session,
            window_size=window_size,
            stride=stride,
            num_classes=num_classes,
            return_prob_map=True,
            progress=progress,
            downscale=downscale,
            apply_softmax=apply_softmax,
        )
        full_pred[y1:y2, x1:x2] = crop_pred
        crop_bboxes = mask_to_bboxes(crop_pred, num_classes, prob_map=crop_prob_map)
        # 将 crop 内的 bbox 坐标映射回整图坐标
        for cls, boxes in crop_bboxes.items():
            for box in boxes:
                if len(box) == 5:
                    bx1, by1, bx2, by2, score = box
                else:
                    bx1, by1, bx2, by2 = box
                    score = -1
                full_bboxes[cls].append([bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, score])

    return full_pred, full_bboxes

    # ...existing code...
def sliding_window_inference(
    image,
    session,
    window_size=1024,
    stride=896,
    num_classes=4,
    return_prob_map: bool = False,
    progress: bool = False,
    progress_every: int = 10,
    downscale: float = 1.0,
    apply_softmax: bool = True,
):
    # 当 downscale<1 时先整体下采样再推理，完毕后再恢复至原分辨率，保持训练/推理尺寸一致以提升速度。
    if downscale != 1.0:
        h, w = image.shape
        target_w = max(1, int(round(w * downscale)))
        target_h = max(1, int(round(h * downscale)))
        scaled_image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

        scaled_window = max(8, int(round(window_size * downscale)))
        scaled_stride = max(1, int(round(stride * downscale)))

        if return_prob_map:
            pred_ds, prob_map_ds = sliding_window_inference(
                scaled_image,
                session,
                window_size=scaled_window,
                stride=scaled_stride,
                num_classes=num_classes,
                return_prob_map=True,
                progress=progress,
                progress_every=progress_every,
                downscale=1.0,
                apply_softmax=apply_softmax,
            )
        else:
            pred_ds = sliding_window_inference(
                scaled_image,
                session,
                window_size=scaled_window,
                stride=scaled_stride,
                num_classes=num_classes,
                return_prob_map=False,
                progress=progress,
                progress_every=progress_every,
                downscale=1.0,
                apply_softmax=apply_softmax,
            )

        pred = cv2.resize(pred_ds.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        if return_prob_map:
            prob_map_up = np.zeros((num_classes, h, w), dtype=np.float32)
            for c in range(num_classes):
                prob_map_up[c] = cv2.resize(prob_map_ds[c], (w, h), interpolation=cv2.INTER_LINEAR)
            return pred, prob_map_up
        return pred

    h, w = image.shape
    prob_map = np.zeros((num_classes, h, w), dtype=np.float32)
    count_map = np.zeros((h, w), dtype=np.float32)

    n_y = len(range(0, h, stride))
    n_x = len(range(0, w, stride))
    total_windows = n_y * n_x
    done = 0
    t0 = time.time()

    t_crop = 0.0
    t_infer = 0.0
    t_softmax = 0.0
    t_accum = 0.0

    # UNet 结构常要求输入尺寸为特定倍数（通常为 2^n），避免下采样/上采样拼接尺寸不匹配
    pad_divisor = 32

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1 = y
            x1 = x
            y2 = min(y1 + window_size, h)
            x2 = min(x1 + window_size, w)
            y1 = max(y2 - window_size, 0)
            x1 = max(x2 - window_size, 0)

            t1 = time.perf_counter()
            patch = image[y1:y2, x1:x2]
            ph, pw = patch.shape
            pad_h = (pad_divisor - (ph % pad_divisor)) % pad_divisor
            pad_w = (pad_divisor - (pw % pad_divisor)) % pad_divisor
            if pad_h or pad_w:
                patch = np.pad(patch, ((0, pad_h), (0, pad_w)), mode='reflect')
            patch_in = patch.astype(np.float32)[None, None] / 255.0  # (1,1,H,W)
            t_crop += time.perf_counter() - t1

            ort_inputs = {session.get_inputs()[0].name: patch_in}
            t2 = time.perf_counter()
            tn1 =time.time()
            out = session.run(None, ort_inputs)[0]  # (1, num_classes, H, W)
            # 跑一次推理后：
            profile_file = session.end_profiling()
            print(profile_file)
            tn = time.time()
            print(f'推理：：：{1000*(tn-tn1):.2f}ms')
            t_infer += time.perf_counter() - t2
            patch_out = out[0]  # (C, ph, pw)
            if pad_h or pad_w:
                patch_out = patch_out[:, :ph, :pw]

            if return_prob_map and apply_softmax:
                t3 = time.perf_counter()
                m = np.max(patch_out, axis=0, keepdims=True)
                e = np.exp(patch_out - m)
                patch_out = e / np.sum(e, axis=0, keepdims=True)
                t_softmax += time.perf_counter() - t3

            t4 = time.perf_counter()
            prob_map[:, y1:y2, x1:x2] += patch_out
            count_map[y1:y2, x1:x2] += 1
            t_accum += time.perf_counter() - t4

            done += 1
            if progress:
                if done % max(1, progress_every) == 0 or done == total_windows:
                    elapsed = time.time() - t0
                    it_s = elapsed / max(1, done)
                    eta = it_s * max(0, total_windows - done)
                    print(f'  windows {done}/{total_windows} ({done/total_windows:.0%}) | {it_s:.2f}s/it | ETA {eta/60:.1f} min')

    total = time.time() - t0
    print(
        f'profile: windows={done}/{total_windows} | total={total:.2f}s | '
        f'crop={t_crop:.2f}s | infer={t_infer:.2f}s | softmax={t_softmax:.2f}s | accum={t_accum:.2f}s'
    )
    prob_map = prob_map / np.clip(count_map, 1e-6, None)
    pred = np.argmax(prob_map, axis=0).astype(np.uint8)
    if return_prob_map:
        return pred, prob_map
    return pred

def mask_to_bboxes(mask, num_classes, prob_map=None):
    """从类别 mask 提取各类缺陷的 bbox。

    mask: (H, W) uint8，像素值为类别 id
    prob_map: (C, H, W) float32，可选，用于给每个 bbox 计算平均置信度

    返回: {"1": [[x1,y1,x2,y2,score], ...], ...}
    score 为该连通域内该类别概率均值；若 prob_map 为空则为 -1。
    """
    bboxes = {}

    for cls in range(1, num_classes):
        cls_mask = (mask == cls).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cls_mask, connectivity=8)

        if num_labels <= 1:
            bboxes[str(cls)] = []
            continue

        scores = None
        if prob_map is not None:
            # 一次性统计每个连通域的概率和，避免逐个连通域布尔索引（大图极慢）
            lbl = labels.reshape(-1)
            vals = prob_map[cls].reshape(-1).astype(np.float64, copy=False)
            sums = np.bincount(lbl, weights=vals, minlength=num_labels)
            areas = stats[:, cv2.CC_STAT_AREA].astype(np.float64, copy=False)
            scores = np.full((num_labels,), -1.0, dtype=np.float64)
            valid = areas > 0
            scores[valid] = sums[valid] / areas[valid]

        boxes = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area <= 0:
                continue
            score = float(scores[i]) if scores is not None else -1
            boxes.append([int(x), int(y), int(x + w - 1), int(y + h - 1), score])

        bboxes[str(cls)] = boxes

    return bboxes

def main():
    start =time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='输入图片路径')
    parser.add_argument('--onnx', type=str, default='unet.onnx', help='ONNX模型路径')
    parser.add_argument('--out', type=str, default='pred_mask.png', help='输出mask路径')
    parser.add_argument('--json', type=str, default='pred_bboxes.json', help='输出bbox json路径')
    parser.add_argument('--window', type=int, default=1024)
    parser.add_argument('--stride', type=int, default=896)
    parser.add_argument('--num_classes', type=int, default=4)
    parser.add_argument('--downscale', type=float, default=0.5, help='推理前整体下采样比例（如0.5可将1024窗口等效为512），1.0为关闭')
    parser.add_argument('--no-softmax-cpu', action='store_true', help='关闭CPU softmax（仅当模型输出已是概率时使用）')

    parser.add_argument('--progress', action='store_true', help='打印滑窗推理进度（CPU推理大图时建议开启）')

    parser.add_argument(
        '--providers',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help='auto: 优先CUDA(若可用)否则CPU；cpu: 强制只用CPU；cuda: 强制使用CUDA(不可用则报错)'
    )

    # ROI 裁剪推理：根据图片文件名前缀数字选择 configs/roi_merged/{id}.json
    parser.add_argument('--roi_dir', type=str, default=os.path.join('configs', 'roi_merged'), help='ROI配置目录')
    parser.add_argument('--roi_mode', type=str, default='multi', choices=['multi', 'union'], help='ROI多框处理方式')
    parser.add_argument('--no-roi', dest='use_roi', action='store_false', help='关闭ROI裁剪，直接对整图滑窗推理')
    parser.set_defaults(use_roi=True)

    # 新增 ignore 区域 json 目录参数
    parser.add_argument('--ignore_dir', type=str, default=os.path.join('configs', 'roi_merged'), help='ignore区域json目录')

    # 已去除聚合参数，始终输出细粒度（12类）结果
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f'找不到图片: {args.image}')

    if args.downscale != 1.0:
        approx_window = max(1, int(round(args.window * args.downscale)))
        approx_stride = max(1, int(round(args.stride * args.downscale)))
        print(f'使用下采样推理: factor={args.downscale}，等效 window≈{approx_window} stride≈{approx_stride}')

    available = ort.get_available_providers()
    print(f'onnxruntime available providers: {available}')

    if args.providers == 'cpu':
        providers = ['CPUExecutionProvider']
    elif args.providers == 'cuda':
        if 'CUDAExecutionProvider' not in available:
            raise RuntimeError(f'当前 onnxruntime 不支持 CUDAExecutionProvider，available={available}')
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if 'CUDAExecutionProvider' in available else ['CPUExecutionProvider']
    so = ort.SessionOptions()
    so.enable_profiling = True
    session = ort.InferenceSession(args.onnx, sess_options=so, providers=providers)
    print(f'onnxruntime session providers (in use): {session.get_providers()}')

    roi_id = None
    if args.use_roi:
        roi_id = _parse_roi_id_from_image_path(args.image)
        # 先处理 ignore 区域，将 img 对应像素置为背景
        ignore_json_path = os.path.join('configs', 'roi_ignore', f'{roi_id}_ignore.json')
        if os.path.exists(ignore_json_path):
            try:
                ignore_rects = _load_labelme_rectangles(ignore_json_path)
                h, w = img.shape
                for rect in ignore_rects:
                    clipped = _clip_and_int_rect(rect, w=w, h=h)
                    if clipped is None:
                        continue
                    x1, y1, x2, y2 = clipped
                    img[y1:y2, x1:x2] = 0  # 灰度图直接置为背景像素
                print(f'已应用 ignore 区域: {ignore_json_path}')
            except Exception as e:
                print(f'忽略区域处理失败（已跳过）: {ignore_json_path}, 错误: {e}')
        roi_json_path = os.path.join(args.roi_dir, f'{roi_id}.json')
        if not os.path.exists(roi_json_path):
            raise FileNotFoundError(f'找不到ROI配置: {roi_json_path}（由图片名解析得到 id={roi_id}）')
        pred_mask, bboxes = roi_sliding_window_inference(
            img,
            session,
            roi_json_path=roi_json_path,
            window_size=args.window,
            stride=args.stride,
            num_classes=args.num_classes,
            roi_mode=args.roi_mode,
            progress=args.progress,
            downscale=args.downscale,
            apply_softmax=not args.no_softmax_cpu,
        )
    else:
        pred_mask, prob_map = sliding_window_inference(
            img,
            session,
            args.window,
            args.stride,
            args.num_classes,
            return_prob_map=True,
            progress=args.progress,
            downscale=args.downscale,
            apply_softmax=not args.no_softmax_cpu,
        )
        bboxes = mask_to_bboxes(pred_mask, args.num_classes, prob_map=prob_map)

    out_mask = pred_mask
    vis_num_classes = args.num_classes

    cv2.imwrite(args.out, out_mask.astype(np.uint8) * (255 // max(1, vis_num_classes - 1)))

    with open(args.json, 'w', encoding='utf-8') as f:
        json.dump(bboxes, f, ensure_ascii=False, indent=2)
    end =time.time()
    print(f'推理完成，mask已保存: {args.out}，bbox已保存: {args.json},耗时{(end-start):.3f}s')

if __name__ == '__main__':
    main()

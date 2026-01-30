#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Draw predicted bboxes (pred_bboxes.json) on an image.

The json format is expected to be:
{
  "1": [[x1,y1,x2,y2], ...],
  "2": [[x1,y1,x2,y2], ...],
  "3": [[x1,y1,x2,y2], ...]
}

Coordinates are in the original image coordinate system.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


def _clip_box(box: List[int], w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [int(v) for v in box]
    # allow inclusive x2/y2 in input; convert to inclusive drawing directly
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w - 1))
    y2 = max(0, min(y2, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _box_area(x1: int, y1: int, x2: int, y2: int) -> int:
    return max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)


def _default_color_for_class(cls: str) -> Tuple[int, int, int]:
    # BGR (use display-group colors)
    name = class_id_to_display_name(cls)
    return {
        "scratch": (0, 255, 255),        # yellow
        "paint_defect": (0, 165, 255),   # orange
        "foreign_object": (255, 0, 255), # magenta
    }.get(name, (0, 255, 0))


def class_id_to_display_name(cls: str) -> str:
    """将预测类别 id 映射成你要展示的类别名。

    规则（与你描述一致）：
    - 1 -> scratch
    - 2 / 3 -> paint_defect
    - 其余（包括原本要 ignore 的那些）-> foreign_object
    """
    try:
        cid = int(cls)
    except Exception:
        return "foreign_object"

    if cid == 1:
        return "scratch"
    if cid in (2, 3):
        return "paint_defect"
    return "foreign_object"


def draw_bboxes(
    image: np.ndarray,
    bboxes: Dict[str, List[List[int]]],
    min_area: int = 0,
    max_boxes_per_class: int = 0,
    thickness: int = 2,
    draw_label: bool = True,
) -> np.ndarray:
    if image.ndim == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()

    h, w = vis.shape[:2]

    for cls, boxes in bboxes.items():
        color = _default_color_for_class(cls)
        filtered_boxes = []
        for box in boxes:
            # 支持 [x1, y1, x2, y2, score] 或 [x1, y1, x2, y2]
            if len(box) == 5:
                x1, y1, x2, y2, score = box
            else:
                x1, y1, x2, y2 = box
                score = None
            clipped = _clip_box([x1, y1, x2, y2], w=w, h=h)
            if clipped is None:
                continue
            x1, y1, x2, y2 = clipped
            if min_area and _box_area(x1, y1, x2, y2) < min_area:
                continue
            # score过滤
            if score is not None and draw_bboxes.score_thresh is not None:
                if score < draw_bboxes.score_thresh:
                    continue
            filtered_boxes.append((x1, y1, x2, y2, score))

        # sort by area descending (draw bigger boxes first)
        filtered_boxes.sort(key=lambda b: _box_area(b[0], b[1], b[2], b[3]), reverse=True)
        if max_boxes_per_class and max_boxes_per_class > 0:
            filtered_boxes = filtered_boxes[:max_boxes_per_class]

        for x1, y1, x2, y2, score in filtered_boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            if draw_label:
                label = class_id_to_display_name(cls)
                if score is not None and score >= 0:
                    label = f"{label}={score:.2f}"
                # background for text
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                tx1, ty1 = x1, max(0, y1 - th - 6)
                tx2, ty2 = x1 + tw + 6, y1
                cv2.rectangle(vis, (tx1, ty1), (tx2, ty2), color, -1)
                cv2.putText(
                    vis,
                    label,
                    (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 0),
                    2,
                    cv2.LINE_AA,
                )

    return vis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--json", type=str, default="pred_bboxes.json", help="bbox json 路径")
    parser.add_argument("--out", type=str, default="pred_bboxes_vis.jpg", help="输出可视化图片路径")
    parser.add_argument("--min_area", type=int, default=500, help="过滤面积小于该值的框")
    parser.add_argument(
        "--max_boxes_per_class",
        type=int,
        default=200,
        help="每个类别最多画多少个框（0 表示不限制）",
    )
    parser.add_argument("--thickness", type=int, default=2, help="矩形框线宽")
    parser.add_argument("--no_label", action="store_true", help="不绘制 cls 标签")
    parser.add_argument("--score_thresh", type=float, default=None, help="只显示置信度高于该值的框")
    args = parser.parse_args()

    img = cv2.imread(args.image, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"找不到图片: {args.image}")

    if not os.path.exists(args.json):
        raise FileNotFoundError(f"找不到json: {args.json}")

    with open(args.json, "r", encoding="utf-8") as f:
        bboxes = json.load(f)

    # 传递 score_thresh
    draw_bboxes.score_thresh = args.score_thresh
    vis = draw_bboxes(
        img,
        bboxes,
        min_area=args.min_area,
        max_boxes_per_class=args.max_boxes_per_class,
        thickness=args.thickness,
        draw_label=(not args.no_label),
    )

    ok = cv2.imwrite(args.out, vis)
    if not ok:
        raise RuntimeError(f"保存失败: {args.out}")

    # Simple stats
    counts = {k: len(v) for k, v in bboxes.items() if isinstance(v, list)}
    print(f"已保存: {args.out}")
    print(f"原始框数量: {counts}")
    print(f"绘制参数: min_area={args.min_area}, max_boxes_per_class={args.max_boxes_per_class}")


if __name__ == "__main__":
    main()

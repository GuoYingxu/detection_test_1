"""Label normalization and class mapping with ignore index support."""

# 建议定义一个全局的忽略索引，通常 PyTorch 的 CrossEntropyLoss 默认 ignore_index=255
IGNORE_INDEX = 255

LABEL_NORMALIZATION = {
    "Energy rating label": "energy_rating_label",
    "Foregin QR CODE": "foreign_qr_code",
    "LOGO": "logo",
    "LOGO ERROR": "logo_error",
    "adhesive tape": "adhesive_tape",
    "air bubble": "air_bubble",
    "film": "film",
    "film damage": "film_damage",
    "foreign matter": "foreign_matter",
    "hair": "hair",
    "paint defect": "paint_defect",
    "paint peeling": "paint_peeling",
    "scratch": "scratch",
    "sticky note": "sticky_note",
}

CLASS_MAPPING = {
    # --- 每个缺陷标签一个独立类别（便于训练后再聚合） ---
    "scratch": 1,
    "paint_defect": 2,
    "paint_peeling": 3,
    "air_bubble": 4,
    "adhesive_tape": 5,
    "sticky_note": 6,
    "film": 7,
    "film_damage": 8,
    "foreign_matter": 9,
    "hair": 10,
    "logo_error": 11,

    # --- 忽略区域 (Ignore / Don't Care) ---
    "energy_rating_label": IGNORE_INDEX,
    "foreign_qr_code": IGNORE_INDEX,
    "logo": IGNORE_INDEX,
}

BACKGROUND = 0

# -----------------------------
# 后处理聚合映射（训练细粒度 -> 推理/评估聚合）
# 将多种异物类聚合为“异物”(3)，将“掉漆/漆病”聚合为 2；保持划痕为 1。
# 忽略区域与背景保持不变。
# -----------------------------

# 基于标签名的聚合目标（便于阅读与维护）
AGGREGATED_CLASS_BY_LABEL = {
    # 划痕
    "scratch": 1,

    # 掉漆/漆病聚合为 2
    "paint_defect": 2,
    "paint_peeling": 2,

    # 异物聚合为 3
    "air_bubble": 3,
    "adhesive_tape": 3,
    "sticky_note": 3,
    "film": 3,
    "film_damage": 3,
    "foreign_matter": 3,
    "hair": 3,
    "logo_error": 3,
}

# 将细粒度类别 ID 映射到聚合后的类别 ID
ID_AGGREGATION_MAP = {}
for label, fine_id in CLASS_MAPPING.items():
    if label in AGGREGATED_CLASS_BY_LABEL:
        ID_AGGREGATION_MAP[fine_id] = AGGREGATED_CLASS_BY_LABEL[label]

# 明确背景与忽略索引的行为
ID_AGGREGATION_MAP[BACKGROUND] = BACKGROUND
ID_AGGREGATION_MAP[IGNORE_INDEX] = IGNORE_INDEX


def aggregate_mask(mask):
    """将细粒度预测掩码聚合为三类方案（1:划痕, 2:掉漆, 3:异物）。

    支持 torch.Tensor 或 numpy.ndarray 输入；输出与输入类型一致。
    - 背景(0)保持不变
    - 忽略(IGNORE_INDEX)保持不变
    - 其它类别按 ID_AGGREGATION_MAP 进行聚合
    """
    try:
        import torch  # 可选依赖
    except Exception:
        torch = None

    # 计算映射数组大小，确保可覆盖 IGNORE_INDEX
    max_id_in_map = max(ID_AGGREGATION_MAP.keys()) if ID_AGGREGATION_MAP else 0
    size = max(IGNORE_INDEX, max_id_in_map) + 1

    def _build_mapping_array(dtype, device=None):
        # 默认恒等映射，未显式指定的类别保持原值
        if torch is not None and device is not None:
            mapping = torch.arange(size, dtype=dtype, device=device)
        else:
            import numpy as np
            mapping = np.arange(size, dtype=dtype)
        for k, v in ID_AGGREGATION_MAP.items():
            mapping[k] = v
        return mapping

    # torch.Tensor 路径（优先保持设备与 dtype）
    if torch is not None and isinstance(mask, torch.Tensor):
        mapping = _build_mapping_array(mask.dtype, device=mask.device)
        return mapping[mask]

    # numpy.ndarray 路径
    try:
        import numpy as np
    except Exception:
        np = None

    if np is not None and isinstance(mask, np.ndarray):
        mapping = _build_mapping_array(mask.dtype)
        return mapping[mask]

    # 其它类型不支持，抛出异常以提醒调用方
    raise TypeError("aggregate_mask() 仅支持 torch.Tensor 或 numpy.ndarray 掩码输入")

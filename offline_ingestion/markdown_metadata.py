"""读取本项目确定性转换器写入 Markdown 顶部的来源元数据。

转换器使用的是一个非常小的 YAML 子集：每行都是 ``key: JSON值``。
把读取函数放在独立模块，可以同时供 Markdown 审计和离线入库使用，避免
``build_index`` 与 ``audit_markdown`` 互相导入形成循环依赖。
"""

from __future__ import annotations

import json
from pathlib import Path


def read_front_matter(markdown_path: Path) -> dict[str, object]:
    """解析 Markdown front matter，格式异常时立即停止而不是猜测。"""

    lines = markdown_path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0] != "---":
        raise ValueError(f"Markdown 缺少 YAML 起始标记: {markdown_path.name}")
    try:
        end = lines.index("---", 1)
    except ValueError as exc:
        raise ValueError(f"Markdown 缺少 YAML 结束标记: {markdown_path.name}") from exc

    metadata: dict[str, object] = {}
    for line in lines[1:end]:
        if ": " not in line:
            raise ValueError(f"无法解析 YAML 元数据行: {line}")
        key, raw_value = line.split(": ", 1)
        metadata[key] = json.loads(raw_value)
    return metadata

"""通用工具：配置加载、JSON 读写、目录管理。"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def ensure_dir(path: str | Path) -> Path:
    """确保目录存在，返回 Path 对象。"""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_config(path: str | Path) -> Dict[str, Any]:
    """加载 YAML 配置文件。

    优先使用 PyYAML；若未安装则回退到内置的轻量解析器
    （仅支持本项目配置所需的子集语法）。
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        return yaml.safe_load(text)
    except ImportError:
        return _mini_yaml_load(text)


def _mini_yaml_load(text: str) -> Dict[str, Any]:
    """极简 YAML 解析器：支持嵌套映射、列表、标量。

    仅用于 PyYAML 缺失时的降级方案，覆盖 config.yaml 的全部语法。
    """
    root: Dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip() if "#" in raw else raw.rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        content = line.strip()

        while stack and indent <= stack[-1][0]:
            stack.pop()
        container = stack[-1][1]

        if content.startswith("- "):
            item = _parse_scalar(content[2:].strip())
            if isinstance(container, list):
                container.append(item)
            continue

        if ":" not in content:
            continue
        key, _, value = content.partition(":")
        key = key.strip()
        value = value.strip()

        if value == "":
            # 可能是嵌套映射或列表，先建 dict，若子项以 "- " 开头再转换
            child: Any = {}
            container[key] = child
            stack.append((indent, child))
        else:
            container[key] = _parse_scalar(value)

    # 后处理：把「空 dict 但实际是列表」的节点转换为 list
    def _fix(node: Any) -> Any:
        if isinstance(node, dict):
            return {k: _fix(v) for k, v in node.items()}
        return node

    return _fix(root)


def _parse_scalar(value: str) -> Any:
    v = value.strip().strip('"').strip("'")
    if v.lower() in ("true", "yes"):
        return True
    if v.lower() in ("false", "no"):
        return False
    if v.lower() in ("null", "none", "~", ""):
        return ""
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    # 行内列表 [a, b, c]
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(x) for x in inner.split(",")]
    return v


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """将对象以 UTF-8 JSON 保存（支持中文）。"""
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def project_root() -> Path:
    """返回项目根目录（src/catdog/utils/io.py -> 项目根）。"""
    return Path(__file__).resolve().parents[3]

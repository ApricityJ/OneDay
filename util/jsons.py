from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def of_json(path: [str | Path]) -> Any:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_json(obj: Any, path: [str | Path]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=4, ensure_ascii=False)

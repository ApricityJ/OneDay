from __future__ import annotations

import json
from pathlib import Path


def of_json(path: [str|Path]) -> dict:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def to_json(data: dict, path: [str|Path]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

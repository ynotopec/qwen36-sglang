#!/usr/bin/env python3
"""Make SGLang's Qwen3-ASR config registration coexist with Transformers."""

from __future__ import annotations

import argparse
from importlib.util import find_spec
from pathlib import Path


OLD_REGISTRATION = 'AutoConfig.register("qwen3_asr", Qwen3ASRConfig)'
NEW_REGISTRATION = (
    'AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)'
)


def installed_config_path() -> Path:
    spec = find_spec("sglang")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Cannot locate the active sglang package")
    package_dir = Path(next(iter(spec.submodule_search_locations)))
    return package_dir / "srt/configs/qwen3_asr.py"


def patch_config(path: Path) -> bool:
    if not path.exists():
        print(f"No local qwen3_asr config at {path}; no patch needed")
        return False
    source = path.read_text()
    if NEW_REGISTRATION in source:
        return False
    if "AutoConfig.register" not in source or '"qwen3_asr"' not in source:
        return False
    if source.count(OLD_REGISTRATION) != 1:
        raise RuntimeError(f"Unexpected qwen3_asr registration in {path}")
    path.write_text(source.replace(OLD_REGISTRATION, NEW_REGISTRATION))
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    path = args.path or installed_config_path()
    changed = patch_config(path)
    print(f"{'Patched' if changed else 'Already patched'} {path}")


if __name__ == "__main__":
    main()

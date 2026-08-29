#!/usr/bin/env python3
"""Make SGLang's Qwen3-ASR config registration coexist with Transformers."""

from __future__ import annotations

import argparse
from importlib.metadata import distribution
from pathlib import Path


OLD_REGISTRATION = 'AutoConfig.register("qwen3_asr", Qwen3ASRConfig)'
NEW_REGISTRATION = (
    'AutoConfig.register("qwen3_asr", Qwen3ASRConfig, exist_ok=True)'
)


def installed_config_path() -> Path:
    return Path(
        distribution("sglang").locate_file("sglang/srt/configs/qwen3_asr.py")
    )


def patch_config(path: Path) -> bool:
    source = path.read_text()
    if NEW_REGISTRATION in source:
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

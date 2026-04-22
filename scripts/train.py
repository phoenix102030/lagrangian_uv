from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lagrangian_uv_stage2.train import train_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the stage-2 UV Lagrangian state-space model.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()
    train_from_config(args.config)


if __name__ == "__main__":
    main()

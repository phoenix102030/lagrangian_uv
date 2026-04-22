from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lagrangian_uv_stage2.config import load_config
from lagrangian_uv_stage2.data import build_data_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect stage-2 UV data shapes and config-derived layout.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = build_data_bundle(config)

    print("Train windows:", len(bundle.train_dataset))
    print("Validation windows:", len(bundle.val_dataset))
    print("Online sequence shape:", tuple(bundle.online_sequence["obs"].shape))
    print("Site coords:", bundle.site_coords.tolist())
    print("Feature names:", bundle.feature_names)


if __name__ == "__main__":
    main()

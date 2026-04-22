from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lagrangian_uv_stage2.evaluate import rolling_forecast_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Run rolling forecasts on the online UV data.")
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--device", default=None, help="Optional device override, for example cpu or cuda.")
    parser.add_argument("--context-window", type=int, default=None, help="Optional context window override.")
    parser.add_argument("--forecast-horizon", type=int, default=None, help="Optional forecast horizon override.")
    parser.add_argument("--stride", type=int, default=None, help="Optional stride override.")
    parser.add_argument("--output-json", default=None, help="Optional path to save the rolling forecast summary as JSON.")
    args = parser.parse_args()

    results = rolling_forecast_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        context_window_override=args.context_window,
        forecast_horizon_override=args.forecast_horizon,
        stride_override=args.stride,
    )

    print(f"Forecast horizon: {results['forecast_horizon']} steps")
    print(f"Model MAE: {results['model_metrics']['mae']:.6f}")
    print(f"Model RMSE: {results['model_metrics']['rmse']:.6f}")
    print(f"Persistence MAE: {results['persistence_metrics']['mae']:.6f}")
    print(f"Persistence RMSE: {results['persistence_metrics']['rmse']:.6f}")
    print(
        "Improvement vs persistence: "
        f"MAE={results['improvement_vs_persistence_pct']['mae']:.2f}% "
        f"RMSE={results['improvement_vs_persistence_pct']['rmse']:.2f}%"
    )

    print("Mean MAE by state dimension:")
    for feature_name, mae in zip(results["feature_names"], results["mean_mae"]):
        print(f"  {feature_name}: {float(mae):.6f}")

    print("Horizon-wise MAE:")
    for step_idx, (model_mae, persistence_mae) in enumerate(
        zip(results["horizon_metrics"]["model_mae"], results["horizon_metrics"]["persistence_mae"]),
        start=1,
    ):
        print(
            f"  step={step_idx:02d} "
            f"model={float(model_mae):.6f} "
            f"persistence={float(persistence_mae):.6f}"
        )

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        serializable = {
            "feature_names": results["feature_names"],
            "forecast_horizon": results["forecast_horizon"],
            "model_metrics": results["model_metrics"],
            "persistence_metrics": results["persistence_metrics"],
            "improvement_vs_persistence_pct": results["improvement_vs_persistence_pct"],
            "mean_mae": [float(value) for value in results["mean_mae"]],
            "mean_persistence_mae": [float(value) for value in results["mean_persistence_mae"]],
            "horizon_metrics": {
                "model_mae": [float(value) for value in results["horizon_metrics"]["model_mae"]],
                "model_rmse": [float(value) for value in results["horizon_metrics"]["model_rmse"]],
                "persistence_mae": [float(value) for value in results["horizon_metrics"]["persistence_mae"]],
                "persistence_rmse": [float(value) for value in results["horizon_metrics"]["persistence_rmse"]],
            },
        }
        output_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        print(f"Saved summary JSON to {output_path}")


if __name__ == "__main__":
    main()

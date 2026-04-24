from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lagrangian_uv_stage2.evaluate import export_window_diagnostics_from_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export learned parameters, advection diagnostics, transition matrices, and plots for a chosen window."
    )
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--checkpoint", required=True, help="Path to a model checkpoint.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory for exported diagnostics.")
    parser.add_argument("--device", default=None, help="Optional device override, for example cpu or cuda.")
    parser.add_argument("--split", choices=["train", "val", "online"], default="val", help="Which sequence split to export.")
    parser.add_argument("--window-index", type=int, default=0, help="Window index for train/val splits.")
    parser.add_argument("--online-start", type=int, default=0, help="Start index when split=online.")
    parser.add_argument("--online-length", type=int, default=None, help="Optional sequence length when split=online.")
    args = parser.parse_args()

    summary = export_window_diagnostics_from_checkpoint(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
        split=args.split,
        window_index=args.window_index,
        online_start=args.online_start,
        online_length=args.online_length,
    )

    print("Export complete.")
    print(f"  Split: {summary['split']}")
    print(f"  Window index: {summary['window_index']}")
    print(f"  Negative log-likelihood: {summary['negative_log_likelihood']:.6f}")
    print(f"  Predicted MAE: {summary['predicted_metrics']['mae']:.6f}")
    print(f"  Predicted RMSE: {summary['predicted_metrics']['rmse']:.6f}")
    print(f"  Filtered MAE: {summary['filtered_metrics']['mae']:.6f}")
    print(f"  Filtered RMSE: {summary['filtered_metrics']['rmse']:.6f}")
    print(f"  Persistence MAE: {summary['persistence_comparison']['persistence_metrics']['mae']:.6f}")
    print(f"  Persistence RMSE: {summary['persistence_comparison']['persistence_metrics']['rmse']:.6f}")
    print(
        "  Improvement vs persistence: "
        f"MAE={summary['persistence_comparison']['improvement_vs_persistence_pct']['mae']:.2f}% "
        f"RMSE={summary['persistence_comparison']['improvement_vs_persistence_pct']['rmse']:.2f}%"
    )
    transition_diag = summary.get("structural_diagnostics", {}).get("transition", {})
    kernel_diag = summary.get("structural_diagnostics", {}).get("kernel_transition", {})
    if transition_diag:
        row_sum = transition_diag.get("row_sum", {})
        print(
            "  Dynamics row sum: "
            f"mean={row_sum.get('mean', 0.0):.6f} "
            f"min={row_sum.get('min', 0.0):.6f} "
            f"max={row_sum.get('max', 0.0):.6f}"
        )
        cross_blocks = {
            key: value
            for key, value in transition_diag.get("blocks", {}).items()
            if "_to_" in key and key.split("_to_")[0] != key.split("_to_")[1]
        }
        if cross_blocks:
            formatted = ", ".join(
                f"{key} mean_abs={block['mean_abs']:.3e}" for key, block in sorted(cross_blocks.items())
            )
            print(f"  Dynamics cross-component blocks: {formatted}")
    if kernel_diag:
        row_sum = kernel_diag.get("row_sum", {})
        print(
            "  Kernel row sum: "
            f"mean={row_sum.get('mean', 0.0):.6f} "
            f"min={row_sum.get('min', 0.0):.6f} "
            f"max={row_sum.get('max', 0.0):.6f}"
        )
    operator_diag = summary.get("one_step_operator_diagnostics", {})
    if operator_diag.get("available"):
        metrics = operator_diag["metrics"]
        persistence_mae = metrics["raw_persistence"]["denormalized"]["mae"]
        dynamics_mae = metrics["dynamics_transition_with_forcing"]["denormalized"]["mae"]
        kernel_mae = metrics["kernel_transition"]["denormalized"]["mae"]
        print(
            "  One-step operator MAE: "
            f"persistence={persistence_mae:.6f} "
            f"kernel={kernel_mae:.6f} "
            f"dynamics={dynamics_mae:.6f}"
        )
    findings = summary.get("diagnostic_findings", [])
    if findings:
        print("  Diagnostic findings:")
        for finding in findings[:8]:
            print(f"    [{finding.get('severity', 'info')}] {finding.get('code', 'diagnostic')}: {finding.get('message', '')}")
    print(f"  Diagnostic summary: {summary['files']['diagnostic_summary']}")
    print(f"  Array archive: {summary['files']['arrays']}")
    print("  Generated plots:")
    for path in summary.get("generated_plots", []):
        print(f"    {path}")


if __name__ == "__main__":
    main()

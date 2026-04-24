python -m scripts.export_diagnostics    --config configs/default.yaml   --checkpoint outputs/stage2_uv_02/best.pt   --split val    --window-index 0

python scripts/rolling_forecast.py   --config configs/default.yaml   --checkpoint outputs/stage2_uv/best.pt

# Lagrangian UV Stage-2

This is a clean, isolated reimplementation of the original DeepMIDE idea for the second-stage kernel you described.

The new project changes the modeling assumptions in four important ways:

1. It only models the `140m` layer.
2. It uses the `u` and `v` wind components directly instead of wind speed.
3. It predicts a stochastic advection distribution for each component:
   `mu_u, Sigma_u, mu_v, Sigma_v`.
4. It builds the transition kernel with the stage-2 dispersion term:

```math
K_t^{(ij)}(a,b)
\propto
|D_t^{(ij)}|^{-1/2}
\exp\left(
-1/2
\left(h_{ab} - d_t^{(ij)}\right)^\top
\left(D_t^{(ij)}\right)^{-1}
\left(h_{ab} - d_t^{(ij)}\right)
\right)
```

In this implementation:

- `h_ab` is the spatial lag between sites `a` and `b`
- `d_t^{(ij)}` is built from the predicted advection means
- `D_t^{(ij)}` is built from the predicted advection covariances
- the full transition matrix is assembled as a `2 x 2` block matrix over the `u/v` components, with each block of size `3 x 3`

By default, this code uses the stage-2 specialization

```math
d_t^{(ij)} = \Delta t \, (\mu_{i,t} - \mu_{j,t})
```

and, under an independent-advection approximation between `u` and `v`,

```math
D_t^{(ij)} = I_2 + 2 \Delta t^2 (\Sigma_{i,t} + \Sigma_{j,t})
```

which is the block-diagonal `Sigma_gamma` analogue of the theorem-inspired construction.

The current default experiment also adds three forecasting-oriented changes:

1. A learned persistence backbone, so the state update starts from an explicit autoregressive baseline.
2. A persistence-versus-kernel mixture, so transport stays on the main path instead of collapsing into a tiny residual.
3. A direct NWP forcing head and a hybrid training objective that mixes Kalman likelihood with one-step and multi-step forecast losses.

## Project Layout

- `configs/default.yaml`: main experiment config
- `src/lagrangian_uv_stage2/data.py`: `.mat` loading, 140m `u/v` extraction, scaling, datasets
- `src/lagrangian_uv_stage2/models/backbone.py`: NWP encoders for advection mean/covariance and direct forcing
- `src/lagrangian_uv_stage2/models/kernel.py`: theorem-inspired stage-2 kernel assembly
- `src/lagrangian_uv_stage2/models/state_space.py`: persistence-plus-residual state-space model and hybrid objective
- `src/lagrangian_uv_stage2/train.py`: offline training loop
- `src/lagrangian_uv_stage2/evaluate.py`: rolling forecast helpers on online data
- `scripts/train.py`: CLI entrypoint for training
- `scripts/rolling_forecast.py`: CLI entrypoint for online rolling forecasts
- `scripts/inspect_data.py`: quick data inspection helper

## Data Assumptions

Observation data:

- variable name: `Ws_uv`
- raw order at `140m`: `[U_E05, V_E05, U_E06, V_E06, U_ASOW6, V_ASOW6]`
- internal model order in this project: `[U_E05, U_E06, U_ASOW6, V_E05, V_E06, V_ASOW6]`

NWP data:

- variable name: `allVariMin_Grid`
- `u` branch channels: `U_100m, U_140m, U_180m`
- `v` branch channels: `V_100m, V_140m, V_180m`

## Quick Start

Create an environment with at least:

- `torch`
- `numpy`
- `PyYAML`
- `h5py`
- `scipy`

Then run:

```bash
cd /Users/felix/Downloads/30753199/DeepMIDE-main/lagrangian_uv_stage2
python scripts/inspect_data.py --config configs/default.yaml
python scripts/train.py --config configs/default.yaml
python scripts/rolling_forecast.py --config configs/default.yaml --checkpoint outputs/stage2_uv/best.pt
python scripts/rolling_forecast.py --config configs/default.yaml --checkpoint outputs/stage2_uv/best.pt --forecast-horizon 12
python scripts/export_diagnostics.py --config configs/default.yaml --checkpoint outputs/stage2_uv/best.pt --split val --window-index 0
```

For real multi-GPU training on a single node, launch with `torchrun`:

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.train --config configs/default.yaml
```

In this setup, `batch_size` is the per-GPU batch size, so with two GPUs and `batch_size: 8`, the effective global batch size is `16`.

If host memory is tight, prefer a smaller per-GPU batch size together with gradient accumulation. For example, with two GPUs, `batch_size: 8` and `gradient_accumulation_steps: 2`, the effective global batch size is `32` while each forward pass remains much lighter on CPU/GPU memory.

## Diagnostics Export

After training, you can export a full diagnostic package for any train/validation/online window:

```bash
python scripts/export_diagnostics.py \
  --config configs/default.yaml \
  --checkpoint outputs/stage2_uv/best.pt \
  --split val \
  --window-index 0
```

This writes:

- `diagnostics_arrays.npz`: observations, predicted/filtered states, advection means/covariances, forcing, drift terms, dispersion terms, kernel transitions, dynamics matrices, `Q`, `R`, and initial state quantities
- `learned_parameters.npz`: every learned parameter from the model state dict
- `parameter_summary.json`: names and shapes of all learned parameters
- `transition_matrices.npy`: the full time-varying `6 x 6` dynamics matrix sequence
- `kernel_transition_matrices.npy`: the raw kernel-induced transition sequence before persistence/kernel mixing
- `forcing.npy`: the learned NWP forcing sequence
- `advection_means.csv` and `advection_covariances.csv`: tabular exports of `\mu_t` and `\Sigma_t`
- `transition_matrix_long.csv`: all transition entries in long format
- `diagnostic_summary.json` and `validation_summary.json`: hybrid-loss components plus predictive/filtering MAE/RMSE summaries
- PNG figures for advection, covariance, forcing time series, matrix heatmaps, and a transition-matrix GIF where bubble size/color tracks weight magnitude

## Multi-Step Evaluation

To test whether the model starts to beat persistence at a longer horizon such as `12` steps (`2` hours at `10`-minute resolution), run:

```bash
python scripts/rolling_forecast.py \
  --config configs/default.yaml \
  --checkpoint outputs/stage2_uv/best.pt \
  --forecast-horizon 12 \
  --output-json outputs/stage2_uv/rolling_h12.json
```

The script reports:

- overall model MAE/RMSE
- overall persistence MAE/RMSE
- percentage improvement over persistence
- horizon-wise MAE for both model and persistence at each lead time

## What Is Configurable

Everything likely to change is exposed in `configs/default.yaml`, including:

- data paths
- observation/NWP channel selection
- site coordinates or coordinate source file
- window length and stride
- hybrid objective weights and forecast-loss horizon
- NWP encoder width and dropout
- advection mean scale
- persistence diagonal range, kernel-mix range, and forcing scale
- kernel jitter and identity mixing
- Kalman linear algebra dtype and adaptive Cholesky jitter
- robust matrix sanitization and diagonal fallback for early training
- conservative optimizer defaults plus non-finite batch skipping
- larger CUDA batch sizes and real multi-GPU `DistributedDataParallel`
- process/measurement covariance initializations
- learning rate, gradient clipping, epochs, and device

## Practical Notes

- The code tries `h5py` first for MATLAB v7.3 files and falls back to `scipy.io.loadmat` for older MATLAB files.
- Station coordinates can be read from the original `dataset/data_h100_180_offline.mat` file if `scipy` is available.
- If you prefer to hard-code station coordinates instead, set `data.coordinates.manual` in the config.

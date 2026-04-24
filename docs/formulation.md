# Stage-2 Formulation

This note explains the mathematical model implemented in `lagrangian_uv_stage2`.

## 1. State Definition

We only model the `140m` layer and keep the two horizontal wind components explicitly.

Let the three stations be

```math
s_1, s_2, s_3 \in \mathbb{R}^2.
```

At time `t`, define the latent state

```math
y_t =
\begin{bmatrix}
u_t(s_1) \\
u_t(s_2) \\
u_t(s_3) \\
v_t(s_1) \\
v_t(s_2) \\
v_t(s_3)
\end{bmatrix}
\in \mathbb{R}^6.
```

The observation vector is the imputed measured wind vector at the same sites:

```math
z_t \in \mathbb{R}^6.
```

In the current implementation, the observation equation is

```math
z_t = y_t + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0, R).
```

So the observation operator is `H = I_6`.

## 2. NWP Inputs and Advection Networks

At each time `t`, we split the NWP grids into two three-channel tensors:

```math
x^u_t \in \mathbb{R}^{3 \times 40 \times 40},
\qquad
x^v_t \in \mathbb{R}^{3 \times 40 \times 40},
```

where

```math
x^u_t = (U_{100m}, U_{140m}, U_{180m}),
\qquad
x^v_t = (V_{100m}, V_{140m}, V_{180m}).
```

We use two CNN branches:

```math
f_{\theta_u}: x^u_t \mapsto (\mu_{u,t}, \Sigma_{u,t}),
\qquad
f_{\theta_v}: x^v_t \mapsto (\mu_{v,t}, \Sigma_{v,t}),
```

with outputs

```math
\mu_{u,t}, \mu_{v,t} \in \mathbb{R}^2,
\qquad
\Sigma_{u,t}, \Sigma_{v,t} \in \mathbb{R}^{2 \times 2}.
```

The covariance outputs are parameterized through a Cholesky factor:

```math
\Sigma_{c,t} = L_{c,t} L_{c,t}^\top,
\qquad c \in \{u,v\},
```

so they are positive semidefinite by construction.

Interpretation:

```math
\gamma_{u,t} \sim \mathcal{N}(\mu_{u,t}, \Sigma_{u,t}),
\qquad
\gamma_{v,t} \sim \mathcal{N}(\mu_{v,t}, \Sigma_{v,t}),
```

where `\gamma_{u,t}` and `\gamma_{v,t}` are the stage-2 stochastic advections for the two wind components.

## 3. Theorem-Inspired Stage-2 Kernel

For sites `a,b \in \{1,2,3\}`, define the spatial lag

```math
h_{ab} = s_a - s_b \in \mathbb{R}^2.
```

For component indices `i,j \in \{u,v\}`, define the drift term

```math
d_t^{(ij)} = \Delta t \, (\mu_{i,t} - \mu_{j,t}),
```

with `\Delta t = 10/60` hours in the current code.

We then use the stage-2 dispersion matrix

```math
D_t^{(ij)} = I_2 + 2 \Delta t^2 (\Sigma_{i,t} + \Sigma_{j,t}).
```

This is the theorem-inspired correction that lets the advection covariance control the propagation shape, not only the center.

For each component pair `(i,j)`, define the raw block kernel

```math
K_t^{(ij)}(a,b)
=
\alpha_{ij}
\left|D_t^{(ij)}\right|^{-1/2}
\exp\left(
-
\bigl(h_{ab} - d_t^{(ij)}\bigr)^\top
\left(D_t^{(ij)}\right)^{-1}
\bigl(h_{ab} - d_t^{(ij)}\bigr)
\right),
```

where `\alpha_{ij} > 0` is a learnable block scale.

This gives four `3 \times 3` blocks:

```math
K_t =
\begin{bmatrix}
K_t^{(uu)} & K_t^{(uv)} \\
K_t^{(vu)} & K_t^{(vv)}
\end{bmatrix}
\in \mathbb{R}^{6 \times 6}.
```

The cross-blocks `K_t^{(uv)}` and `K_t^{(vu)}` let the model transfer signal between the two components.

## 4. From Kernel to Forecast Dynamics

The raw kernel matrix is row-normalized to obtain a discrete propagation operator:

```math
\widetilde{M}_t(r,c)
=
\frac{K_t(r,c)}{\sum_{c'=1}^6 K_t(r,c')}.
```

We then mix in an identity term for stability:

```math
M_t = (1-\rho)\widetilde{M}_t + \rho I_6,
```

where `\rho \in (0,1)` is learnable.

So `M_t` is the learned, time-varying propagation matrix induced by the stochastic advections.

The forecasting model then augments this kernel transport with:

- a learned persistence diagonal,
- a kernel-mix coefficient,
- and a direct NWP forcing term.

Let

```math
P = \mathrm{diag}(p_1,\dots,p_6),
\qquad
p_r \in [p_{\min}, p_{\max}],
```

be the learned persistence matrix, and let

```math
k \in [k_{\min}, k_{\max}]
```

be a learned scalar kernel-mix coefficient. The implemented dynamics matrix is

```math
A_t = (1-k)P + k M_t.
```

The NWP encoders also produce a direct forcing term

```math
b_t \in \mathbb{R}^6.
```

So the actual state evolution used in filtering and forecasting is

```math
y_t = A_t y_{t-1} + b_t + \eta_t,
\qquad
\eta_t \sim \mathcal{N}(0, Q).
```

This keeps a persistence backbone, but it no longer lets the model collapse the transport term to an arbitrarily small residual.

## 5. IDE Interpretation

The continuous-space model behind this construction is an IDE of the form

```math
Y_i(s,t)
=
\sum_{j \in \{u,v\}}
\int_{\mathbb{R}^2}
K_t^{(ij)}(s-r)\,
Y_j(r,t-\Delta t)\,
dr

+ \eta_i(s,t),
```

with process noise `\eta_i(s,t)`.

This is the key point:

- `d_t^{(ij)}` shifts the kernel center, so the advection mean determines where mass is transported.
- `D_t^{(ij)}` changes the kernel anisotropy and spread, so the advection covariance determines the propagation shape.

Because we only observe three discrete sites, we replace the spatial integral by a discrete quadrature over the station set:

```math
Y_i(s_a,t)
\approx
\sum_{j \in \{u,v\}}
\sum_{b=1}^3
w_b\,
K_t^{(ij)}(s_a-s_b)\,
Y_j(s_b,t-\Delta t)
+ \eta_i(s_a,t),
```

where the current implementation effectively takes equal quadrature weights and then row-normalizes the full kernel matrix. After stacking the six site-component values, the kernel transport becomes

```math
y_t = M_t y_{t-1} + \eta_t,
\qquad
\eta_t \sim \mathcal{N}(0, Q).
```

The implemented state-space model then replaces `M_t` with the persistence-plus-residual dynamics `A_t` from Section 4 and adds the forcing term `b_t`.

## 6. Process and Measurement Covariances

We use separable cross-covariance matrices for both `Q` and `R`.

Let `B_Q, B_R \in \mathbb{R}^{2 \times 2}` be positive definite core matrices and let

```math
S_\ell(a,b)
=
\exp\left(
-\frac{\|s_a-s_b\|^2}{2\ell^2}
\right).
```

Then

```math
Q = B_Q \otimes S_{\ell_Q},
\qquad
R = B_R \otimes S_{\ell_R},
```

up to the small diagonal jitter used for numerical stability.

This construction does two things at once:

- the spatial kernel `S_\ell` models dependence across stations
- the `2 \times 2` core matrix models direct and cross dependence between `u` and `v`

## 7. Initial State

The model uses

```math
y_0 \sim \mathcal{N}(m_0, P_0),
```

where `m_0` is learnable and `P_0` is initialized as a diagonal positive matrix.

## 8. Likelihood and Hybrid Training

Given one training window

```math
(x^u_{1:T}, x^v_{1:T}, z_{1:T}),
```

the model performs the following steps:

1. For each time `t`, predict

```math
(\mu_{u,t}, \Sigma_{u,t}) = f_{\theta_u}(x^u_t),
\qquad
(\mu_{v,t}, \Sigma_{v,t}) = f_{\theta_v}(x^v_t),
\qquad
b_t = f_{\theta_b}(x^u_t, x^v_t).
```

2. Build `K_t`, then `M_t`, then `A_t = (1-k)P + k M_t`.
3. Run the Kalman filter for

```math
y_t = A_t y_{t-1} + b_t + \eta_t,
\qquad
z_t = y_t + \varepsilon_t.
```

4. Compute the negative log-likelihood:

```math
\mathcal{L}_{\mathrm{nll}}
=
\frac{1}{T}
\sum_{t=1}^T
\frac{1}{2}
\left[
\log |F_t|
+
v_t^\top F_t^{-1} v_t
+
6 \log(2\pi)
\right],
```

where

```math
v_t = z_t - \hat{y}_{t|t-1},
\qquad
F_t = P_{t|t-1} + R.
```

To better align training with open-loop prediction, the implementation optimizes a hybrid objective

```math
\mathcal{J}
=
\lambda_{\mathrm{nll}} \mathcal{L}_{\mathrm{nll}}
+ \lambda_{1} \mathcal{L}_{\mathrm{1step}}
+ \lambda_{H} \mathcal{L}_{\mathrm{rollout}},
```

where the one-step term is a Smooth L1 loss on the pre-update predictive mean

```math
\mathcal{L}_{\mathrm{1step}}
=
\frac{1}{T-1}
\sum_{t=2}^{T}
\mathrm{SmoothL1}(\hat{y}_{t|t-1}, z_t),
```

and the rollout term compares an open-loop forecast over the last `H` steps of the window:

```math
\mathcal{L}_{\mathrm{rollout}}
=
\frac{1}{H}
\sum_{h=1}^{H}
\mathrm{SmoothL1}(\hat{y}^{\mathrm{roll}}_{T-H+h}, z_{T-H+h}).
```

All parameters are trained end-to-end by minimizing `\mathcal{J}`:

```math
\Theta
=
\{\theta_u, \theta_v, \theta_b, \alpha_{ij}, \rho, p, g, B_Q, \ell_Q, B_R, \ell_R, m_0, P_0\}.
```

## 9. How Advection Is Trained

There is no direct supervision for the advection mean or covariance.

Instead, the advection networks are trained implicitly through the state-space likelihood:

```math
(x^u_t, x^v_t)
\rightarrow
\bigl(\mu_{u,t}, \Sigma_{u,t}, \mu_{v,t}, \Sigma_{v,t}, b_t\bigr)
\rightarrow
K_t
\rightarrow
\bigl(M_t, A_t\bigr)
\rightarrow
\mathcal{J}(z_{1:T}).
```

So the model learns advections that make the induced propagation operator improve both state-space likelihood and direct forecast accuracy.

This means:

- if the NWP suggests a directional transport pattern that improves one-step predictive likelihood, the corresponding advection mean is reinforced
- if uncertainty in the transport should broaden or deform the propagation footprint, the advection covariance learns that through `D_t^{(ij)}`

In other words, advection is learned as a latent transport mechanism, not as a separately labeled regression target.

More explicitly, the gradient path is

```math
\nabla_{\theta_u}\mathcal{J}
=
\sum_{t=1}^T
\frac{\partial \mathcal{J}}{\partial A_t}
\frac{\partial A_t}{\partial M_t}
\frac{\partial M_t}{\partial K_t}
\left(
\sum_{i,j \in \{u,v\}}
\frac{\partial K_t^{(ij)}}{\partial d_t^{(ij)}}
\frac{\partial d_t^{(ij)}}{\partial \mu_{u,t}}
+
\frac{\partial K_t^{(ij)}}{\partial D_t^{(ij)}}
\frac{\partial D_t^{(ij)}}{\partial \Sigma_{u,t}}
\right)
\frac{\partial (\mu_{u,t},\Sigma_{u,t})}{\partial \theta_u},
```

and the same structure holds for `\theta_v`. The forcing head receives gradients directly from the one-step and rollout forecast terms.

## 10. DDP Training

The implementation supports real single-node multi-GPU training through PyTorch DDP.

Launch it with:

```bash
torchrun --standalone --nproc_per_node=2 -m scripts.train --config configs/default.yaml
```

In DDP mode:

- each process gets its own GPU
- each process sees a disjoint shard of the training windows through `DistributedSampler`
- gradients are synchronized by `DistributedDataParallel`
- only rank `0` prints logs and writes checkpoints

The `batch_size` in the config is the per-GPU batch size, so the effective global batch size is

```math
\text{global batch size} = \text{world size} \times \text{per-GPU batch size}.
```

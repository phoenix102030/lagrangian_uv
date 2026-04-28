# Stage-2 Formulation: Site-Level IDE Kernel for UV Wind

This note describes the model implemented in `lagrangian_uv_stage2`.
The goal is a complete time-varying IDE transition kernel over three
observation sites and two wind components, not a persistence model and not a
residual correction around persistence.

## 1. State And Observation

Let the three sites be

```math
s_1, s_2, s_3 \in \mathbb{R}^2.
```

The latent state is component-first:

```math
y_t =
\begin{bmatrix}
u_t(s_1) & u_t(s_2) & u_t(s_3) &
v_t(s_1) & v_t(s_2) & v_t(s_3)
\end{bmatrix}^{\top}
\in \mathbb{R}^6.
```

The measurement vector uses the same order:

```math
z_t = y_t + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0, R).
```

So the observation matrix is the identity, `H = I_6`.

## 2. NWP-Conditioned Site Parameters

At each time step, the selected NWP `u` and `v` grids are concatenated:

```math
x_t = [x^u_t, x^v_t].
```

A CNN extracts a background feature map, station coordinates are sampled from
that map with bilinear `grid_sample`, and a Transformer lets the three station
tokens exchange spatial information:

```math
\mathcal{F}_t = \mathrm{CNN}_{\theta}(x_t),
```

```math
f_t(s_a) = \mathrm{GridSample}(\mathcal{F}_t, s_a),
```

```math
[h_t(s_1), h_t(s_2), h_t(s_3)]
= \mathrm{Transformer}([f_t(s_1), f_t(s_2), f_t(s_3)] + E_{\mathrm{site}}).
```

The network emits site-level advection and dispersion parameters for both
components:

```math
\mu_t(s_a, c) \in \mathbb{R}^2,
\qquad
\Sigma_t(s_a, c) \in \mathbb{R}^{2 \times 2},
\qquad c \in \{u, v\}.
```

The implementation can also emit a per-site joint covariance over
`[u_x, u_y, v_x, v_y]`, which is used to shape cross-component blocks when
`allow_cross_component: true`.

## 3. Open-System IDE Kernel

For a source state `(source component j, source site b)` and a target state
`(target component i, target site a)`, the kernel entry is

```math
K_t[(i,a),(j,b)]
= \alpha_{ij}\gamma\,
\exp\left(
-\frac{1}{2}
\Delta x_{ab}^{(ij)\top}
D_t(s_b, i, j)^{-1}
\Delta x_{ab}^{(ij)}
\right),
```

with component coupling scale `alpha_ij`, fixed open-system decay `gamma`, and

```math
\Delta x_{ab}^{(ij)}
= s_a - s_b - d_t(s_b, i, j).
```

For same-component transport:

```math
d_t(s_b, j, j) = \Delta t\,\mu_t(s_b, j),
```

```math
D_t(s_b, j, j)
= I_2 + 2\Delta t^2\Sigma_t(s_b, j).
```

For cross-component transport:

```math
d_t(s_b, i, j)
= \Delta t\,(\mu_t(s_b, i) - \mu_t(s_b, j)),
```

```math
D_t(s_b, i, j)
= I_2 + 2\Delta t^2
\left(
\Sigma_t(s_b, i)
+ \Sigma_t(s_b, j)
- \Sigma_t^{ij}(s_b)
- \Sigma_t^{ji}(s_b)
\right).
```

If no joint covariance is supplied, the cross-covariance terms are omitted.

The kernel is open-system: rows and columns are not normalized. If the learned
advection moves mass away from all three monitored sites, the corresponding
entries naturally shrink instead of being forced back into the observed domain.

## 4. State Dynamics

The state transition is the IDE kernel itself:

```math
A_t = K_t.
```

No learned persistence matrix is mixed in, and the kernel is not treated as a
residual around the identity. The optional forcing head is disabled by default;
when enabled it adds a direct NWP-conditioned bias:

```math
y_t = A_t y_{t-1} + b_t + \eta_t,
\qquad
\eta_t \sim \mathcal{N}(0, Q).
```

With the default config, `b_t = 0`, so the forecast motion comes from the IDE
operator.

## 5. Kalman Filtering

Prediction:

```math
\hat{y}_{t|t-1} = A_t\hat{y}_{t-1|t-1} + b_t,
```

```math
P_{t|t-1} = A_tP_{t-1|t-1}A_t^{\top} + Q.
```

Update:

```math
S_t = P_{t|t-1} + R,
```

```math
G_t = P_{t|t-1}S_t^{-1},
```

```math
\hat{y}_{t|t}
= \hat{y}_{t|t-1} + G_t(z_t - \hat{y}_{t|t-1}).
```

The covariance update uses the Joseph form:

```math
P_{t|t}
= (I-G_t)P_{t|t-1}(I-G_t)^{\top} + G_tRG_t^{\top}.
```

`Q` and `R` are separable cross-component/site covariances:

```math
Q = B_Q \otimes S_{\ell_Q},
\qquad
R = B_R \otimes S_{\ell_R}.
```

## 6. Training Objective

Training is end-to-end. Gradients flow through

```math
x_t
\rightarrow
(\mu_t,\Sigma_t)
\rightarrow
K_t
\rightarrow
\mathrm{Kalman\ filter}
\rightarrow
\mathcal{J}.
```

The objective combines calibrated filtering and forecast accuracy:

```math
\mathcal{J}
= \lambda_{\mathrm{nll}}\mathcal{L}_{\mathrm{nll}}
+ \lambda_1\mathcal{L}_{\mathrm{1step}}
+ \lambda_H\mathcal{L}_{\mathrm{rollout}}
+ \lambda_K\mathcal{L}_{\mathrm{kernel\ 1step}}.
```

The persistence forecast appears only as an evaluation baseline in diagnostics;
it is not part of the model dynamics.

# Graph

The graph experiments are based on the experimental setup of **Node-Level Differentially Private Graph Neural Networks**.

Their implementation is written in JAX, so this project provides a JAX/Optax implementation of TCG in:

```text
../optimizer/tcg_jax.py
```

For graph experiments, keep the node-level DP gradient computation, clipping, noise addition, and privacy accounting from the JAX graph training code. Then replace the base DP optimizer with the JAX TCG wrapper.

Example usage inside the graph optimizer builder:

```python
from optimizer.tcg_jax import tcg_adam

aggregate_tx = dp_aggregate(
    l2_norms_threshold=l2_norms_threshold,
    base_sensitivity=base_sensitivity,
    noise_multiplier=noise_multiplier,
    init_rng=init_rng,
)

optimizer = tcg_adam(
    learning_rate=learning_rate,
    aggregate_tx=aggregate_tx,
    beta_fast=0.8,
    beta_slow=0.98,
    gate_type="hard",
    gate_tau=0.0,
)
```

Use `gate_type="hard"` for hard gating or `gate_type="soft"` for soft gating. For
soft gating, `gate_tau` controls the temperature.

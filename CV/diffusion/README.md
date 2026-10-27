# Diffusion

This part is based on the DP implementation for diffusion models from **DP-Promise: Differentially Private Diffusion Probabilistic Models for Image Synthesis**.

We do not use the DP-Promise method itself. We only refer to its implementation strategy for adding differential privacy to diffusion training, and then apply our `TCG` optimizer wrapper:

```python
from optimizer import TCG
```

In practice, wrap the base optimizer used in private diffusion training with:

```python
optimizer = TCG(base_optimizer=base_optimizer)
```

The DP clipping, noise addition, and privacy accounting are handled by the DP diffusion implementation; TCG only gates the coordinate-wise update produced by the base DP optimizer.

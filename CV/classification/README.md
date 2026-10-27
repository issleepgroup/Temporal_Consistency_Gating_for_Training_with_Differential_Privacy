# CIFAR10 Classification

This script logs results to W&B, so please set up W&B before running.

Run DP-SGD:

```bash
python cifar10.py --base-opt sgd --method baseline
```

Run DP-SGD + TCG:

```bash
python cifar10.py --base-opt sgd --method tcg
```

Run the four-layer CNN example for both methods at `epsilon=1`:

```bash
bash example.sh
```

Common options:

```text
--arch                 cnn | resnet18
--base-opt             adam | sgd
--method               baseline | tcg
--batch-size           training batch size
--batch-size-test      evaluation batch size
--lr-adam              Adam learning rate
--lr-sgd               SGD learning rate
--beta-fast            TCG fast EMA decay
--beta-slow            TCG slow EMA decay
--gate-type            TCG gate type: hard | soft
--gate-tau             temperature used by the soft gate
```

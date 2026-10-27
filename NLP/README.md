# NLP

The NLP code is based on the **Fast Differential Privacy** library.

This script logs results to W&B, so please set up W&B before running.

Run DP-Adam baseline:

```bash
python bert.py --method baseline
```

Run DP-Adam + TCG:

```bash
python bert.py --method tcg
```

Run the SST-2 BERT-base example for both methods with three training epochs,
clip norm `0.4`, and `epsilon=1`:

```bash
bash example.sh
```

Common options:

```text
--model-name-or-path   bert-base-uncased | bert-large-uncased | roberta-base | roberta-large
--method               baseline | tcg
--batch-size           logical training batch size
--physical-batch-size  per-step physical batch size
--learning-rate        AdamW learning rate
--beta-fast            TCG fast EMA decay
--beta-slow            TCG slow EMA decay
--gate-type            TCG gate type: hard | soft
--gate-tau             temperature used by the soft gate
```

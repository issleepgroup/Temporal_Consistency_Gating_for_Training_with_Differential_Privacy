import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.optim as optim
import wandb
from datasets import load_dataset
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NLP_ROOT = Path(__file__).resolve().parent
for path in (PROJECT_ROOT, NLP_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from optimizer import TCG
from fastDP import PrivacyEngine


TRUE_TAGS = {"yes", "y", "true", "t", "1"}


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_task_name(task_name: str) -> str:
    task_name = task_name.lower()
    if task_name in {"sst2", "sst-2"}:
        return "sst-2"
    if task_name == "qnli":
        return "qnli"
    raise ValueError("Only sst-2 and qnli are supported.")


def default_epochs(task_name: str) -> int:
    return 3 if task_name == "sst-2" else 10


def default_batch_size(task_name: str) -> int:
    return 256


def default_learning_rate(non_private: bool, bias_only: bool) -> float:
    return 5e-4


def text_fields(task_name: str) -> Tuple[str, str]:
    if task_name == "sst-2":
        return "sentence", None
    if task_name == "qnli":
        return "question", "sentence"
    raise ValueError(task_name)


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == labels).float().mean().item()


def mean_of_last_k(xs, k: int):
    if len(xs) == 0:
        return None
    k = min(k, len(xs))
    return float(np.mean(xs[-k:]))


def get_inner_optimizer(optimizer):
    cur = optimizer
    while hasattr(cur, "base_optimizer"):
        cur = cur.base_optimizer
    return cur


def maybe_freeze_parameters(model, args):
    attention_only = args.attention_only.lower() in TRUE_TAGS
    bias_only = args.bias_only.lower() in TRUE_TAGS
    static_embedding = args.static_embedding.lower() in TRUE_TAGS

    if attention_only:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if "query" in name or "value" in name or "classifier" in name or "lm_head" in name:
                param.requires_grad_(True)
    elif bias_only:
        for name, param in model.named_parameters():
            if ".bias" not in name:
                param.requires_grad_(False)
    else:
        model.requires_grad_(True)
        if static_embedding:
            model.get_input_embeddings().requires_grad_(False)


def build_optimizer(model, args):
    named_params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in named_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in named_params if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    base = Adam(
        optimizer_grouped_parameters,
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.adam_epsilon,
    )

    if args.method == "baseline":
        return base
    if args.method == "tcg":
        return TCG(
            base_optimizer=base,
            beta_fast=args.beta_fast,
            beta_slow=args.beta_slow,
            gate_type=args.gate_type,
            gate_tau=args.gate_tau,
        )

    raise ValueError(f"Unknown method: {args.method}")


def build_output_dir(args) -> Path:
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    out = root / args.task_name / args.noise_mechanism / group_name(args)
    out.mkdir(parents=True, exist_ok=True)
    return out


def group_name(args):
    raw = (
        f"task={args.task_name}|model={args.model_name_or_path}|init={args.init_mode}|"
        f"method={args.method}|noise={args.noise_mechanism}|eps={args.target_epsilon}|"
        f"lr={args.learning_rate}|epochs={args.num_train_epochs}|batch={args.batch_size}|"
        f"bf={args.beta_fast}|bs={args.beta_slow}|gate={args.gate_type}|"
        f"tau={args.gate_tau}|attn={args.attention_only}|"
        f"bias={args.bias_only}|staticemb={args.static_embedding}"
    )
    short = (
        f"{args.task_name}"
        f"__{args.init_mode}"
        f"__{args.method}"
        f"__{args.noise_mechanism}"
        f"__eps{args.target_epsilon}"
        f"__ep{args.num_train_epochs}"
    )
    h = hashlib.md5(raw.encode()).hexdigest()[:8]
    return f"{short}__{h}"


def config_for_wandb(args, seed: int) -> Dict:
    return {
        "task_name": args.task_name,
        "model_name_or_path": args.model_name_or_path,
        "model_init": args.init_mode,
        "base_opt": "adam",
        "method": args.method,
        "noise_mechanism": args.noise_mechanism,
        "seed": seed,
        "num_train_epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "physical_batch_size": args.physical_batch_size,
        "target_epsilon": args.target_epsilon,
        "target_delta": args.target_delta,
        "per_example_max_grad_norm": args.per_example_max_grad_norm,
        "learning_rate": args.learning_rate,
        "eval_steps": args.eval_steps,
        "max_seq_len": args.max_seq_len,
        "clipping_mode": args.clipping_mode,
        "clipping_fn": args.clipping_fn,
        "clipping_style": args.clipping_style,
        "attention_only": args.attention_only,
        "bias_only": args.bias_only,
        "static_embedding": args.static_embedding,
        "plrvo_k": args.plrvo_k,
        "plrvo_theta": args.plrvo_theta,
        "plrvo_use_gamma": args.plrvo_use_gamma,
        "gate_type": args.gate_type,
        "gate_tau": args.gate_tau,
    }


def build_plrv_config_dict(args) -> Dict:
    cfg = {
        "theta": float(args.plrvo_theta),
        "gamma": bool(args.plrvo_use_gamma),
        "max_grad_norm": float(args.per_example_max_grad_norm),
    }
    if args.plrvo_use_gamma:
        cfg["k"] = float(args.plrvo_k)
    return cfg


def maybe_make_temp_plrv_config(args, out_dir: Path) -> str:
    if args.noise_mechanism != "plrvo":
        return ""
    cfg = build_plrv_config_dict(args)
    tmp_path = out_dir / "auto_plrv_config.json"
    tmp_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return str(tmp_path)


def build_dataloaders(args, tokenizer):
    subset = "sst2" if args.task_name == "sst-2" else "qnli"
    raw = load_dataset("glue", subset)

    field1, field2 = text_fields(args.task_name)

    def preprocess(batch):
        if field2 is None:
            return tokenizer(
                batch[field1],
                truncation=True,
                max_length=args.max_seq_len,
            )
        return tokenizer(
            batch[field1],
            batch[field2],
            truncation=True,
            max_length=args.max_seq_len,
        )

    tokenized = raw.map(preprocess, batched=True)

    keep_cols = ["input_ids", "attention_mask", "label"]
    if "token_type_ids" in tokenized["train"].column_names:
        keep_cols.append("token_type_ids")

    tokenized = tokenized.remove_columns(
        [c for c in tokenized["train"].column_names if c not in keep_cols]
    )
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch")

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_loader = DataLoader(
        tokenized["train"],
        batch_size=args.physical_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        tokenized["validation"],
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.workers,
        pin_memory=True,
    )

    return tokenized, train_loader, eval_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = []
    accs = []

    for batch in tqdm(loader, leave=False):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        losses.append(outputs.loss.item())
        accs.append(accuracy_from_logits(outputs.logits, batch["labels"]))

    return {
        "eval_loss": float(np.mean(losses)),
        "eval_acc": float(np.mean(accs)),
    }


def build_model(args):
    if args.init_mode == "random":
        config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=2)
        return AutoModelForSequenceClassification.from_config(config)
    if args.init_mode == "pretrained":
        return AutoModelForSequenceClassification.from_pretrained(
            args.model_name_or_path,
            num_labels=2,
        )
    raise ValueError(f"Unknown init_mode: {args.init_mode}")


def run_single_seed(args, seed: int):
    set_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = build_output_dir(args)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=False)
    model = build_model(args)

    maybe_freeze_parameters(model, args)
    model = model.to(device)

    tokenized, train_loader, eval_loader = build_dataloaders(args, tokenizer)
    optimizer = build_optimizer(model, args)

    total_train_batch_size = args.batch_size
    if args.target_delta is None:
        args.target_delta = 1.0 / len(tokenized["train"])

    privacy_engine = None
    if not args.non_private:
        origin_params = None if (
            args.bias_only.lower() in TRUE_TAGS or args.attention_only.lower() in TRUE_TAGS
        ) else ["_embeddings"]

        resolved_plrv_config = maybe_make_temp_plrv_config(args, out_dir)

        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=total_train_batch_size,
            sample_size=len(tokenized["train"]),
            epochs=args.num_train_epochs,
            max_grad_norm=args.per_example_max_grad_norm,
            noise_multiplier=None,
            target_epsilon=args.target_epsilon,
            target_delta=args.target_delta,
            accounting_mode=args.accounting_mode,
            clipping_mode=args.clipping_mode,
            clipping_fn=args.clipping_fn,
            clipping_style=args.clipping_style,
            origin_params=origin_params,
            plrv=(args.noise_mechanism == "plrvo"),
            plrv_config=resolved_plrv_config,
        )
        privacy_engine.attach(optimizer)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = steps_per_epoch * args.num_train_epochs
    scheduler = LambdaLR(
        get_inner_optimizer(optimizer),
        lr_lambda=lambda step: max(0.0, 1.0 - step / max(total_steps, 1)),
    )

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        group=group_name(args),
        job_type=args.method,
        name=f"{group_name(args)}|seed{seed}",
        tags=["nlp", "glue", args.init_mode, args.task_name, args.noise_mechanism, args.method],
        config=config_for_wandb(args, seed),
        reinit="finish_previous",
    )

    (out_dir / "argparse.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    global_step = 0
    best_eval_acc = -1.0
    best_metrics = None

    eval_acc_history = []
    eval_loss_history = []
    train_acc_history = []
    train_loss_history = []
    gate_mean_history = []

    for epoch in range(1, args.num_train_epochs + 1):
        model.train()
        running_losses = []
        running_accs = []

        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, leave=False), start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            running_losses.append(outputs.loss.item())
            running_accs.append(accuracy_from_logits(outputs.logits.detach(), batch["labels"]))

            if step % args.gradient_accumulation_steps == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step % args.eval_steps == 0:
                    eval_stats = evaluate(model, eval_loader, device)

                    train_stats = {
                        "train_loss": float(np.mean(running_losses)),
                        "train_acc": float(np.mean(running_accs)),
                    }

                    row = {
                        "epoch": epoch,
                        "global_step": global_step,
                        **train_stats,
                        **eval_stats,
                    }

                    if hasattr(optimizer, "current_gate_mean"):
                        row["gate_mean"] = float(optimizer.current_gate_mean)

                    if privacy_engine is not None:
                        try:
                            privacy_spent = privacy_engine.get_privacy_spent(
                                accounting_mode="rdp",
                                lenient=True,
                            )
                            for k, v in privacy_spent.items():
                                if isinstance(v, (int, float)):
                                    row[k] = float(v)
                        except Exception:
                            pass

                    eval_acc_history.append(row["eval_acc"])
                    eval_loss_history.append(row["eval_loss"])
                    train_acc_history.append(row["train_acc"])
                    train_loss_history.append(row["train_loss"])
                    if "gate_mean" in row:
                        gate_mean_history.append(row["gate_mean"])

                    wandb.log(row)

                    if eval_stats["eval_acc"] > best_eval_acc:
                        best_eval_acc = eval_stats["eval_acc"]
                        best_metrics = row.copy()
                        model.save_pretrained(out_dir / "best_model")
                        tokenizer.save_pretrained(out_dir / "best_model")

        eval_stats = evaluate(model, eval_loader, device)
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": float(np.mean(running_losses)),
            "train_acc": float(np.mean(running_accs)),
            **eval_stats,
        }

        if hasattr(optimizer, "current_gate_mean"):
            row["gate_mean"] = float(optimizer.current_gate_mean)

        if privacy_engine is not None:
            try:
                privacy_spent = privacy_engine.get_privacy_spent(
                    accounting_mode="rdp",
                    lenient=True,
                )
                for k, v in privacy_spent.items():
                    if isinstance(v, (int, float)):
                        row[k] = float(v)
            except Exception:
                pass

        eval_acc_history.append(row["eval_acc"])
        eval_loss_history.append(row["eval_loss"])
        train_acc_history.append(row["train_acc"])
        train_loss_history.append(row["train_loss"])
        if "gate_mean" in row:
            gate_mean_history.append(row["gate_mean"])

        wandb.log(row)

        if eval_stats["eval_acc"] > best_eval_acc:
            best_eval_acc = eval_stats["eval_acc"]
            best_metrics = row.copy()
            model.save_pretrained(out_dir / "best_model")
            tokenizer.save_pretrained(out_dir / "best_model")

    final_results = {
        "best_eval_acc": best_eval_acc,
        "best_metrics": best_metrics,
    }
    (out_dir / "final_results.json").write_text(
        json.dumps(final_results, indent=2),
        encoding="utf-8",
    )

    run.summary["best_eval_acc"] = best_eval_acc
    if best_metrics is not None:
        for k, v in best_metrics.items():
            if isinstance(v, (int, float)):
                run.summary[f"best_{k}"] = v

    if len(eval_acc_history) > 0:
        run.summary["final_eval_acc"] = float(eval_acc_history[-1])
        run.summary["last10_eval_acc_mean"] = mean_of_last_k(eval_acc_history, 10)

    if len(eval_loss_history) > 0:
        run.summary["final_eval_loss"] = float(eval_loss_history[-1])
        run.summary["last10_eval_loss_mean"] = mean_of_last_k(eval_loss_history, 10)

    if len(train_acc_history) > 0:
        run.summary["final_train_acc"] = float(train_acc_history[-1])
        run.summary["last10_train_acc_mean"] = mean_of_last_k(train_acc_history, 10)

    if len(train_loss_history) > 0:
        run.summary["final_train_loss"] = float(train_loss_history[-1])

    if len(gate_mean_history) > 0:
        run.summary["last10_gate_mean"] = mean_of_last_k(gate_mean_history, 10)


    wandb.finish()


def parse_args():
    parser = argparse.ArgumentParser(description="GLUE NLP training with fastDP + wandb")

    parser.add_argument("--task-name", type=str, default="sst-2", choices=["sst-2", "sst2", "qnli"])
    parser.add_argument(
        "--model-name-or-path",
        type=str,
        default="bert-base-uncased",
        choices=[
            "bert-base-uncased",
            "bert-large-uncased",
            "roberta-base",
            "roberta-large",
        ],
    )
    parser.add_argument("--init-mode", type=str, default="pretrained", choices=["random", "pretrained"])

    parser.add_argument("--method", choices=["baseline", "tcg"], default="baseline")
    parser.add_argument("--noise-mechanism", choices=["gaussian", "plrvo"], default="gaussian")

    parser.add_argument("--output-dir", type=str, default="./results")
    parser.add_argument("--wandb-project", type=str, default="dp-nlp-glue")
    parser.add_argument("--wandb-entity", type=str, default="")

    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--physical-batch-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=256)

    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--target-epsilon", type=float, default=1.0)
    parser.add_argument("--target-delta", type=float, default=1e-5)
    parser.add_argument("--per-example-max-grad-norm", type=float, default=0.4)
    parser.add_argument("--accounting-mode", type=str, default="rdp")
    parser.add_argument("--clipping-mode", type=str, default="ghost")
    parser.add_argument("--clipping-fn", type=str, default="Abadi")
    parser.add_argument("--clipping-style", type=str, default="all-layer")

    parser.add_argument("--plrvo-theta", type=float, default=None)
    parser.add_argument("--plrvo-k", type=float, default=None)
    parser.add_argument("--plrvo-use-gamma", action="store_true", default=False)

    parser.add_argument("--beta-fast", type=float, default=0.8)
    parser.add_argument("--beta-slow", type=float, default=0.98)
    parser.add_argument("--gate-type", choices=["hard", "soft"], default="soft")
    parser.add_argument("--gate-tau", type=float, default=0.05)

    parser.add_argument("--attention-only", type=str, default="no")
    parser.add_argument("--bias-only", type=str, default="no")
    parser.add_argument("--static-embedding", type=str, default="no")
    parser.add_argument("--non-private", action="store_true", default=False)

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main():
    args = parse_args()
    args.task_name = normalize_task_name(args.task_name)

    if args.batch_size is None:
        args.batch_size = default_batch_size(args.task_name)
    if args.num_train_epochs is None:
        args.num_train_epochs = default_epochs(args.task_name)

    bias_only = args.bias_only.lower() in TRUE_TAGS
    if args.learning_rate is None:
        args.learning_rate = default_learning_rate(
            non_private=args.non_private,
            bias_only=bias_only,
        )

    if args.gradient_accumulation_steps is None:
        if args.batch_size % args.physical_batch_size != 0:
            raise ValueError("batch_size must be divisible by physical_batch_size")
        args.gradient_accumulation_steps = args.batch_size // args.physical_batch_size

    if args.noise_mechanism == "plrvo":
        if args.plrvo_theta is None:
            raise ValueError("PLRVO requires --plrvo-theta")
        if args.plrvo_use_gamma and args.plrvo_k is None:
            raise ValueError("When --plrvo-use-gamma is set, you must also provide --plrvo-k")

    run_single_seed(args, args.seed)


if __name__ == "__main__":
    main()

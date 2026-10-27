from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax


ALIGNMENT_EPS = 1e-8


class TCGState(NamedTuple):
    aggregate_state: optax.OptState
    base_state: optax.OptState
    m_fast: optax.Updates
    m_slow: optax.Updates
    last_gate_mean: jnp.ndarray


def _zeros_like_tree(tree):
    return jax.tree.map(jnp.zeros_like, tree)


def _ones_like_tree(tree):
    return jax.tree.map(jnp.ones_like, tree)


def _tree_sum(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return sum(jnp.sum(x) for x in leaves)


def _tree_size(tree):
    leaves = jax.tree.leaves(tree)
    if not leaves:
        return jnp.asarray(0.0)
    return sum(jnp.asarray(x.size, dtype=jnp.float32) for x in leaves)


def tcg(
    aggregate_tx: optax.GradientTransformation,
    base_tx: optax.GradientTransformation,
    beta_fast: float = 0.8,
    beta_slow: float = 0.98,
    gate_type: str = "hard",
    gate_tau: float = 0.0,
) -> optax.GradientTransformation:
    if not 0.0 <= beta_fast < beta_slow < 1.0:
        raise ValueError("Expected 0 <= beta_fast < beta_slow < 1")
    if gate_type not in {"hard", "soft"}:
        raise ValueError("gate_type must be either 'hard' or 'soft'")
    if gate_type == "soft" and gate_tau <= 0.0:
        raise ValueError("gate_tau must be > 0")

    def init_fn(params):
        return TCGState(
            aggregate_state=aggregate_tx.init(params),
            base_state=base_tx.init(params),
            m_fast=_zeros_like_tree(params),
            m_slow=_zeros_like_tree(params),
            last_gate_mean=jnp.asarray(1.0, dtype=jnp.float32),
        )

    def update_fn(updates, state, params=None):
        privatized_grads, aggregate_state = aggregate_tx.update(
            updates, state.aggregate_state, params
        )

        consistency_score = jax.tree.map(
            lambda fast, slow: (fast * slow)
            / (jnp.abs(fast) * jnp.abs(slow) + ALIGNMENT_EPS),
            state.m_fast,
            state.m_slow,
        )
        support_score = jax.tree.map(
            lambda grad, slow: (grad * slow)
            / (jnp.abs(grad) * jnp.abs(slow) + ALIGNMENT_EPS),
            privatized_grads,
            state.m_slow,
        )
        score = jax.tree.map(jnp.maximum, consistency_score, support_score)

        if gate_type == "hard":
            gate = jax.tree.map(lambda s: (s > 0).astype(s.dtype), score)
        else:
            gate = jax.tree.map(lambda s: jax.nn.sigmoid(s / gate_tau), score)

        base_updates, base_state = base_tx.update(
            privatized_grads, state.base_state, params
        )
        gated_updates = jax.tree.map(
            lambda w, update: w * update, gate, base_updates
        )
        last_gate_mean = _tree_sum(gate) / jnp.maximum(_tree_size(gate), 1.0)

        m_fast = jax.tree.map(
            lambda old, grad: beta_fast * old + (1.0 - beta_fast) * grad,
            state.m_fast,
            privatized_grads,
        )
        m_slow = jax.tree.map(
            lambda old, grad: beta_slow * old + (1.0 - beta_slow) * grad,
            state.m_slow,
            privatized_grads,
        )

        new_state = TCGState(
            aggregate_state=aggregate_state,
            base_state=base_state,
            m_fast=m_fast,
            m_slow=m_slow,
            last_gate_mean=last_gate_mean,
        )
        return gated_updates, new_state

    return optax.GradientTransformation(init_fn, update_fn)


def gate_mean(state: TCGState) -> jnp.ndarray:
    return state.last_gate_mean


def tcg_sgd(
    learning_rate,
    aggregate_tx,
    momentum=None,
    nesterov=False,
    beta_fast=0.8,
    beta_slow=0.98,
    gate_type="hard",
    gate_tau=0.0,
):
    base_tx = optax.sgd(learning_rate, momentum, nesterov)
    return tcg(
        aggregate_tx=aggregate_tx,
        base_tx=base_tx,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        gate_type=gate_type,
        gate_tau=gate_tau,
    )


def tcg_adam(
    learning_rate,
    aggregate_tx,
    beta_fast=0.8,
    beta_slow=0.98,
    gate_type="hard",
    gate_tau=0.0,
):
    base_tx = optax.adam(learning_rate)
    return tcg(
        aggregate_tx=aggregate_tx,
        base_tx=base_tx,
        beta_fast=beta_fast,
        beta_slow=beta_slow,
        gate_type=gate_type,
        gate_tau=gate_tau,
    )

from time import time
from functools import partial
from typing import Any, Mapping, Tuple, List

import logging

import jax
from jax import Array
from jax import numpy as jnp
import optax

from xjax.signals import train_epoch_completed, train_epoch_started
from xjax.tools import default_arg

__all__ = [
    "char_rnn",
    "train",
    "predict",
    "generate",
    "CharRNN",
    "_sample",
    "_loss",
    "_step",
]


Parameters = Mapping[str, Any]

logger = logging.getLogger(__name__)


class CharRNN:

    def __call__(self, params, h, X) -> Tuple[Array]:

        Whx, Whh, Wyh, bh, by = params

        Y_logits = jnp.zeros_like(X)

        # iterate over the input sequence
        for i in range(X.shape[0]):

            # pick out the current input token one-hot
            x = jnp.expand_dims(X[i, :], 1)

            # update the hidden state
            h = jnp.tanh(jnp.dot(Whx, x) + jnp.dot(Whh, h) + bh)

            # compute the output logits
            logits = jnp.dot(Wyh, h) + by
            logits = jnp.squeeze(logits)

            # update the output sequence
            Y_logits = Y_logits.at[i].set(logits)

        # return the updated hidden state and output logits
        return h, Y_logits


def char_rnn(rng: Array, vocab_size: int, hidden_size: int):

    # Initialize params
    Whx = jax.random.normal(rng, shape=(hidden_size, vocab_size))

    k1, k2 = jax.random.split(rng, 2)
    Whh = jax.random.normal(k1, shape=(hidden_size, hidden_size))
    bh = jnp.zeros((hidden_size, 1))

    Wyh = jax.random.normal(k2, shape=(vocab_size, hidden_size))
    by = jnp.zeros((vocab_size, 1))

    # Initialize model
    model = CharRNN()

    return (Whx, Whh, Wyh, bh, by), model


def generate(
    rng: Array, prefix: List[int], params: Parameters, hidden_size: int, vocab_size: int
) -> List[int]:

    # initialize the model and hidden state
    model = CharRNN()
    h = jnp.zeros((hidden_size, 1))

    # create the initial input vector
    result = []
    x = jnp.zeros((1, vocab_size))
    for i in prefix:
        result.append(i)
        x = x.at[0, i].set(1)

    idx_y = prefix[-1]

    # Assume that the last letter in the vocab is the stop character
    # and terminate if the stop character is generated
    while idx_y != vocab_size - 1:

        rng, sub_rng = jax.random.split(rng)

        h, y = predict(
            model, rng=sub_rng, params=params, vocab_size=vocab_size, h=h, x=x
        )

        # get the predicted character
        idx_y = int(jnp.argmax(y))

        # update the input vector
        x = jnp.zeros((1, vocab_size))
        x = x.at[0, idx_y].set(1)

        # look up the character and add to final result
        result.append(idx_y)

    return result


def predict(
    model: CharRNN,
    *,
    rng: Array,
    params: Parameters,
    vocab_size: int,
    h: Array,
    x: Array,
):

    h, y_logits = model(params, h, x)
    y_pred = jax.nn.softmax(y_logits, axis=1)

    # randomly select an output token based on the probabilities
    idx = jax.random.choice(key=rng, a=vocab_size, p=y_pred[0, :])
    y_pred = jnp.zeros_like(x)
    y_pred = y_pred.at[0, idx].set(1)

    return h, y_pred


def train(
    model: CharRNN,
    *,
    rng: Array,
    params: Parameters,
    X_train: List[Array],
    Y_train: List[Array],
    vocab_size: int,
    epochs: int | None = None,
    learning_rate: float | None = None,
    max_grad: float | None = None,
) -> Parameters:

    epochs = default_arg(epochs, 10)
    learning_rate = default_arg(learning_rate, 0.01)
    max_grad = default_arg(max_grad, 10)

    start_time = time()

    # Set up optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    loss_fn = jax.value_and_grad(partial(_loss, model))
    step_fn = jax.jit(partial(_step, loss_fn, optimizer, max_grad))

    loss = None
    for epoch in range(epochs):

        # Emit signal
        train_epoch_started.send(model, epoch=epoch, elapsed=(time() - start_time))

        epoch_loss = 0
        for _ in range(len(X_train)):
            rng, sub_rng = jax.random.split(rng)
            X, Y = _sample(sub_rng, X_train, Y_train, vocab_size)
            params, optimizer_state, loss = step_fn(optimizer_state, params, X, Y)
            epoch_loss += loss

        # Emit signal
        train_epoch_completed.send(
            model, epoch=epoch, loss=epoch_loss, elapsed=(time() - start_time)
        )

    return params


def _sample(
    rng: Array, X: List[Array], Y: List[Array], vocab_size: int
) -> tuple[Array, Array]:

    # Pick a random index from the dataset
    i = jax.random.randint(rng, (1,), 0, len(X))[0]

    # Convert to one-hot representation
    x_out = jnp.eye(vocab_size)[X[i], :]
    y_out = jnp.eye(vocab_size)[Y[i], :]

    return x_out, y_out


def _loss(model, params, h, X_batch, y_batch):

    _, y_logits = model(params, h, X_batch)

    return optax.sigmoid_binary_cross_entropy(y_logits, y_batch).mean()


def _step(loss_fn, optimizer, max_grad, optimizer_state, params, X_batch, Y_batch):

    # Extract the hidden size from the Whx weights
    hidden_size = params[0].shape[0]
    # Initialize the hidden state as zeros
    h = jnp.zeros((hidden_size, 1))

    loss, grads = loss_fn(params, h, X_batch, Y_batch)

    # print(jnp.max(grads[0]), jnp.min(grads[0]))

    # Clip the gradients to prevent them from getting too large
    clipped_grads = tuple(jnp.clip(grad, -max_grad, max_grad) for grad in grads)

    # print(jnp.max(clipped_grads[0]), jnp.min(clipped_grads[0]))

    # Compute updates
    updates, optimizer_state = optimizer.update(clipped_grads, optimizer_state)

    # Update params
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, loss

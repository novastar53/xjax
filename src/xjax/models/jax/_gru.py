from typing import List
from functools import partial

from time import time

import jax
import optax
from jax import Array
import jax.numpy as jnp

from xjax.tools import default_arg
from xjax.signals import train_epoch_started, train_epoch_completed


Parameters = List[Array]


__all__ = [
    "gru",
    "train",
    "predict",
    "generate",
    "_sample",
]


def _sgm(x):
    return 1 / (1 + jnp.exp(-x))


class GRU:

    # Calculate the forward pass
    def __call__(self, *, params: Parameters, H: Array, X: Array):

        # Extract the parameters
        Wxr, Whr, Wxz, Whz, Wxh, Whh, Why, br, bz, bh, by = params

        # Initialize the output logits
        Y = jnp.zeros_like(X)

        # Loop over the inputs
        for i in range(X.shape[0]):
            Xt = jnp.expand_dims(X[i, :], 1)
            # First calculate the Reset and Update gates
            R = _sgm(jnp.dot(Wxr, Xt) + jnp.dot(Whr, H) + br)
            Z = _sgm(jnp.dot(Wxz, Xt) + jnp.dot(Whz, H) + bz)
            # Then calculate the candidate Hidden State
            Hc = _sgm(jnp.dot(Wxh, Xt) + R * jnp.dot(Whh, H) + bh)
            # Then calculate the updated Hidden State
            H = Z * H + (1 - Z) * Hc

            # Finally, calculate the output logits
            Yt = jnp.dot(Why, H) + by
            Yt = jnp.squeeze(Yt)

            # Update the output sequence
            Y = Y.at[i].set(Yt)

        # Return the updated Hidden State and the output logits
        return H, Y


def _init_W(rng: Array, dim1: int, dim2: int, sigma=0.01):
    return jax.random.normal(rng, shape=(dim1, dim2)) * sigma


def _init_b(dim: int):
    return jnp.zeros((dim, 1))


def gru(rng: Array, vocab_size: int, hidden_size: int):

    # Wxr, Whr, Wxz, Whz, Wxh, Whh, Why, br, bz, bh, by

    # Reset Gate
    # Decides how much of the previous hidden state
    # to use to calculate the candidate hidden state
    Wxr = _init_W(rng, hidden_size, vocab_size)

    rng, sub_rng = jax.random.split(rng)
    Whr = _init_W(sub_rng, hidden_size, hidden_size)

    br = _init_b(hidden_size)

    # Update Gate
    # Used to calculate a linear combination of
    # the previous and candidate hidden states
    # to obtain the updated hidden state
    rng, sub_rng = jax.random.split(rng)
    Wxz = _init_W(sub_rng, hidden_size, vocab_size)

    rng, sub_rng = jax.random.split(rng)
    Whz = _init_W(sub_rng, hidden_size, hidden_size)

    bz = _init_b(hidden_size)

    # Candidate hidden state
    # Used to calculate the updated hidden state
    # Along with the previous hidden state and the update gate
    rng, sub_rng = jax.random.split(rng)
    Wxh = _init_W(sub_rng, hidden_size, vocab_size)

    rng, sub_rng = jax.random.split(rng)
    Whh = _init_W(sub_rng, hidden_size, hidden_size)

    bh = _init_b(hidden_size)

    # Output Layer
    # Projects the final hidden state to the vocab dimension
    # in order to predict the next token
    rng, sub_rng = jax.random.split(rng)
    Why = _init_W(sub_rng, vocab_size, hidden_size)

    by = _init_b(vocab_size)

    # Initialize model
    gru = GRU()

    return (Wxr, Whr, Wxz, Whz, Wxh, Whh, Why, br, bz, bh, by), gru

def _sample(
    rng: Array, X: List[Array], Y: List[Array], vocab_size: int
) -> tuple[Array, Array]:

    # Pick a random index from the dataset
    i = jax.random.randint(rng, (1,), 0, len(X))[0]

    # Convert to one-hot representation
    x_out = jnp.eye(vocab_size)[X[i], :]
    y_out = jnp.eye(vocab_size)[Y[i], :]

    return x_out, y_out


def _loss(model: GRU, params: Parameters, H: Array, X_batch: Array, Y_batch: Array):

    _, Y_logits = model(params=params, 
                        H=H, 
                        X=X_batch)

    return optax.sigmoid_binary_cross_entropy(Y_logits, Y_batch).mean()

def _step(loss_fn, optimizer, max_grad, optimizer_state, params, X_batch, Y_batch):

    hidden_size = params[0].shape[0]

    H = jnp.zeros((hidden_size,1))
    loss, grads = loss_fn(params, H, X_batch, Y_batch)
    clipped_grads = tuple(jnp.clip(grad, -max_grad, max_grad) for grad in grads)
    updates, optimizer_state = optimizer.update(clipped_grads, optimizer_state)
    params = optax.apply_updates(params, updates)

    return params, optimizer_state, loss


def train(
    model: GRU,
    *,
    rng: Array,
    params: Parameters,
    X: jax.Array,
    Y: jax.Array,
    vocab_size,
    batch_size: int | None = None,
    num_epochs: int | None = None,
    learning_rate: int | None = None,
    max_grad: int | None = None,
) -> Parameters:

    # get default vaules
    batch_size = default_arg(batch_size, 32)
    num_epochs = default_arg(num_epochs, 10)
    learning_rate = default_arg(learning_rate, 10**-3)
    max_grad = default_arg(max_grad, 10**(-3))

    start_time = time()

    # set up the optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    # set up the loss function
    loss_fn = jax.value_and_grad(partial(_loss, model))
    step_fn = jax.jit(partial(_step, loss_fn, optimizer, max_grad))

    # loop through epochs

    loss = None
    for epoch in range(num_epochs):

        # Emit signal
        train_epoch_started.send(model, epoch=epoch, elapsed=(time() - start_time))

        epoch_loss = 0
        for _ in range(len(X)):
            rng, sub_rng = jax.random.split(rng)
            X_batch, Y_batch = _sample(sub_rng, X, Y, vocab_size)
            params, optimizer_state, loss = step_fn(optimizer_state, params, X_batch, Y_batch)
            epoch_loss += loss

        # Emit signal
        train_epoch_completed.send(
            model, epoch=epoch, loss=epoch_loss, elapsed=(time() - start_time)
        )
    
    return params


def predict(
    model: GRU, *, rng: Array, params: Parameters, vocab_size: int, h: Array, x: Array
):

    h, y_logits = model(params=params, 
                        H=h, 
                        X=x)
    y_pred = jax.nn.softmax(y_logits, axis=1)

    # randomly select an output token based on the probabilities
    idx = jax.random.choice(key=rng, a=vocab_size, p=y_pred[0, :])
    y_pred = jnp.zeros_like(x)
    y_pred = y_pred.at[0, idx].set(1)

    return h, y_pred


def generate(
    rng: Array, prefix: List[int], params: Parameters, hidden_size: int, vocab_size: int, max_len: int,
) -> List[int]:

    # initialize the model and hidden state
    model = GRU()
    h = jnp.zeros((hidden_size, 1))

    # create the initial input vector
    result = []
    x = jnp.zeros((1, vocab_size))
    for i in prefix:
        result.append(i)
        x = x.at[0, i].set(1)

    idx_y = prefix[-1]

    # Assume that the last token in the vocab is the stop character
    # and terminate if the stop token is generated
    while idx_y != vocab_size - 1:

        rng, sub_rng = jax.random.split(rng)

        h, y = predict(
            model, rng=sub_rng, params=params, vocab_size=vocab_size, h=h, x=x
        )

        # get the predicted token
        idx_y = int(jnp.argmax(y))

        # update the input vector
        x = jnp.zeros((1, vocab_size))
        x = x.at[0, idx_y].set(1)

        # look up the token and add to final result
        result.append(idx_y)

        if len(result) >= max_len:
            break

    return result
from typing import List, Any
from functools import partial

from time import time

import jax
import optax
from jax import Array
import jax.numpy as jnp
from jax import random
from jax.nn import log_softmax

from xjax.tools import default_arg
from xjax.signals import train_epoch_started, train_epoch_completed


Parameters = List[Array]


__all__ = [
    "gru",
    "train",
    "predict",
    "generate",
    "perplexity",
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
        Y = []

        # Loop over the sequence
        for i in range(X.shape[1]):
            # Select the ith sequence and the entire minibatch
            Xt = X[:, i, :].T  # vocab size x 1 x batch size
            # First calculate the Reset and Update gates
            R = _sgm(jnp.dot(Wxr, Xt) + jnp.dot(Whr, H) + br)
            Z = _sgm(jnp.dot(Wxz, Xt) + jnp.dot(Whz, H) + bz)
            # Then calculate the candidate Hidden State
            Hc = _sgm(jnp.dot(Wxh, Xt) + R * jnp.dot(Whh, H) + bh)
            # Then calculate the updated Hidden State
            H = Z * H + (1 - Z) * Hc

            # Finally, calculate the output logits
            Yt = jnp.dot(Why, H) + by

            # Update the output sequence
            Y.append(Yt)

        # Return the updated Hidden State and the output logits
        return H, jnp.stack(Y)


def _init_W(rng: Array, dim1: int, dim2: int, sigma: float | None = None):

    sigma = default_arg(sigma, 1/jnp.sqrt(dim1))
    #sigma = 0.01

    return jax.random.normal(rng, shape=(dim1, dim2)) * sigma


def _init_b(dim: int, default_init: int | None = None):
    if default_init == None:
        return jnp.zeros((dim, 1))
    return jnp.array([[default_init]*dim]).reshape((dim, 1))


def gru(rng: Array, vocab_size: int, hidden_size: int):

    # Wxr, Whr, Wxz, Whz, Wxh, Whh, Why, br, bz, bh, by

    # Reset Gate
    # Decides how much of the previous hidden state
    # to use to calculate the candidate hidden state
    Wxr = _init_W(rng, hidden_size, vocab_size)

    rng, sub_rng = jax.random.split(rng)
    Whr = _init_W(sub_rng, hidden_size, hidden_size)

    br = _init_b(hidden_size, default_init=1.0)

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
    rng: Array, X: List[Array], Y: List[Array], batch_size: int, vocab_size: int
) -> tuple[Array, Array]:

    # Pick a set of random indices from the dataset
    idxs = jax.random.randint(rng, (batch_size,), 0, len(X))

    # Convert to one-hot representation
    x_out = jnp.eye(vocab_size)[X[idxs], :]
    y_out = jnp.eye(vocab_size)[Y[idxs], :]

    return x_out, y_out


def _loss(model: GRU, params: Parameters, H: Array, X_batch: Array, Y_batch: Array):

    _, Y_logits = model(params=params, 
                        H=H, 
                        X=X_batch)

    Y_logits = jnp.transpose(Y_logits, (2, 0, 1))
    loss = optax.losses.softmax_cross_entropy(logits=Y_logits, labels=Y_batch)
    return loss.mean()

def _step(loss_fn, optimizer, max_grad, optimizer_state, params, X_batch, Y_batch):

    hidden_size = params[0].shape[0]
    batch_size = X_batch.shape[0]
    H = jnp.zeros((hidden_size,batch_size))
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
    X_train: jax.Array,
    Y_train: jax.Array,
    X_valid: jax.Array | None = None,
    Y_valid: jax.Array | None = None,
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
    max_grad = default_arg(max_grad, 1)

    start_time = time()

    # set up the optimizer
    optimizer = optax.adam(learning_rate=learning_rate)
    optimizer_state = optimizer.init(params)

    # set up the loss function
    loss_fn = jax.value_and_grad(partial(_loss, model))
    step_fn = jax.jit(partial(_step, loss_fn, optimizer, max_grad))
    #step_fn = partial(_step, loss_fn, optimizer, max_grad)

    # loop through epochs

    loss = None
    for epoch in range(num_epochs):

        epoch_loss = 0
        num_iter = len(X_train)//batch_size
        for _ in range(num_iter):
            rng, sub_rng = jax.random.split(rng)
            X_batch, Y_batch = _sample(sub_rng, X_train, Y_train, batch_size, vocab_size)
            params, optimizer_state, loss = step_fn(optimizer_state, params, X_batch, Y_batch)
            epoch_loss += loss
        epoch_loss = epoch_loss/num_iter
        # Emit signal
        if epoch % 20 == 0:
            # Calculate validation loss 
            rng, sub_rng = jax.random.split(rng)
            valid_perp=None
            if X_valid is not None and Y_valid is not None:
                valid_perp = perplexity(model, params, vocab_size, X_valid, Y_valid)
            train_epoch_completed.send(
                model, epoch=epoch, train_loss=epoch_loss, valid_perplexity=valid_perp, elapsed=(time() - start_time)
            )

    valid_perp=None
    if X_valid is not None and Y_valid is not None:
        valid_perp = perplexity(model, params, vocab_size, X_valid, Y_valid)
    train_epoch_completed.send(
        model, epoch=epoch, train_loss=epoch_loss, valid_perplexity=valid_perp, elapsed=(time() - start_time)
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


def perplexity(model, params, vocab_size, X, Y):

    X_onehot = jnp.eye(vocab_size)[X, :]
    Y_onehot = jnp.eye(vocab_size)[Y, :]
    batch_size = X.shape[0]
    hidden_size = params[0].shape[0]
    H = jnp.zeros((hidden_size, batch_size))
    loss = _loss(model, params, H, X_onehot, Y_onehot)

    return 2**loss


def generate(
    rng: Array, prefix: List[int], params: Parameters, hidden_size: int, vocab_size: int, max_len: int,
) -> List[int]:

    # Initialize the model and hidden state
    model = GRU()
    H = jnp.zeros((hidden_size, 1))

    # Feed the prefix to the model
    X = jnp.eye(vocab_size)[prefix, :]
    X = jnp.expand_dims(X, 0) # 1 x seq len x vocab 
    H, Y_pred = model(params=params, H=H, X=X)
    probs = jax.nn.softmax(jnp.squeeze(Y_pred[-1,:]))
    pred_token_idx = random.categorical(rng, jnp.log(probs))

    # Prepare the final output to feed back into the model
    Y_pred = jnp.squeeze(Y_pred)
    Y_pred = jnp.expand_dims(Y_pred[-1, :], axis=(0,1))

    # Collect predictions from the model
    output = prefix.copy()
    for _ in range(max_len):

        # Feed the previous output into the model
        H, Y_pred = model(params=params, H=H, X=Y_pred)

        # Retrieve the token and append to output
        Y_pred = Y_pred.squeeze()
        probs = jax.nn.softmax(Y_pred)
        rng, sub_rng = random.split(rng)
        pred_token_idx = random.categorical(sub_rng, jnp.log(probs))
        output.append(int(pred_token_idx))

        # Reformat the output for the model
        Y_pred = jnp.eye(vocab_size)[pred_token_idx, :]
        Y_pred = jnp.expand_dims(Y_pred, axis=(0,1))

    return output

    
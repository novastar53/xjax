from typing import List


import jax
from jax import Array
import jax.numpy as jnp

Parameters = List[Array]


__all__ = [
    "gru",
    "train",
    "predict",
    "generate",
]

def _sgm(x):
    return 1/(1 + jnp.exp(-x))

class GRU:

    # Calculate the forward pass 
    def __call__(*, params: Parameters, H: Array, X: List[Array]):
        
        # Extract the parameters
        Wxr, Whr, Wxz, Whz, Wxh, Whh, Why, br, bz, bh, by = params

        # Initialize the output logits
        Y = jnp.zeros_like(X)

        # Loop over the inputs
        for i in range(len(X[0])):
            Xt = jnp.expand_dims(X[i, :], 1)
            # First calculate the Reset and Update gates
            R = _sgm(jnp.dot(Wxr,Xt) + jnp.dot(Whr, H) + br)
            Z = _sgm(jnp.dot(Wxz, Xt) + jnp.dot(Whz, H) + bz)
            # Then calculate the candidate Hidden State
            Hc = _sgm(jnp.dot(Wxh, Xt) + R*jnp.dot(Whh, H) + bh)
            # Then calculate the updated Hidden State
            H = Z*H + (1-Z)*Hc

            # Finally, calculate the output logits
            Yt = jnp.dot(Why, H) + by
            Yt = jnp.squeeze(Yt)

            # Update the output sequence
            Y = Y.at[i].set(Yt)

        # Return the updated Hidden State and the output logits
        return Y


def _init_W(rng: Array, dim1: int, dim2: int, sigma=0.01):
    return jax.random.normal(rng, shape=(dim1, dim2))*sigma

def _init_b(dim: int):
    return jnp.zeros((dim,1))

def gru(rng: Array, vocab_size: int, hidden_size: int):

    #Wxr, Whr, Wxz, Whz, Wxh, Whh, Why, br, bz, bh, by 

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









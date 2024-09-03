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





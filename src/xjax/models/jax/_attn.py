from typing import List

import jax
from jax import Array
import jax.numpy as jnp



__all__ = [
    "BasicDotProdAttention"
]



class BasicDotProdAttention:
    """
    A basic non-parameterized dot-product self-attention layer 
    for time sequences
    
    inputs: 
    """

    def __call__(self, inputs: Array):

        inputs_t = jnp.transpose(inputs, (0, 2, 1)) # (batch_size, hidden_dim, timesteps)
        attn_scores = jnp.dot(inputs, inputs_t) # (batch_size, timesteps, timesteps)
        attn_weights =  jax.nn.softmax(attn_scores, axis=2)  # (batch_size, timesteps, timesteps)
        outputs = jnp.dot(attn_weights, inputs) # (batch_size, timesteps, hidden_dim)
        outputs = jnp.mean(outputs, axis=1) # (batch_size, hidden_dim)
        return outputs 
        







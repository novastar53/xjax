import jax
from jax import Array
import jax.numpy as jnp


__all__ = [
    "BasicDotProdAttention"
]

class BasicDotProdAttention:
    """
    A basic non-parameterized dot-product self-attention layer 
    
    inputs: 
        query: The Query vector (batch_size, keys_dim)
        keys:  The Key vector (batch_size, timesteps, keys_dim) 
        values: The Values vector (batch_size, timesteps, values_dim)
    """

    def __call__(self,* , query: Array, keys: Array, values: Array):
        
        dim_keys = keys.shape[2]

        # Calculate the attention weights using the keys and queries
        query = jnp.expand_dims(query, axis=1) # (batch_size, timesteps)
        scores = jnp.sum(query * keys, axis=2) / jnp.sqrt(dim_keys) # (batch_size, timesteps)
        attn_weights = jax.nn.softmax(scores, axis=1) # (batch_size, timesteps)

        # Calculate the context using the attention weights and the values
        attn_weights = jnp.expand_dims(attn_weights, axis=2) # (batch_size, timesteps, 1)
        context = jnp.sum(values * attn_weights, axis=1) # (batch_size, hidden_dim) 

        return context # (batch_size, hidden_dim)
 
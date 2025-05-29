#%%

#jax has its own version of numpy (sitax is very similar for most things)
import jax.numpy as jnp
import jax
import time

#%%

#create a vector
a = jnp.array([1,2,3])


@jax.jit # thanks to this decorator jax understand that has to compile the fucntion (this is going to make things faster)
def algo(x):
    return jnp.where(x%2 == 0, x, x-3)

_ = algo(a)
    
start_time = time.time()
algo(a).block_until_ready() #block_until_ready() wait for the result before executing the next line
end_time = time.time()
print(f"Compiled execution: {end_time - start_time}")
      
#%%

#create a vector
a = jnp.array([1,2,3])


def algo(x):
    return jnp.where(x%2 == 0, x, x-3)

_ = algo(a)
    
start_time = time.time()
algo(a).block_until_ready() #block_until_ready() wait for the result before executing the next line
end_time = time.time()
print(f"Standard execution: {end_time - start_time}")
# %%

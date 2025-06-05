#import section
import jax
import jax.numpy as jnp


#we can create a function taking as input a single data point and automatically vectorize it with jax

#define a key, the equivalend of the seed (because we want to jenerate random numbers). 
#N.B. it can be used just ONCE!
key = jax.random.key(42)
#define weight matrix (as if it was a NN)
W = jax.random.normal(key,(150,100)) #mapping from 100 samples to 150 neurons
#define input data
X = jax.random.normal(key,(10,100)) #mapping from 10 samples to 100 neurons

#this function works for a single sample
def forward_pass(x):
    return jnp.dot(W,x)
    
#single array works
single = jax.random.normal(key,(100))
print(forward_pass(single).shape)

#batch array doesn't!
#print(forward_pass(X))

#but if we automatically vecotrize the function
vectorized_forward_pass = jax.vmap(forward_pass)
print(vectorized_forward_pass(X).shape)
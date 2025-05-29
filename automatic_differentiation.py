#%%

#import section
import jax

#%%

#with automatic differentiation it is possible to differentiate fucntions

def squared(x):
    return x ** 2

# x = 10
# f(x) = x ^2 = 100
# f'(x) = 2x = 20
# f''(x) = 2 = 2
# f'''(x) = 0 = 0
x = 10.0 #put number as float, with int would not work
print(squared(x))
print(jax.grad(squared)(x))
print(jax.grad(jax.grad(squared))(x))
print(jax.grad(jax.grad(jax.grad(squared)))(x))
# %%

#it is also possbile to do partial differentiations (i.e. differentiation with respect to a single variable)

def multi_function(x,y,z):
    return x**2 + 2*y**2 + 3*z**2
# x = y = z = 2.0
# df/dx = 2x = 4
# df/dy = 4y = 8
# df/dz = 6z = 12

x = y = z = 2.0

print(jax.grad(multi_function, argnums=0)(x,y,z))
print(jax.grad(multi_function, argnums=1)(x,y,z))
print(jax.grad(multi_function, argnums=2)(x,y,z))



# %%

#import section
import jax
import jax.numpy as jnp

from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

#load data
data = load_iris()

#features
X = data.data
#targets (reshaped vertically: N rows, 1 col )
y = data.target.reshape(-1,1)
#transform labels in one hot encoding
one_hot_encoder = OneHotEncoder(sparse_output=False)
#encode data
y = one_hot_encoder.fit_transform(y)

#split data in train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#scale input data
scaler = StandardScaler()
#transform training
X_train = scaler.fit_transform(X_train)
#project testing
X_test = scaler.transform(X_test)



#### STRUCTURE OF THE NETWORKD ###
# input -> hidden layer 1 -> hidden layer 2 -> output
"""
for each transition you have a number of weigts equal to (previous dim * next dim)
(e.g. the weigths to go from input to hidden layer 1 is going to be [input_dim * hl1_dim])
and a number fo biases equal to the number of neurons on the next layer
(e.g. the biases to go from input to hidden layer 1 is going to be [hl1_dim])
"""
def init_params(inp_dim, hid_dim_1, hid_dim_2, out_dim, random_key):
    
    random_keys = jax.random.split(random_key, 3)
    
    w0 = jax.random.normal(random_keys[0], (inp_dim, hid_dim_1))
    w1 = jax.random.normal(random_keys[1], (hid_dim_1, hid_dim_2))
    w2 = jax.random.normal(random_keys[2], (hid_dim_2,out_dim))
    
    b1 = jnp.zeros((hid_dim_2))
    b2 = jnp.zeros((out_dim))
    b0 = jnp.zeros((hid_dim_1))
    
    return w0, w1, w2, b0, b1, b2


def forward_pass(x, params):
    
    w0, w1, w2, b0, b1, b2 = params
    
    logits_1 = jax.nn.relu(jnp.dot(x,w0) + b0)
    logits_2 = jax.nn.relu(jnp.dot(logits_1,w1) + b1)
    logits_out = jnp.dot(logits_2,w2) + b2
    
    return logits_out


def loss_fn(params,x,y,l2_reg = 0.001):
    
    y_hat = forward_pass(x,params)
    probs = jax.nn.softmax(y_hat)
    
    #l2 regularization term for loss function
    l2_reg_term = l2_reg * sum(jnp.sum(w ** 2) for w in params[:3])
    
    #cross entropy loss
    ce_loss = -jnp.mean(jnp.sum(y * jnp.log(probs + 1e-8), axis=1))
    
    return ce_loss + l2_reg_term

@jax.jit
def train_step(params,x,y,lr):
    #compute the grads
    grads = jax.grad(loss_fn)(params,x,y)
    #return twiked params
    return [(param - lr * grad) for param, grad in zip(params, grads)]


def accuracy(params,x,y):
    preds = jnp.argmax(forward_pass(x,params), axis=1)
    targets = jnp.argmax(y, axis=1)
    
    return jnp.mean(preds == targets)


#perform training    
random_key = jax.random.key(42)
params = init_params(4,16,8,3, random_key)
epochs = 200
test_acc = []
train_acc = []
for epoch in range(epochs):
    params = train_step(params,X_train,y_train,lr=0.01)
    test_acc.append(accuracy(params,X_test,y_test))
    train_acc.append(accuracy(params,X_train,y_train))
    #print(accuracy_score(y_test,jax.nn.softmax(forward_pass(X_test,params)).astype(int)))
    
    
plt.plot(test_acc, label="test")
plt.plot(train_acc, label="train")
plt.legend()
plt.show()




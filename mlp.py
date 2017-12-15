import mxnet as mx
import numpy as np

# cpu or gpu
model_ctx = mx.cpu()


def evaluate_accuracy(data_iterator, net):
    numerator = 0.
    denominator = 0.
    for i, (data, label) in enumerate(data_iterator):
        data = data.as_in_context(model_ctx).reshape((-1, 784))
        label = label.as_in_context(model_ctx)
        output = net(data)
        predictions = mx.nd.argmax(output, axis=1)
        numerator += mx.nd.sum(predictions == label)
        denominator += data.shape[0]
    return (numerator / denominator).asscalar()


###########################
# Load MNIST data
batch_size = 64
transform = lambda data, label: (data.astype(np.float32)/255, label.astype(np.float32))
mnist_train_dataset = mx.gluon.data.vision.MNIST(train=True, transform=transform)
mnist_test_dataset = mx.gluon.data.vision.MNIST(train=False, transform=transform)
train_data = mx.gluon.data.DataLoader(mnist_train_dataset, batch_size, shuffle=True)
test_data = mx.gluon.data.DataLoader(mnist_test_dataset, batch_size, shuffle=False)

#########################
# network definition
num_inputs = 784
num_hidden = 256
num_outputs = 10
weight_scale = .01
# first hidden layer
W1 = mx.nd.random_normal(shape=(num_inputs, num_hidden), scale=weight_scale, ctx=model_ctx)
b1 = mx.nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)
# second hidden layer
W2 = mx.nd.random_normal(shape=(num_hidden, num_hidden), scale=weight_scale, ctx=model_ctx)
b2 = mx.nd.random_normal(shape=num_hidden, scale=weight_scale, ctx=model_ctx)
# output layer
W3 = mx.nd.random_normal(shape=(num_hidden, num_outputs), scale=weight_scale, ctx=model_ctx)
b3 = mx.nd.random_normal(shape=num_outputs, scale=weight_scale, ctx=model_ctx)

# allocate space for the gradients
params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

# rectified linear unit
relu = lambda x: mx.nd.maximum(x, mx.nd.zeros_like(x))

# softmax for the output
def softmax(x):
    exp = mx.nd.exp(x - mx.nd.max(x))
    partition = mx.nd.nansum(exp, axis=0, exclude=True).reshape((-1, 1))
    return exp / partition

def net(x):
    h1_linear = mx.nd.dot(x, W1) + b1
    h1 = relu(h1_linear)
    h2_linear = mx.nd.dot(h1, W2) + b2
    h2 = relu(h2_linear)
    yhat_linear = mx.nd.dot(h2, W3) + b3
    return yhat_linear


def optimizer(params, lr):
    for param in params:
        param[:] = param - lr * param.grad

epochs = 10
learning_rate = .001
smoothing_constant = .01
num_examples = 60000

def softmax_cross_entropy(yhat_linear, y):
    return - mx.nd.nansum(y * mx.nd.log_softmax(yhat_linear), axis=0, exclude=True)


for e in range(epochs):
    cumulative_loss = 0
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(model_ctx).reshape((-1, num_inputs))
        label = label.as_in_context(model_ctx)
        label_one_hot = mx.nd.one_hot(label, num_outputs)
        with mx.autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        optimizer(params, learning_rate)
        cumulative_loss += mx.nd.sum(loss).asscalar()

    test_accuracy = evaluate_accuracy(test_data, net)
    train_accuracy = evaluate_accuracy(train_data, net)
    print("Epoch %s. Loss: %s, Train_acc %s, Test_acc %s" %
          (e, cumulative_loss/num_examples, train_accuracy, test_accuracy))
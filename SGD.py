import pennylane as qml
from pennylane import qaoa
import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random

# Fix the seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def create_test_graph(n_nodes=20):
    k = random.randint(3, n_nodes-1)
    edge_prob = k / n_nodes
    return nx.erdos_renyi_graph(n_nodes, edge_prob)

def qaoa_maxcut_graph(graph, n_layers=2):
    """Compute the maximum cut of a graph using QAOA."""
    n_nodes = graph.number_of_nodes()
    dev = qml.device("default.qubit", wires=n_nodes)
    cost_h, mixer_h = qaoa.maxcut(graph)

    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)

    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def circuit(params, **kwargs):
        for w in range(n_nodes):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(cost_h)

    def qaoa_cost(params, **kwargs):
        return circuit(params)

    return qaoa_cost

# Create a test graph
test_graph = create_test_graph(20)
graph_cost = qaoa_maxcut_graph(test_graph, n_layers=2)
print(f"Number of nodes: {test_graph.number_of_nodes()}")
# Parameters are initialized to ones
n_layers = 2
params_shape = (n_layers, 2)
x = tf.Variable(tf.ones(params_shape))

# We set the optimizer to be a Stochastic Gradient Descent
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
step = 5

# Training process
sdg_losses = []
for i in range(step):
    print('dentro il for')
    with tf.GradientTape() as tape:
        print('dentro il tape')
        loss = graph_cost(x)
    print('i')
    sdg_losses.append(loss.numpy().flatten()[0])

    gradients = tape.gradient(loss, [x])
    opt.apply_gradients(zip(gradients, [x]))
    print(f"Step {i+1} - Loss = {loss.numpy().flatten()[0]}")

# Print final results
print(f"Final cost function: {graph_cost(x).numpy().flatten()[0]}")
print(f"Optimized angles: {x.numpy()}")

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(sdg_losses, marker='o', linestyle='-', color='orange', lw=3, label='SGD')
plt.xlabel('Iteration')
plt.ylabel('Cost Function Value')
plt.title('Cost Function Value During SGD Iterations')
plt.grid(True)
plt.legend()
plt.savefig("cost_function_plot_sgd.png")
#plt.show()

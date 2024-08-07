# Quantum Machine Learning
import pennylane as qml
from pennylane import qaoa

# Classical Machine Learning
import tensorflow as tf

# Generation of graphs
import networkx as nx

# Standard Python libraries
import numpy as np
import matplotlib.pyplot as plt
import random

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Disabilita tutti i messaggi di logging
tf.get_logger().setLevel('ERROR')

# Fix the seed for reproducibility, which affects all random functions in this demo
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def create_graph_train_dataset(num_graphs):
    dataset = []
    for _ in range(num_graphs):
        n_nodes = random.randint(6, 9)
        k = random.randint(3, n_nodes - 1)
        edge_prob = k / n_nodes
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
        
        dataset.append(G)
    return dataset

def iterate_minibatches(inputs, batchsize, shuffle=False):
    indices = np.arange(len(inputs))
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield [inputs[i] for i in excerpt]

def qaoa_maxcut_graph(graph, n_layers=2):
    """Compute the maximum cut of a graph using QAOA."""
    # Number of nodes in the graph
    n_nodes = graph.number_of_nodes()
    # Initialize the QAOA device
    dev = qml.device("default.qubit.tf", wires=n_nodes)
    # Define the QAOA cost function
    cost_h, mixer_h = qaoa.maxcut(graph)
    # Define the QAOA layer structure
    def qaoa_layer(gamma, alpha):
        qaoa.cost_layer(gamma, cost_h)
        qaoa.mixer_layer(alpha, mixer_h)
    # Define the QAOA quantum circuit
    @qml.qnode(dev, interface="tf", diff_method="backprop")
    def circuit(params, **kwargs):
        for w in range(n_nodes):
            qml.Hadamard(wires=w)
        qml.layer(qaoa_layer, n_layers, params[0], params[1])
        return qml.expval(cost_h)
    # Define the QAOA optimization cost function
    def qaoa_cost(params, **kwargs):
        """Evaluate the cost Hamiltonian, given the angles and the graph."""
        # This qnode evaluates the expectation value of the cost hamiltonian operator
        return circuit(params)
    return qaoa_cost

def observed_improvement_loss(costs):
    """
    Compute the observed improvement loss based on the costs from each iteration.
    
    Args:
    costs (list of tf.Tensor): A list of tensors representing the cost at each iteration.
    
    Returns:
    tf.Tensor: The observed improvement loss.
    """
    initial_cost = costs[0]
    final_cost = costs[-1]

    # Calculate the observed improvement
    improvement = initial_cost - final_cost

    # Calculate the loss as the negative improvement (since we want to minimize loss, maximizing improvement)
    loss = -improvement

    return tf.reshape(loss, shape=(1, 1))

def hybrid_iteration(inputs, graph_cost, n_layers=2):
    """Perform a single time step in the computational graph of the custom RNN."""

    # Unpack the input list containing the previous cost, parameters, and hidden states (denoted as 'h').
    prev_cost = inputs[0]
    prev_params = inputs[1]
    prev_h = inputs[2]

    # Concatenate the previous parameters and previous cost to create new input
    new_input = tf.concat([prev_cost, prev_params], axis=-1)

    # Call the GRU cell, which outputs new values for the parameters along with new internal state h
    new_params, new_h = cell(new_input, states=prev_h)

    # Reshape the parameters to correctly match those expected by PennyLane
    _params = tf.reshape(new_params, shape=(2, n_layers))

    # Performing one calculation with the quantum circuit with new parameters
    _cost = graph_cost(_params)

    # Reshape to be consistent with other tensors
    new_cost = tf.reshape(tf.cast(_cost, dtype=tf.float32), shape=(1, 1))

    return [new_cost, new_params, new_h]

def recurrent_loop(graph_cost, n_layers=2, intermediate_steps=False, num_iterations=10):
    """Creates the recurrent loop for the Recurrent Neural Network."""
    # Initialize starting all inputs (cost, parameters, hidden states) as zeros.
    initial_cost = tf.zeros(shape=(1, 1))
    initial_params = tf.zeros(shape=(1, 2 * n_layers))
    initial_h = tf.zeros(shape=(1, 2 * n_layers))

    # Initialize the output list with the initial state
    outputs = [hybrid_iteration([initial_cost, initial_params, initial_h], graph_cost, n_layers)]

    # Perform the iterations
    for _ in range(1, num_iterations):
        outputs.append(hybrid_iteration(outputs[-1], graph_cost, n_layers))

    # Extract the costs from the outputs
    costs = [output[0] for output in outputs]

    #DEBUG
    print("Intermediary costs:", [cost.numpy() for cost in costs])
    #DEBUG
    
    # Calculate the observed improvement loss
    loss = observed_improvement_loss(costs)
    
    if intermediate_steps:
        params = [output[1] for output in outputs]
        return params + [loss]
    else:
        return loss

def train_step(graph_cost):
    """Single optimization step in the training procedure."""

    with tf.GradientTape() as tape:
        # Evaluates the cost function
        loss = recurrent_loop(graph_cost)

    # Evaluates gradients, cell is the GRU cell defined previously
    grads = tape.gradient(loss, cell.trainable_weights)

    # Apply gradients and update the weights of the GRU cell
    opt.apply_gradients(zip(grads, cell.trainable_weights))
    return loss

n_layers = 2
cell = tf.keras.layers.GRUCell(2 * n_layers)

graphs = create_graph_train_dataset(40)
# This is the list of QAOA cost functions for each graph
graph_cost_list = [qaoa_maxcut_graph(g) for g in graphs]

# These cost functions will be used to train the GRU model:
# The GRU model will take as input the cost functions and will output the QAOA solution.
# At each iteration of the GRU, one execution of the quantum circuit is performed.

# Select an optimizer
opt = tf.keras.optimizers.Adam(learning_rate=0.1)

# Set the number of training epochs
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    for graph_cost in graph_cost_list:
        total_loss = np.array([])       
        loss = train_step(graph_cost)
        total_loss = np.append(total_loss, loss.numpy())
        print(f" >> Mean Loss during epoch: {np.mean(total_loss)}")

new_graph = nx.gnp_random_graph(12, p=3 / 7)
new_cost = qaoa_maxcut_graph(new_graph)

plt.figure(figsize=(8, 8))
nx.draw(new_graph)
plt.savefig("GRU_test_graph_120dataset_10epochs.png")

start_zeros = tf.zeros(shape=(2 * n_layers, 1))
res = recurrent_loop(new_cost, intermediate_steps=True)

guess_list = []
guess_list.append(start_zeros)

# Execute 10 iterations of the GRU model
for i in range(10):
    guess = res[i]
    guess_list.append(guess)

# Losses from the hybrid GRU model
gru_losses = [new_cost(tf.reshape(guess, shape=(2, n_layers))) for guess in guess_list]

fig, ax = plt.subplots()

plt.plot(gru_losses, color="blue", lw=3, ls="-.", label="GRU")

plt.grid(ls="--", lw=2, alpha=0.25)
plt.ylabel("Cost function", fontsize=12)
plt.xlabel("Iteration", fontsize=12)
plt.legend()
ax.set_xticks([0, 5, 10, 15, 20])
plt.show()

# Parameters are randomly initialized
x = tf.Variable(np.random.rand(2, 2))

# We set the optimizer to be a Stochastic Gradient Descent
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
step = 10

# Training process
steps = []
sdg_losses = []
for _ in range(step):
    with tf.GradientTape() as tape:
        loss = new_cost(x)

    steps.append(x)
    sdg_losses.append(loss)

    gradients = tape.gradient(loss, [x])
    opt.apply_gradients(zip(gradients, [x]))
    print(f"Step {_+1} - Loss = {loss}")

print(f"Final cost function: {new_cost(x).numpy()}\nOptimized angles: {x.numpy()}")

fig, ax = plt.subplots()

plt.plot(sdg_losses, color="orange", lw=3, label="SGD")

plt.plot(gru_losses, color="blue", lw=3, ls="-.", label="GRU")

plt.grid(ls="--", lw=2, alpha=0.25)
plt.legend()
plt.ylabel("Cost function", fontsize=12)
plt.xlabel("Iteration", fontsize=12)
ax.set_xticks([0, 5, 10, 15, 20])
plt.savefig("GRU_40dataset_10epochs.png")
plt.show()

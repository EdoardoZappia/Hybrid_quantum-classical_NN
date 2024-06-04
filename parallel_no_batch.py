import pennylane as qml
from pennylane import qaoa
import tensorflow as tf
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random
import concurrent.futures

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

def qaoa_maxcut_graph(graph, n_layers=2):
    """Compute the maximum cut of a graph using QAOA."""
    n_nodes = graph.number_of_nodes()
    dev = qml.device("default.qubit.tf", wires=n_nodes)
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

def observed_improvement_loss(costs):
    initial_cost = costs[0]
    final_cost = costs[-1]
    improvement = initial_cost - final_cost
    loss = -improvement
    return tf.reshape(loss, shape=(1, 1))

def build_lstm_model(n_layers=2):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2 * n_layers + 1,)),
        tf.keras.layers.Reshape((1, 2 * n_layers + 1)),
        tf.keras.layers.LSTM(2 * n_layers, return_sequences=True, stateful=False),
        tf.keras.layers.Reshape((2 * n_layers,))
    ])
    return model

lstm_model = build_lstm_model(n_layers=2)

def hybrid_iteration(inputs, graph_cost, n_layers=2):
    prev_cost, prev_params, prev_h, prev_c = inputs
    new_input = tf.concat([prev_cost, prev_params], axis=-1)
    new_input = tf.reshape(new_input, shape=(1, -1))  # Reshape the new_input tensor to ensure it has the correct shape
    new_input = tf.cast(new_input, tf.float32)  # Ensure new_input is of type float32
    states = [tf.cast(prev_h, tf.float32), tf.cast(prev_c, tf.float32)]  # Ensure states are of type float32
    new_params = lstm_model(new_input)
    _params = tf.reshape(new_params, shape=(2, n_layers))
    _cost = graph_cost(_params)
    new_cost = tf.reshape(tf.cast(_cost, dtype=tf.float32), shape=(1, 1))
    new_h, new_c = tf.split(new_params, 2, axis=-1)
    return [new_cost, new_params, new_h, new_c]

def recurrent_loop(graph_cost, n_layers=2, intermediate_steps=False, num_iterations=10):
    initial_cost = tf.ones(shape=(1, 1))
    initial_params = tf.ones(shape=(1, 2 * n_layers))
    initial_h = tf.ones(shape=(1, 2 * n_layers))
    initial_c = tf.ones(shape=(1, 2 * n_layers))
    outputs = [hybrid_iteration([initial_cost, initial_params, initial_h, initial_c], graph_cost, n_layers)]

    for _ in range(1, num_iterations):
        outputs.append(hybrid_iteration(outputs[-1], graph_cost, n_layers))

    costs = [output[0] for output in outputs]
    loss = observed_improvement_loss(costs)

    if intermediate_steps:
        params = [output[1] for output in outputs]
        return params + [loss]
    else:
        return loss

def train_step(graph_cost):
    with tf.GradientTape() as tape:
        loss = recurrent_loop(graph_cost)
    grads = tape.gradient(loss, lstm_model.trainable_weights)
    opt.apply_gradients(zip(grads, lstm_model.trainable_weights))
    return loss

n_layers = 2
graphs = create_graph_train_dataset(20)
graph_cost_list = [qaoa_maxcut_graph(g) for g in graphs]
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
epochs = 10

for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    total_loss = np.array([])

    # Parallelizzazione dei train_step con ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(train_step, graph_cost) for graph_cost in graph_cost_list]
        for i, future in enumerate(concurrent.futures.as_completed(futures)):
            loss = future.result()
            total_loss = np.append(total_loss, loss.numpy())
            print(f" > Graph {i+1}/{len(graph_cost_list)} - Loss: {loss[0][0]}")

    print(f" >> Mean Loss during epoch: {np.mean(total_loss)}")

new_graph = nx.gnp_random_graph(15, p=3 / 7)
new_cost = qaoa_maxcut_graph(new_graph)

nx.draw(new_graph)

start_zeros = tf.zeros(shape=(2 * n_layers, 1))
res = recurrent_loop(new_cost, intermediate_steps=True)
guess_list = []
guess_list.append(start_zeros)

for i in range(10):
    guess = res[i]
    guess_list.append(guess)

lstm_losses = [new_cost(tf.reshape(guess, shape=(2, n_layers))) for guess in guess_list]

fig, ax = plt.subplots()
plt.plot(lstm_losses, color="blue", lw=3, ls="-.", label="LSTM")
plt.grid(ls="--", lw=2, alpha=0.25)
plt.ylabel("Cost function", fontsize=12)
plt.xlabel("Iteration", fontsize=12)
plt.legend()
ax.set_xticks([0, 5, 10, 15, 20])
plt.savefig("lstm_notebook.png")
plt.show()

x = tf.Variable(np.random.rand(2, 2))
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
step = 15
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
plt.plot(lstm_losses, color="blue", lw=3, ls="-.", label="LSTM")
plt.grid(ls="--", lw=2, alpha=0.25)
plt.legend()
plt.ylabel("Cost function", fontsize=12)
plt.xlabel("Iteration", fontsize=12)
ax.set_xticks([0, 5, 10, 15, 20])
plt.savefig("lstm_vs_sgd_notebook.png")
plt.show()

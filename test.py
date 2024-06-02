# Quantum Machine Learning
import pennylane as qml
from pennylane import qaoa

# Classical Machine Learning
import tensorflow as tf
#from tensorflow.keras.layers import LSTMCell

# Generation of graphs
import networkx as nx

# Standard Python libraries
import numpy as np
import matplotlib.pyplot as plt
import random

# Fix the seed for reproducibility, which affects all random functions in this demo
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

def create_graph_train_dataset(num_graphs):
    dataset = []
    for _ in range(num_graphs):
        n_nodes = random.randint(6, 9)
        k = random.randint(3, n_nodes-1)
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

def create_graph_test_dataset(num_graphs):
    dataset = []
    n_nodes = 12
    for _ in range(num_graphs):
        k = random.randint(3, n_nodes-1)
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

def hybrid_iteration(inputs, graph_cost, lstm_cell, n_layers=2):
    prev_cost = inputs[0]
    prev_params = inputs[1]
    prev_h = inputs[2]
    prev_c = inputs[3]
    
    # Debug: print shapes
    print(f"prev_cost shape: {prev_cost.shape}")
    print(f"prev_params shape: {prev_params.shape}")
    print(f"prev_h shape: {prev_h.shape}")
    print(f"prev_c shape: {prev_c.shape}")
    
    new_input = tf.concat([prev_cost, prev_params], axis=-1)
    new_params, [new_h, new_c] = lstm_cell(new_input, states=[prev_h, prev_c])
    _params = tf.reshape(new_params, shape=(2, n_layers))
    _cost = graph_cost(_params)
    new_cost = tf.reshape(tf.cast(_cost, dtype=tf.float32), shape=(1, 1))
    return [new_cost, new_params, new_h, new_c]

def recurrent_loop(graph_cost, lstm_cell, n_layers=2, intermediate_steps=False, num_iterations=10):
    initial_cost = tf.ones(shape=(1, 1))
    initial_params = tf.ones(shape=(1, 2 * n_layers))
    initial_h = tf.ones(shape=(1, 2 * n_layers))
    initial_c = tf.ones(shape=(1, 2 * n_layers))
    
    # Debug: print initial shapes
    print(f"initial_cost shape: {initial_cost.shape}")
    print(f"initial_params shape: {initial_params.shape}")
    print(f"initial_h shape: {initial_h.shape}")
    print(f"initial_c shape: {initial_c.shape}")
    
    outputs = [hybrid_iteration([initial_cost, initial_params, initial_h, initial_c], graph_cost, lstm_cell, n_layers)]
    for _ in range(1, num_iterations):
        outputs.append(hybrid_iteration(outputs[-1], graph_cost, lstm_cell, n_layers))
    costs = [output[0] for output in outputs]
    print("Intermediary costs:", [cost.numpy() for cost in costs])
    loss = observed_improvement_loss(costs)
    if intermediate_steps:
        params = [output[1] for output in outputs]
        return params + [loss]
    else:
        return loss

def InitializeParameters(n_layers):
    return tf.keras.layers.LSTMCell(2 * n_layers)

def ParametersCopy(src, dest):
    dest.set_weights(src.get_weights())

def ZeroInitialize(lstm_cell):
    return [tf.zeros_like(var) for var in lstm_cell.trainable_variables]

def getBatch(M, di):
    return [di[i:i + M] for i in range(0, len(di), M)]

def Forward(params, graph_cost, lstm_cell, n_layers):
    return recurrent_loop(graph_cost, lstm_cell, n_layers=n_layers, intermediate_steps=False, num_iterations=10)

def loss_impr(initial_cost, final_cost):
    return observed_improvement_loss([initial_cost, final_cost])

def Backward(tape, loss, lstm_cell):
    return tape.gradient(loss, lstm_cell.trainable_weights)

def update(opt, lstm_cell, grads, learning_rate, batch_size):
    opt.apply_gradients(zip(grads, lstm_cell.trainable_weights))

def ThreadCode(di, lstm_cell, learning_rate, batch_size, n_layers, opt):
    B = getBatch(batch_size, di)
    for bi in B:
        for graph in bi:
            graph_cost = qaoa_maxcut_graph(graph, n_layers=n_layers)
            initial_cost = tf.ones(shape=(1, 1))
            with tf.GradientTape() as tape:
                final_cost = Forward(initial_cost, graph_cost, lstm_cell, n_layers=n_layers)
                loss = loss_impr(initial_cost, final_cost)
            grads = Backward(tape, loss, lstm_cell)
            update(opt, lstm_cell, grads, learning_rate, batch_size)
        

def TrainLSTM(graphs, learning_rate, batch_size, epoch, n_thread, n_layers):
    lstm_cell = InitializeParameters(n_layers)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    e = 1
    while e <= epoch:
        partitions = [graphs[i::n_thread] for i in range(n_thread)]
        for di in partitions:
            ThreadCode(di, lstm_cell, learning_rate, batch_size, n_layers, opt)
        e += 1
    return lstm_cell

def create_test_graph(n_nodes=20):
    k = random.randint(3, n_nodes-1)
    edge_prob = k / n_nodes
    return nx.erdos_renyi_graph(n_nodes, edge_prob)

def test_model(lstm_cell_trained, graph_cost, n_layers=2, num_iterations=10, output_filename="cost_function_plot.png"):
    initial_cost = tf.ones(shape=(1, 1))
    initial_params = tf.ones(shape=(1, 2 * n_layers))
    initial_h = tf.ones(shape=(1, 2 * n_layers))
    initial_c = tf.ones(shape=(1, 2 * n_layers))

    #DEBUG
    print('test')
    #

    costs = []
    outputs = [hybrid_iteration([initial_cost, initial_params, initial_h, initial_c], graph_cost, lstm_cell_trained, n_layers)]
    costs.append(outputs[0][0].numpy().flatten()[0])
    
    for _ in range(1, num_iterations):
        outputs.append(hybrid_iteration(outputs[-1], graph_cost, lstm_cell_trained, n_layers))
        costs.append(outputs[-1][0].numpy().flatten()[0])
    
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title('Cost Function Value During Iterations')
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()

    return costs[-1]

graphs = create_graph_train_dataset(12)
learning_rate = 0.01
batch_size = 4
epoch = 5
n_thread = 4
n_layers = 2

lstm_cell_trained = TrainLSTM(graphs, learning_rate, batch_size, epoch, n_thread, n_layers)

# Creiamo un grafico di test con 20 nodi
test_graph = create_test_graph(20)
graph_cost = qaoa_maxcut_graph(test_graph, n_layers=n_layers)

#DEBUG
print('1')
#

# Eseguiamo il test del modello
final_cost = test_model(lstm_cell_trained, graph_cost, n_layers=n_layers)

# Stampa il costo finale ottenuto dal modello
print("Final cost for the test graph:", final_cost)

# Parameters are randomly initialized
x = tf.Variable(tf.ones((2, 2)))

# We set the optimizer to be a Stochastic Gradient Descent
opt = tf.keras.optimizers.SGD(learning_rate=0.01)
step = 15

#DEBUG
print('SGD')
#

# Training process
steps = []
sdg_losses = []
for _ in range(step):
    with tf.GradientTape() as tape:
        loss = graph_cost(x)

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

plt.savefig("cost_function_plot.png")  # Salvataggio dell'immagine finale

import pennylane as qml
from pennylane import qaoa
import tensorflow as tf
import networkx as nx
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
    
    outputs = [hybrid_iteration([initial_cost, initial_params, initial_h, initial_c], graph_cost, lstm_cell, n_layers)]
    for _ in range(1, num_iterations):
        outputs.append(hybrid_iteration(outputs[-1], graph_cost, lstm_cell, n_layers))
    costs = [output[0] for output in outputs]
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

def Forward(params, graph_cost, lstm_cell, n_layers):
    return recurrent_loop(graph_cost, lstm_cell, n_layers=n_layers, intermediate_steps=False, num_iterations=10)

def loss_impr(initial_cost, final_cost):
    return observed_improvement_loss([initial_cost, final_cost])

def Backward(tape, loss, lstm_cell):
    return tape.gradient(loss, lstm_cell.trainable_weights)

def update(opt, lstm_cell, grads, learning_rate, batch_size):
    opt.apply_gradients(zip(grads, lstm_cell.trainable_weights))

def TrainLSTM(graphs, learning_rate, batch_size, epoch, n_layers):
    lstm_cell = InitializeParameters(n_layers)
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for e in range(epoch):
        #for graph_batch in iterate_minibatches(graphs, batch_size, shuffle=True):
            #for graph in graph_batch:
        for graph in graphs:
            graph_cost = qaoa_maxcut_graph(graph, n_layers=n_layers)
            initial_cost = tf.ones(shape=(1, 1))
            with tf.GradientTape() as tape:
                loss = Forward(initial_cost, graph_cost, lstm_cell, n_layers=n_layers) 
            grads = Backward(tape, loss, lstm_cell)
            update(opt, lstm_cell, grads, learning_rate, batch_size)
            del tape  # Release tape memory
        tf.keras.backend.clear_session()  # Clear session to free memory
    return lstm_cell

def create_test_graph(n_nodes=20):
    k = random.randint(3, n_nodes-1)
    edge_prob = k / n_nodes
    new_graph = nx.gnp_random_graph(n_nodes, edge_prob)
    plt.figure(figsize=(8, 6))
    nx.draw(new_graph, with_labels=True, node_color='skyblue', node_size=700, edge_color='gray')
    plt.title("Test Graph")
    plt.savefig("test_graph.png")
    plt.close()
    return new_graph

def test_model(lstm_cell_trained, graph_cost, n_layers=2, num_iterations=10, output_filename="serial_NN_cost_function_plot.png"):
    start_zeros = tf.zeros(shape=(2 * n_layers, 1))
    res = recurrent_loop(graph_cost, lstm_cell_trained, intermediate_steps=True)
    # Inizializza le variabili per memorizzare i valori di guess
    guess_list = []
    guess_list.append(start_zeros)

    # Esegui 10 iterazioni
    for i in range(10):
        guess = res[i]  # Supponiamo che res abbia almeno 10 elementi
        guess_list.append(guess)

    # Losses from the hybrid LSTM model
    lstm_losses = [graph_cost(tf.reshape(guess, shape=(2, n_layers))) for guess in guess_list]
    
    plt.figure(figsize=(10, 6))
    plt.plot(lstm_losses, marker='o', linestyle='-', color='b')
    plt.xlabel('Iteration')
    plt.ylabel('Cost Function Value')
    plt.title('Cost Function Value During Iterations')
    plt.grid(True)
    plt.savefig(output_filename)
    plt.close()

    return lstm_losses

# Main script
graphs = create_graph_train_dataset(20)
learning_rate = 0.1
batch_size = 20  # Reduce batch size to save memory
epoch = 10  # Reduce epochs to save memory
n_layers = 2

lstm_cell_trained = TrainLSTM(graphs, learning_rate, batch_size, epoch, n_layers)

# Creiamo un grafico di test con 10 nodi per ridurre la memoria
test_graph = create_test_graph(15)
graph_cost = qaoa_maxcut_graph(test_graph, n_layers=n_layers)

# Eseguiamo il test del modello
lstm_losses = test_model(lstm_cell_trained, graph_cost, n_layers=n_layers)

# Stampa il costo finale ottenuto dal modello
print("Final cost for the test graph:", lstm_losses[-1])

# Parameters are randomly initialized
x = tf.Variable(tf.ones(shape=(2, 2)))

# We set the optimizer to be a Stochastic Gradient Descent
opt = tf.keras.optimizers.SGD(learning_rate=0.1)
step = 10

# Training process
sdg_losses = []
for i in range(step):
    with tf.GradientTape() as tape:
        loss = graph_cost(x)
    sdg_losses.append(loss)
    gradients = tape.gradient(loss, [x])
    opt.apply_gradients(zip(gradients, [x]))
    del tape  # Release tape memory
    print(f"Step {i+1} - Loss = {loss}")

# Print final results
print(f"Final cost function: {graph_cost(x).numpy().flatten()[0]}")
print(f"Optimized angles: {x.numpy()}")

# Plot both curves
plt.figure(figsize=(10, 6))
plt.plot(sdg_losses, color="orange", lw=3, label="SGD")
plt.plot(lstm_losses, color="blue", lw=3, ls="-.", label="LSTM")
plt.grid(ls="--", lw=2, alpha=0.25)
plt.legend()
plt.ylabel("Cost function", fontsize=12)
plt.xlabel("Iteration", fontsize=12)
plt.savefig("serial_cost_function_plot.png")  # Save the final image
#plt.show()

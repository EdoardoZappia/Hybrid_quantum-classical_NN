# Quantum Machine Learning
import pennylane as qml
from pennylane import qaoa

# Classical Machine Learning
import tensorflow as tf
from tf.keras.layers import LSTMCell
# Generation of graphs
import networkx as nx

# Parallelization
import threading

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
        n_nodes= random.randint(6, 9)
        k = random.randint(3, n_nodes-1)
        edge_prob = k / n_nodes
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
        
        dataset.append(G)
    return dataset

def create_graph_test_dataset(num_graphs):
    dataset = []
    n_nodes=12
    for _ in range(num_graphs):
        k = random.randint(3, n_nodes-1)
        edge_prob = k / n_nodes
        G = nx.erdos_renyi_graph(n_nodes, edge_prob)
        
        dataset.append(G)
    return dataset

def qaoa_maxcut_graph(graph, n_layers=2):
    """Compute the maximum cut of a graph using QAOA."""
    # Number of nodes in the graph
    n_nodes = graph.number_of_nodes()
    # Initialize the QAOA device
    dev = qml.device("default.qubit.tf", wires=n_nodes) #, analytic=True)
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

    # Unpack the input list containing the previous cost, parameters,
    # and hidden states (denoted as 'h' and 'c').
    prev_cost = inputs[0]
    prev_params = inputs[1]
    prev_h = inputs[2]
    prev_c = inputs[3]

    # Concatenate the previous parameters and previous cost to create new input
    new_input = tf.concat([prev_cost, prev_params], axis=-1)

    # Call the LSTM cell, which outputs new values for the parameters along
    # with new internal states h and c
    new_params, [new_h, new_c] = cell(new_input, states=[prev_h, prev_c])

    # Reshape the parameters to correctly match those expected by PennyLane
    _params = tf.reshape(new_params, shape=(2, n_layers))

    # Performing one calculation with the quantum circuit with new parameters
    _cost = graph_cost(_params)

    # Reshape to be consistent with other tensors
    new_cost = tf.reshape(tf.cast(_cost, dtype=tf.float32), shape=(1, 1))

    return [new_cost, new_params, new_h, new_c]

def recurrent_loop(graph_cost, n_layers=2, intermediate_steps=False, num_iterations=10):
    """Creates the recurrent loop for the Recurrent Neural Network."""
    # Initialize starting all inputs (cost, parameters, hidden states) as zeros.
    initial_cost = tf.ones(shape=(1, 1))
    initial_params = tf.ones(shape=(1, 2 * n_layers))
    initial_h = tf.ones(shape=(1, 2 * n_layers))
    initial_c = tf.ones(shape=(1, 2 * n_layers))

    # Initialize the output list with the initial state
    outputs = [hybrid_iteration([initial_cost, initial_params, initial_h, initial_c], graph_cost, n_layers)]

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

# Funzioni di inizializzazione e aggiornamento dei parametri
def InitializeParameters():
    # Inizializza i parametri del modello LSTM
    return tf.keras.layers.LSTMCell(2 * n_layers)

def ParametersCopy(src, dest):
    # Copia i parametri dal modello sorgente al modello di destinazione
    dest.set_weights(src.get_weights())

def ZeroInitialize():
    # Inizializza i gradienti a zero
    return [tf.zeros_like(var) for var in cell.trainable_variables]

def getBatch(M, di):
    # Suddivide il dataset di in batch di dimensione M
    return [di[i:i + M] for i in range(0, len(di), M)]

def Forward(params, graph_cost, n_layers):
    # Calcola la predizione del modello per l'input x usando i parametri Θ
    return recurrent_loop(graph_cost, n_layers=n_layers, intermediate_steps=False, num_iterations=10)

def loss_impr(initial_cost, final_cost):
    # Calcola la funzione di perdita di entropia
    return observed_improvement_loss([initial_cost, final_cost])

def Backward(tape, loss, cell):
    # Calcola i gradienti dei parametri rispetto alla perdita
    return tape.gradient(loss, cell.trainable_weights)

def update(opt, cell, grads, learning_rate, batch_size):
    # Aggiorna i parametri Θ con i gradienti dΘ usando il learning rate λ
    opt.apply_gradients(zip(grads, cell.trainable_weights))

# Funzione da eseguire in ogni thread
def ThreadCode(di, cell, learning_rate, batch_size):
    B = getBatch(batch_size, di)
    for bi in B:
        for graph in bi:
            graph_cost = qaoa_maxcut_graph(graph, n_layers=n_layers)
            initial_cost = tf.ones(shape=(1, 1))
            with tf.GradientTape() as tape:
                final_cost = Forward(initial_cost, graph_cost, n_layers=n_layers)
                loss = loss_impr(initial_cost, final_cost)
            grads = Backward(tape, loss, cell)
                
                # Blocco mutex per l'aggiornamento dei parametri globali
        with threading.Lock():
            update(opt, cell, grads, learning_rate, batch_size)

# Algoritmo di training principale
def TrainLSTM(graphs, learning_rate, batch_size, epoch, n_thread, n_layers):
    cell = InitializeParameters()  # Inizializza i parametri del modello
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    e = 1
    
    while e <= epoch:
        # Suddividi i dati in p partizioni
        partitions = np.array_split(graphs, n_thread)
        
        threads = []
        for di in partitions:
            t = threading.Thread(target=ThreadCode, args=(di, cell, learning_rate, batch_size))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()
        
        e += 1

graphs = create_graph_train_dataset(20)
learning_rate = 0.01
batch_size = 4 
epoch = 10
n_thread = 4
n_layers = 2

TrainLSTM(graphs, learning_rate, batch_size, epoch, n_thread, n_layers)

new_graph = nx.gnp_random_graph(20, p=3 / 7)
new_cost = qaoa_maxcut_graph(new_graph)

nx.draw(new_graph)
plt.savefig("../test_graph.png")  # Specifica il percorso e il nome del file immagine
plt.close()  # Chiudi la figura per liberare la memoria

start_zeros = tf.zeros(shape=(2 * n_layers, 1))

# Inizializza le variabili per memorizzare i valori di guess
guess_list = []
guess_list.append(start_zeros)

# Esegui 10 iterazioni
for i in range(10):
    guess = res[i]  # Supponiamo che res abbia almeno 10 elementi
    guess_list.append(guess)

# Losses from the hybrid LSTM model
lstm_losses = [new_cost(tf.reshape(guess, shape=(2, n_layers))) for guess in guess_list]

fig, ax = plt.subplots()

plt.plot(lstm_losses, color="blue", lw=3, ls="-.", label="LSTM")

plt.grid(ls="--", lw=2, alpha=0.25)
plt.ylabel("Cost function", fontsize=12)
plt.xlabel("Iteration", fontsize=12)
plt.legend()
ax.set_xticks([0, 5, 10, 15, 20])
#plt.show()

plt.savefig("../results.png")  # Specifica il percorso e il nome del file immagine
plt.close()  # Chiudi la figura per liberare la memoria

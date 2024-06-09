import numpy as np
import matplotlib.pyplot as plt

from qiskit import Aer, QuantumCircuit
from qiskit.opflow import Z, I, StateFn, PauliSumOp, AerPauliExpectation, ListOp, Gradient
from qiskit.utils import QuantumInstance
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap
from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B, ADAM

from qiskit_machine_learning.neural_networks import TwoLayerQNN, CircuitQNN
from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
from qiskit_machine_learning.algorithms.regressors import NeuralNetworkRegressor, VQR


from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split




quantum_instance = QuantumInstance(Aer.get_backend('qasm_simulator'), shots=1024)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 15})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
#    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    #%%


X_dataset=np.loadtxt('fault.txt')
Y_dataset=np.loadtxt('faultlabel.txt')

#scaler_x = MinMaxScaler()
scaler_x = StandardScaler()

X_train, X_test, Y_train, Y_test = train_test_split(X_dataset, Y_dataset, test_size=0.2)

num_qubits = 9

X_train_adj = scaler_x.fit_transform(X_train)
X_test_adj  = scaler_x.fit_transform(X_test)


y_train = Y_train[:, -1]      # in { 0,  1}
y_test  = Y_test[:,-1]        # in { 0,  1}


### TEST SAMPLE
#num_samples = 20
#X_train = 2*np.random.rand(num_samples, num_qubits) - 1
#y01 = 1*(np.sum(X, axis=1) >= 0)  # in { 0,  1}

y_train_adj = 2*y_train-1    # in {-1, +1}
y_test_adj  = 2*y_test-1     # in {-1, +1}

#Classification with the an OpflowQNN
# construct QNN
# specify the feature map
fm = ZZFeatureMap(num_qubits, reps=2)
fm.draw(output='mpl')

# specify the ansatz
ansatz = RealAmplitudes(num_qubits, reps=1)
ansatz.draw(output='mpl')

# specify the observable
observable = PauliSumOp.from_list([('Z'*num_qubits, 1)])
print(observable)

# define two layer QNN
qnn = TwoLayerQNN(num_qubits, 
                   feature_map=fm, 
                   ansatz=ansatz, 
                   observable=observable, quantum_instance=quantum_instance)
print("Created Two Layer QNN")

inputs = np.random.rand(qnn.num_qubits)
weights = np.random.rand(qnn.num_weights)

# QNN forward pass
qnn.forward(inputs, weights)

# QNN backward pass
qnn.backward(inputs, weights)

print("Construct NN Classifier")
# construct neural network classifier
#opflow_classifier = NeuralNetworkClassifier(qnn, optimizer=L_BFGS_B(maxiter=50))
#opflow_classifier = NeuralNetworkClassifier(qnn, optimizer=ADAM(maxiter=1))
opflow_classifier = NeuralNetworkClassifier(qnn, optimizer=COBYLA(maxiter=1))

print("Fit data to NN Classifier")
# fit classifier to data
fit = opflow_classifier.fit(X_train_adj, y_train_adj)

#plt.plot(fit._fit_result[0])
#plt.xlabel("Iterations")
#plt.ylabel("Residuals")
#plt.show()

print("Score Classifier")
# score classifier
score = opflow_classifier.score(X_train_adj, y_train_adj)
print(score)

print("Evaluate")
# evaluate data points
y_predict = opflow_classifier.predict(X_test_adj)

cm = confusion_matrix(y_true=y_test_adj, y_pred=y_predict,normalize='true')

plot_confusion_matrix(cm , 
                      normalize    = False,
                      target_names = ['Normal', 'Faulty'],
                      title        = "Confusion Matrix")
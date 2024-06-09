import numpy as np
from matplotlib import pyplot as plt
from keras.utils import normalize, to_categorical
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from keras import optimizers
import keras.backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Input
from keras import regularizers
from keras.models import Model
from keras.initializers import he_normal
from keras.losses import categorical_crossentropy
from sklearn.metrics import accuracy_score, f1_score
import sklearn.metrics as metrics
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Activation, Dropout, Input
from keras.utils import normalize, to_categorical
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
import os
#%%
X_dataset=np.loadtxt('fault.txt')
Y_dataset=np.loadtxt('faultlabel.txt')

#scaler_x = MinMaxScaler()
scaler_x = StandardScaler()

X_dataset = scaler_x.fit_transform(X_dataset)

X_train, X_test, y_train, y_test = train_test_split(X_dataset, Y_dataset, test_size=0.2)
#%%

model = Sequential()
model.add(Dense(6, input_shape=(9,)))
model.add(Dense(6, activation='tanh'))
model.add(Dense(6, activation='tanh'))
model.add(Dense(2, activation='sigmoid'))
model.summary()
wts= model.get_weights()
#%%
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

#%%
history = model.fit(
    X_train,
    y_train,
    epochs=300,
    batch_size=128,
    validation_split=0.1,
    verbose = 1,
    shuffle=True
)
#%%
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'y', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#%%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'r', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
y_prob = model.predict(X_test) 
Y_prob = np.argmax(y_prob,axis = 1)
Y_test = np.argmax(y_test,axis = 1)

ytrain_vanilla = model.predict(X_train) 
Ytrain_vanilla = np.argmax(ytrain_vanilla,axis = 1)
Y_train = np.argmax(y_train,axis = 1)
#%%
test_acc_vanilla_test = accuracy_score(Y_test, Y_prob)

train_acc_vanilla_train = accuracy_score(Y_train, Ytrain_vanilla)

#%%
cm = metrics.confusion_matrix(Y_test,Y_prob)
print(cm)

#%%
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

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.rcParams.update({'font.size': 15})
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    #plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=90)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
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
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    #%%

plot_confusion_matrix(cm , 
                      normalize    = False,
                      target_names = ['Normal', 'Faulty'],
                      title        = "Confusion Matrix")
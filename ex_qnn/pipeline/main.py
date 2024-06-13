import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

n_epochs = 5   # Number of optimization epochs
n_layers = 1   # Number of random layers
n_train = 30  # Size of the train dataset
n_test = 10  # Size of the test dataset
max_filters = 30

q_histories = []
# c_histories = []

for num_filters in range(1, max_filters, 3):
    # num_filters = 20

    SAVE_PATH = "../../../research_data/qnn_data/"  # Data saving folder
    PREPROCESS = True           # If False, skip quantum processing and load data from SAVE_PATH
    np.random.seed(0)           # Seed for NumPy random number generator
    tf.random.set_seed(0)       # Seed for TensorFlow random number generator


    mnist_dataset = keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

    # Reduce dataset size
    train_images = train_images[:n_train]
    train_labels = train_labels[:n_train]
    test_images = test_images[:n_test]
    test_labels = test_labels[:n_test]

    # Normalize pixel values within 0 and 1
    train_images = train_images / 255
    test_images = test_images / 255

    # Add extra dimension for convolution channels
    train_images = np.array(train_images[..., tf.newaxis], requires_grad=False)
    test_images = np.array(test_images[..., tf.newaxis], requires_grad=False)

    dev = qml.device("lightning.qubit", wires=4)
    # Random circuit parameters
    rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))

    @qml.qnode(dev)
    def circuit(phi):
        # Encoding of 4 classical input values
        for j in range(4):
            qml.RY(np.pi * phi[j], wires=j)
        # print("qml.RY(np.pi * phi[j], wires = j") 
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(4)))
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(4)]


    def quanv(image, num_filters):
        """Convolves the input image with many applications of the same quantum circuit."""
        tot = np.zeros((14, 14, 4, num_filters))
        
        for i in range(num_filters):
            out = np.zeros((14, 14, 4))
        
            # Loop over the coordinates of the top-left pixel of 2X2 squares
            for j in range(0, 28, 2):
                for k in range(0, 28, 2):
                    # Process a squared 2x2 region of the image with a quantum circuit
                    q_results = circuit(
                        [
                            image[j, k, 0],
                            image[j, k + 1, 0],
                            image[j + 1, k, 0],
                            image[j + 1, k + 1, 0]
                        ]
                    )
                    # Assign expectation values to different channels of the output pixel (j/2, k/2)
                    for c in range(4):
                        tot[j // 2, k // 2, c, i] = q_results[c]
        return tot


    if PREPROCESS == True:
        q_train_images = []
        print("Quantum pre-processing of train images:")
        for idx, img in enumerate(train_images):
            print("{}/{}        ".format(idx + 1, n_train), end="\r")
            # _q_train_images = np.array_split(quanv(img, num_filters), num_filters, axis = 3)
            # print(_q_train_images[0].shape)
            for img in quanv(img, num_filters):
                q_train_images.append(img)
        q_train_images = np.asarray(q_train_images)

        q_test_images = []
        print("\nQuantum pre-processing of test images:")
        for idx, img in enumerate(test_images):
            print("{}/{}        ".format(idx + 1, n_test), end="\r")
            # _q_test_images = np.array_split(quanv(img, num_filters), num_filters, axis = 3)
            for img in quanv(img, num_filters):
                q_test_images.append(img)
        q_test_images = np.asarray(q_test_images)

        # Save pre-processed images
        np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
        np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


    # Load pre-processed images
    q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
    q_test_images = np.load(SAVE_PATH + "q_test_images.npy")


    print(q_train_images.shape)
    q_train_images = q_train_images.reshape((n_train, 14, 14, 4 * num_filters))
    print(q_train_images.shape)

    print(q_test_images.shape)
    q_test_images = q_test_images.reshape((n_test, 14, 14, 4 * num_filters))
    print(q_test_images.shape)

    def MyModel():
        """Initializes and returns a custom Keras model
        which is ready to be trained."""
        model = keras.models.Sequential([
            keras.layers.Conv2D(num_filters * 4, (2,2), activation = 'relu', input_shape=(14, 14, num_filters * 4)),
            keras.layers.AveragePooling2D((2,2), padding = "same"),
            keras.layers.Conv2D(num_filters * 8, (2,2), activation = 'relu'),
            keras.layers.Flatten(),
            keras.layers.Dense(num_filters * 4, activation = 'relu'),
            keras.layers.Dense(10, activation="softmax")
        ])

        model.compile(
            optimizer='adam',
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    q_model = MyModel()

    q_history = q_model.fit(
        q_train_images,
        train_labels,
        validation_data=(q_test_images, test_labels),
        batch_size=10,
        epochs=n_epochs,
        verbose=2,
    )

    # c_model = MyModel()

    # c_history = c_model.fit(
    #     train_images,
    #     train_labels,
    #     validation_data=(test_images, test_labels),
    #     batch_size=10,
    #     epochs=n_epochs,
    #     verbose=2,
    # )

    q_histories.append(q_history)
    # c_histories.append(c_history)

print(q_histories[0].history["val_accuracy"])
accuracies_q = [i.history["val_accuracy"][len(i.history["val_accuracy"])-1] for i in q_histories]
# accuracies_c = [i["val_accuracy"][len(i["val_accuracy"])] for i in c_histories]

loss_q = [i.history["val_loss"][len(i.history["val_loss"])-1] for i in q_histories]
# loss_c = [i["val_loss"][len(i["val_loss"])] for i in c_histories]

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

ax1.plot(accuracies_q, "-ob", label="With quantum layer")
# ax1.plot(accuracies_c, "-og", label="Without quantum layer")
ax1.set_ylabel("Accuracy")
ax1.set_ylim([0, 1])
ax1.set_xlabel("Num Filters (scaled by 1/3)")
ax1.legend()

ax2.plot(loss_q, "-ob", label="With quantum layer")
# ax2.plot(loss_c, "-og", label="Without quantum layer")
ax2.set_ylabel("Loss")
ax2.set_ylim(top=2.5)
ax2.set_xlabel("Num Filters (scaled by 1/3)")
ax2.legend()
plt.tight_layout()
plt.show()

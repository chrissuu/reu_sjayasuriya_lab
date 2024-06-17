import pennylane as qml
from pennylane import numpy as np
from pennylane.templates import RandomLayers
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

n_epochs = 10   # Number of optimization epochs
n_layers = 1   # Number of random layers
n_train = 3000  # Size of the train dataset
n_test = 2000  # Size of the test dataset
max_filters = 80

q_histories = []
# c_histories = []
q_img_history_trn = []
q_img_history_tst = []


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

def generate_data(train_images, n_layers, iters, skip):
    train_datasets = [[[] for img in train_images] for i in range(iters)]
    stores = [[] for i in range(len(train_images))]

    dev = qml.device("lightning.qubit", wires=4)
    # Random circuit parameters
    

    @qml.qnode(dev)
    def circuit(phi, rand_params):
        # Encoding of 4 classical input values
        for j in range(4):
            qml.RY(np.pi * phi[j], wires=j)
        # print("qml.RY(np.pi * phi[j], wires = j") 
        # Random quantum circuit
        RandomLayers(rand_params, wires=list(range(4)))
        # Measurement producing 4 classical output values
        return [qml.expval(qml.PauliZ(j)) for j in range(4)]

    #produces 4 kerneled images of size 14,14 per image
    def quanv(image, rand_params):
        """Convolves the input image with many applications of the same quantum circuit."""
        tot = np.zeros((14, 14, 4))
        
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
                    ], 
                    rand_params
                )
                # Assign expectation values to different channels of the output pixel (j/2, k/2)
                for c in range(4):
                    tot[j // 2, k // 2, c] = q_results[c]
        return tot
        
    for num_filters in range(skip * iters): 
        # for skip*iters iterations, create a random circuit 
        # and use that circuit to generate 4 kerneled images
        # when this loop exits, there would be skip * iters * 4 kerneled images,
        # imgs, in stores[imgs]
        
        rand_params = np.random.uniform(high=2 * np.pi, size=(n_layers, 4))
        for idx, img in enumerate(train_images):
            stores[idx].append(quanv(img, rand_params))
    
    # from the previous loop, we now want to put these imgs in datasets
    # i loops over the number of datasets (iters)

    for i in range(0, iters):
        
        # for each img array in stores, 
        # enumerating by idx, we take a subarray of size i * skip, copy, then reshape to desired size
        # appropriate data with appropriate idx is added
        for idx, img_array in enumerate(stores):
            temp = np.array(img_array[0:(i+1) * skip]).copy().reshape((14, 14, 4 * (i+1) * skip))
            train_datasets[i][idx].append(temp)

    return train_datasets

skip = 3
iters = 2
n_layers = 1
datasets = generate_data(train_images=train_images, n_layers = n_layers, iters = iters, skip = skip)

for i, dataset in enumerate(datasets):
    datasets[i] = np.array(np.array(dataset).reshape((len(train_images), 14, 14, skip * (i+1) * 4)))

for dataset in datasets:

    print((dataset).shape)


def MyModel(num_filters):
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

for i in range(0, iters):

    q_history = q_model.fit(
        datasets[i],
        train_labels,
        validation_data=(test_images, test_labels),
        batch_size=10,
        epochs=n_epochs,
        verbose=2,
    )
# for num_filters in range(1, max_filters, 3):
#     # num_filters = 20



    
#     if PREPROCESS == True:
#         q_train_images = []
#         print("Quantum pre-processing of train images:")
#         for idx, img in enumerate(train_images):
#             print("{}/{}        ".format(idx + 1, n_train), end="\r")
#             # _q_train_images = np.array_split(quanv(img, num_filters), num_filters, axis = 3)
#             # print(_q_train_images[0].shape)
#             for img in quanv(img, _num_filters_const):
#                 q_train_images.append(img)
#         q_train_images = np.asarray(q_train_images)

#         q_test_images = []
#         print("\nQuantum pre-processing of test images:")
#         for idx, img in enumerate(test_images):
#             print("{}/{}        ".format(idx + 1, n_test), end="\r")
#             # _q_test_images = np.array_split(quanv(img, num_filters), num_filters, axis = 3)
#             for img in quanv(img, _num_filters_const):
#                 q_test_images.append(img)
#         q_test_images = np.asarray(q_test_images)

#         # Save pre-processed images
#         np.save(SAVE_PATH + "q_train_images.npy", q_train_images)
#         np.save(SAVE_PATH + "q_test_images.npy", q_test_images)


#     # Load pre-processed images
#     q_train_images = np.load(SAVE_PATH + "q_train_images.npy")
#     q_test_images = np.load(SAVE_PATH + "q_test_images.npy")


#     print(q_train_images.shape)
#     q_train_images = q_train_images.reshape((n_train, 14, 14, 4 * _num_filters_const))
#     print(q_train_images.shape)

#     q_img_history_trn.append(q_train_images)

#     print(q_test_images.shape)
#     q_test_images = q_test_images.reshape((n_test, 14, 14, 4 * _num_filters_const))
#     print(q_test_images.shape)

#     q_img_history_tst.append(q_test_images)
    
#     print("len tst", len(q_img_history_tst))
#     print("len trn", len(q_img_history_trn))
#     if len(q_img_history_tst) > 1:
#         q_train_images = np.stack((np.array(q_img_history_trn)), axis = 4)
#         q_test_images = np.stack((np.array(q_img_history_tst)), axis = 4)

#     print(q_train_images.shape)
#     print(q_test_images.shape)
#     def MyModel():
#         """Initializes and returns a custom Keras model
#         which is ready to be trained."""
#         model = keras.models.Sequential([
#             keras.layers.Conv2D(_num_filters_const * 4, (2,2), activation = 'relu', input_shape=(14, 14, _num_filters_const * 4)),
#             keras.layers.AveragePooling2D((2,2), padding = "same"),
#             keras.layers.Conv2D(_num_filters_const * 8, (2,2), activation = 'relu'),
#             keras.layers.Flatten(),
#             keras.layers.Dense(_num_filters_const * 4, activation = 'relu'),
#             keras.layers.Dense(10, activation="softmax")
#         ])

#         model.compile(
#             optimizer='adam',
#             loss="sparse_categorical_crossentropy",
#             metrics=["accuracy"],
#         )
#         return model

#     q_model = MyModel()

    
#     q_history = q_model.fit(
#         q_train_images,
#         train_labels,
#         validation_data=(q_test_images, test_labels),
#         batch_size=10,
#         epochs=n_epochs,
#         verbose=2,
#     )

#     # c_model = MyModel()

#     # c_history = c_model.fit(
#     #     train_images,
#     #     train_labels,
#     #     validation_data=(test_images, test_labels),
#     #     batch_size=10,
#     #     epochs=n_epochs,
#     #     verbose=2,
#     # )

#     q_histories.append(q_history)
#     # c_histories.append(c_history)

# print(q_histories[0].history["val_accuracy"])
# accuracies_q = [i.history["val_accuracy"][len(i.history["val_accuracy"])-1] for i in q_histories]
# # accuracies_c = [i["val_accuracy"][len(i["val_accuracy"])] for i in c_histories]

# loss_q = [i.history["val_loss"][len(i.history["val_loss"])-1] for i in q_histories]
# # loss_c = [i["val_loss"][len(i["val_loss"])] for i in c_histories]

# import matplotlib.pyplot as plt
# plt.style.use("seaborn-v0_8")
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 9))

# ax1.plot(accuracies_q, "-ob", label="With quantum layer")
# # ax1.plot(accuracies_c, "-og", label="Without quantum layer")
# ax1.set_ylabel("Accuracy")
# ax1.set_ylim([0, 1])
# ax1.set_xlabel("Num Filters (scaled by 1/3)")
# ax1.legend()

# ax2.plot(loss_q, "-ob", label="With quantum layer")
# # ax2.plot(loss_c, "-og", label="Without quantum layer")
# ax2.set_ylabel("Loss")
# ax2.set_ylim(top=2.5)
# ax2.set_xlabel("Num Filters (scaled by 1/3)")
# ax2.legend()
# plt.tight_layout()
# plt.show()

import keras
from keras.utils import np_utils
import numpy as np
import pandas as pd
from keras.datasets import mnist
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from IPython.display import display
import seaborn as sns

def get_mnist_labels():
    return list(range(0, 10))

def get_mnist_data():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_train, Y_train, X_test, Y_test

def get_fashion_mnist_labels():
    labelNames = ["top", "trouser", "pullover", "dress", "coat",
                  "sandal", "shirt", "sneaker", "bag", "ankle boot"]
    return labelNames

def get_fashion_mnist_data():
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    # Convert 1-dimensional class arrays to 10-dimensional class matrices
    Y_train = np_utils.to_categorical(y_train, 10)
    Y_test = np_utils.to_categorical(y_test, 10)
    return X_train, Y_train, X_test, Y_test


def evaluate(model,X_train, Y_train, X_test, Y_test, classes, print_results=False, plot_results=True):
    # TODO: Graph P-R-F1 per class for seeing where we fail most
    # TODO: print class with lowest and highest P-R-F1
    train_score = model.evaluate(X_train, Y_train, verbose=0)
    test_score = model.evaluate(X_test, Y_test, verbose=0)


    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    train_predictions = np.argmax(train_predictions,axis=1)
    test_predictions = np.argmax(test_predictions, axis=1)
    train_predictions = [classes[p] for p in train_predictions]
    test_predictions = [classes[p] for p in test_predictions]

    y_train = np.argmax(Y_train,axis=1)
    y_train = [classes[p] for p in y_train]
    y_test = np.argmax(Y_test, axis=1)
    y_test = [classes[p] for p in y_test]

    train_precision,train_recall,train_f1,train_support = precision_recall_fscore_support(y_train, train_predictions, average=None,labels=classes)
    test_precision,test_recall,test_f1,test_support = precision_recall_fscore_support(y_test, test_predictions, average=None, labels=classes)

    results_train = pd.DataFrame({"classes":classes,
                                  "average": [None] * len(train_precision),
                                "precision":train_precision,
                                "recall":train_recall,
                                "support":train_support,
                                "data_source":["train"]*len(train_precision)})

    results_test = pd.DataFrame({"classes": classes,
                                 "average":[None]*len(test_precision),
                                  "precision": test_precision,
                                  "recall": test_recall,
                                  "support": test_support,
                                  "data_source": ["test"] * len(test_precision)})

    results = pd.concat((results_train,results_test))

    train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(y_train, train_predictions,
                                                                                             average='micro',
                                                                                             labels=classes)
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(y_test, test_predictions,
                                                                                         average='micro', labels=classes)

    results_train = pd.DataFrame({"classes":[None],
                                  "average": ['micro'],
                                "precision":[train_precision],
                                "recall":[train_recall],
                                "support":[train_support],
                                "data_source":["train"]})

    results_test = pd.DataFrame({"classes": [None],
                                 "average":['micro'],
                                  "precision": [test_precision],
                                  "recall": [test_recall],
                                  "support": [test_support],
                                  "data_source": ["test"]})

    results = pd.concat((results,results_train,results_test))

    # =======

    train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(y_train, train_predictions,
                                                                                             average='macro',
                                                                                             labels=classes)
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(y_test, test_predictions,
                                                                                         average='macro',
                                                                                         labels=classes)

    results_train = pd.DataFrame({"classes": [None],
                                  "average": ['macro'],
                                  "precision": [train_precision],
                                  "recall": [train_recall],
                                  "support": [train_support],
                                  "data_source": ["train"]})

    results_test = pd.DataFrame({"classes": [None],
                                 "average": ['macro'],
                                 "precision": [test_precision],
                                 "recall": [test_recall],
                                 "support": [test_support],
                                 "data_source": ["test"]})

    results = pd.concat((results, results_train, results_test))

    # ==============
    train_precision, train_recall, train_f1, train_support = precision_recall_fscore_support(y_train, train_predictions,
                                                                                             average='weighted',
                                                                                             labels=classes)
    test_precision, test_recall, test_f1, test_support = precision_recall_fscore_support(y_test, test_predictions,
                                                                                         average='weighted',
                                                                                         labels=classes)

    results_train = pd.DataFrame({"classes": [None],
                                  "average": ['weighted'],
                                  "precision": [train_precision],
                                  "recall": [train_recall],
                                  "support": [train_support],
                                  "data_source": ["train"]})

    results_test = pd.DataFrame({"classes": [None],
                                 "average": ['weighted'],
                                 "precision": [test_precision],
                                 "recall": [test_recall],
                                 "support": [test_support],
                                 "data_source": ["test"]})

    results = pd.concat((results, results_train, results_test))

    if print_results:
        print("\n", "=" * 80)
        print("Train Score = ", train_score)
        print("Test Score = ", test_score)
        display(results)

    if plot_results:
        print()
        plt.figure(figsize=(16,6))
        sns.barplot(x="classes",y="precision",hue="data_source",data=results[~pd.isna(results.classes)])
        lower_bound = max(results['precision'].min() - 0.05,0)
        upper_bound = 1.05
        plt.ylim((lower_bound,upper_bound))
        plt.title("Precision per Class")
        plt.show()

        plt.figure(figsize=(16, 6))
        sns.barplot(x="classes", y="recall", hue="data_source", data=results[~pd.isna(results.classes)])
        lower_bound = max(results['recall'].min() - 0.05, 0)
        upper_bound = 1.05
        plt.ylim((lower_bound, upper_bound))
        plt.title("Recall per Class")
        plt.show()



    return train_score,test_score,results


def show_examples(X,y,classes):
    rows = int(np.ceil(len(X)/5))
    fig = plt.figure(figsize=(20, rows*4))
    for idx in np.arange(len(X)):
        img = X[idx]*255
        assert (len(img.shape)==3 and img.shape[2] in [1,3,4]) or len(img.shape)==2
        ax = fig.add_subplot(rows, 5, idx + 1, xticks=[], yticks=[])
        cmap = None
        if (len(img.shape)==3 and img.shape[2]==1) or len(img.shape)==2:
            cmap="binary"
        if len(img.shape)==3 and img.shape[2]==1:
            img = img.reshape((img.shape[0],img.shape[1]))
        ax.imshow(img,cmap=cmap)
        ax.set_title(classes[np.argmax(y[idx])])
    plt.show()

def show_kernel_activation():
    pass
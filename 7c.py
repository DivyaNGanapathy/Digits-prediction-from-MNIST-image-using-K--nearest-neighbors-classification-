
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from collections import defaultdict, Counter

M = loadmat('MNIST_digit_data.mat')

# Setting up dataset as numpy array for faster mathematical operations.
# Only 5000 for faster prediction - but this results into little inaccurate results.
# Try to use all train data for accurate results.
images_train, images_test, labels_train, labels_test = M['images_train'], M['images_test'], M['labels_train'], M[
    'labels_test']

# just to make all random sequences on all computers the same.
np.random.seed(1)

# randomly permute data points
inds = np.random.permutation(images_train.shape[0])
images_train = images_train[inds]
labels_train = labels_train[inds]
labels_train = [x[0] for x in labels_train]

inds = np.random.permutation(images_test.shape[0])
images_test = images_test[inds]
labels_test = labels_test[inds]
labels_test = [x[0] for x in labels_test]


train_images = images_train[0:7000, :]
train_labels = labels_train[0:7000]

# if you want to use only the first 1000 data points.
test_images = images_test[0:1000, :]
test_labels = labels_test[0:1000]


def euclidean_distance(point_a, point_b):
    # Function finds the distance between two images at point a and point b

    return sum((point_a - point_b) ** 2)

#Once the distances are obtained the MajorityPicking function picks the labels that are in majority

def find_majority(votes):
    # defaultdict is a dictionary that initializes values to zero if they don't exist
    counter = defaultdict(int)
    for label in votes:
        counter[label] += 1

    # Find out who was the majority.
    majority_count = max(counter.values())
    for key, value in counter.items():
        if value == majority_count:
            return key


#In the predict function we pick out the labels of images who are closest to the test data.
# k values determines how many of those least distances would be considered

def predict(k, train_images, train_labels, test_image):
    distances = [(euclidean_distance(test_image, image), label)
                 for (image, label) in zip(train_images, train_labels)]

    by_distances = sorted(distances, key=lambda distance: distance[0])

    k_labels = [label for (_, label) in by_distances[:k]]

    return find_majority(k_labels)

#The KNN function is used to tke the predictions and verfify with the test(validation) data if the predictions are correct.
#We also calculate the accuracy and the average accuracy of prediction for each digit.

def kNN(train_images, train_labels, test_images, test_labels, k):
    # Predicting and printing the accuracy
    i = 0
    total_correct = 0
    acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for test_image in test_images:
        pred = predict(k, train_images, train_labels, test_image)
        if pred == test_labels[i]:
            total_correct += 1
            acc[test_labels[i]] = acc[test_labels[i]] + 1
        acc_av = (total_correct / (i + 1)) * 100
        # print('test image['+str(i)+']', '\tpred:', pred, '\torig:', test_labels[i], '\tacc:', str(round(acc_av, 2))+'%')
        i += 1

    # Finds individual label frequency
    test_labels_counter = Counter(test_labels)
    # print(test_labels_counter)
    # Accuracy of individual labels
    acc = [round(x * 100 / test_labels_counter[indx], 2) for indx, x in enumerate(acc)]
    # print(acc)

    return acc, acc_av


k = 3
acurracy = kNN(train_images, train_labels, test_images, test_labels, k)

#Function to plot the accuarcy change with respect to different values of K along with varying number of training points.



def plot_acc_for_k(train_images, train_labels, test_images, test_labels, k):
    # For 10 different data points (30 to 10000)
    training_count = [30, 100, 500, 1000, 2500, 4000, 5500, 7500, 8250, 10000]

    y_acc = []

    for cnt in training_count:
        train_images1 = train_images[0:cnt, :]
        train_labels1 = train_labels[0:cnt]
        acc, acc_av = kNN(train_images1, train_labels1, test_images, test_labels, k)
        print(k, cnt, acc_av)
        y_acc.append(acc_av)

    return training_count, y_acc


# 7c
training_count, y_acc = plot_acc_for_k(images_train, labels_train, test_images, test_labels, k)
plt.plot(training_count, y_acc)
plt.xlabel('No. of Training Data')
plt.ylabel('Accuracy')
# plt.text(500, 10, "k = " + str(k) )
plt.title("k = " + str(k))
plt.show()

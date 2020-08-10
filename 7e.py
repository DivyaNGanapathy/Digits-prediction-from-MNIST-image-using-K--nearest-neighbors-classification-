#!/usr/bin/env python
# coding: utf-8



import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from collections import defaultdict, Counter

M = loadmat('MNIST_digit_data.mat')



images_train, images_test, labels_train, labels_test = M['images_train'], M['images_test'], M['labels_train'], M[
    'labels_test']

# just to make all random sequences on all computers the same.
np.random.seed(1)



inds = np.random.permutation(images_train.shape[0])
images_train = images_train[inds]
labels_train = labels_train[inds]
labels_train = [x[0] for x in labels_train]



inds = np.random.permutation(images_test.shape[0])
images_test = images_test[inds]
labels_test = labels_test[inds]
labels_test = [x[0] for x in labels_test]



# 2000 training data points are chosen and 1000 points from them are used as validation data.

train_images1 = images_train[0:2000, :]
train_labels1 = labels_train[0:2000]
train_images=train_images1[0:1000, :]
train_labels=train_labels1[0:1000]



validation_images = train_images1[1000:2000, :]
validation_labels = train_labels1[1000:2000]




def euclidean_distance(point_a, point_b):
    # Function finds the distance between two images at point a and point b

    return np.sqrt(np.sum((point_a - point_b) ** 2))



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


def predict(k, train_images, train_labels, validation_image):
    distances = [(euclidean_distance(validation_image, image), label)
                 for (image, label) in zip(train_images, train_labels)]

    by_distances = sorted(distances, key=lambda distance: distance[0])

    k_labels = [label for (_, label) in by_distances[:k]]

    return find_majority(k_labels)



#The KNN function is used to tke the predictions and verfify with the test(validation) data if the predictions are correct.
#We also calculate the accuracy and the average accuracy of prediction for each digit.

def kNN(train_images, train_labels, validation_images, validation_labels, k):
    # Predicting and printing the accuracy
    i = 0
    total_correct = 0
    acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for validation_image in validation_images:
        pred = predict(k, train_images, train_labels, validation_image)
        if pred == validation_labels[i]:
            total_correct += 1
            acc[validation_labels[i]] = acc[validation_labels[i]] + 1
        acc_av = (total_correct / (i + 1)) * 100
        i += 1

    # Finds individual label frequency
    test_labels_counter = Counter(validation_labels)
    # print(test_labels_counter)
    # Accuracy of individual labels
    acc = [round(x * 100 / test_labels_counter[indx], 2) for indx, x in enumerate(acc)]
    # print(acc)
    print(pred , acc , acc_av )
    return acc, acc_av



#Function to plot the accuarcy change with respect to different values of K when 1000 training and
# 1000 validation points are chosen

def plot_acc_for_k(train_images, train_labels, validation_images, validation_labels, k):

    y_acc=[]

    acc, acc_av = kNN(train_images, train_labels, validation_images, validation_labels, k)

    y_acc.append(acc_av)
    print(k , acc)

    return y_acc






import time

t0 = time.time()

# Searcing for K
k_array = [1, 2, 3, 5, 10]
#k_array=[1]
y_acc_array = []
for k in k_array:
    y_acc = plot_acc_for_k(images_train, train_labels, validation_images, validation_labels, k)
    print(k_array, y_acc)
    y_acc_array.append(y_acc)
plt.plot(k_array, y_acc_array)
#plt.legend(['k = 1', 'k = 2', 'k = 3', 'k = 5', 'k = 10'], loc='lower right')
plt.xlabel('Value of k ')
plt.ylabel('Accuracy')
plt.show()
t1 = time.time()

total = t1-t0
print(total)







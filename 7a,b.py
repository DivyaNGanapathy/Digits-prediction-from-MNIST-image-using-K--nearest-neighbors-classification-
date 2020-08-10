

import numpy as np
from scipy.io import loadmat
from collections import defaultdict, Counter

M = loadmat('C:\\Users\divya\Desktop\MNIST_digit_data.mat')

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

inds = np.random.permutation(images_test.shape[0])
images_test = images_test[inds]
labels_test = labels_test[inds]

# if you want to use only the first 1000 data points.
train_images = images_train[0:5000, :]
train_labels = labels_train[0:5000, :]

train_labels = [x[0] for x in train_labels]  # see

# if you want to use only the first 1000 data points.
test_images = images_test[0:1000, :]
test_labels = labels_test[0:1000, :]

test_labels = [x[0] for x in test_labels]  # see

#Function to calculate the Euclidean distance between the test data and the exiting labelled point

def euclidean_distance(point_a, point_b):
    # Function finds the distance between two images at point a and point b
    return np.sqrt(np.sum((point_a - point_b) ** 2))

#Once the distances are obtained the MajorityPicking function picks the labels that are in majority

def MajorityPicking(votes):
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

    return MajorityPicking(k_labels)

#The KNN function is used to tke the predictions and verfify with the test(validation) data if the predictions are correct.
#We also calculate the accuracy and the average accuracy of prediction for each digit.
def kNN(train_images, train_labels, test_images, test_labels, k):
    # Predicting and printing the accuracy
    i = 0
    newclass=[]

    total_correct = 0
    acc = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # array of size 10 inizilized to 0 individual acc assumed to 0
    for test_image in test_images:
        pred = predict(k, train_images, train_labels, test_image)
        if pred == test_labels[i]:
            total_correct += 1
            acc[test_labels[i]] = acc[test_labels[i]] + 1


        # percentange of the accurate prediction is
        acc_av = (total_correct / (i + 1)) * 100  # percentange of the accurate prediction is
        i += 1

    # Finds individual label frequency
    test_labels_counter = Counter(test_labels)

    # prints the frequency of each lable
    #print(test_labels_counter)

    # Accuracy of individual labels
    for indx, x in enumerate(acc):

        acc = round(x * 100 / test_labels_counter[indx], 2)
        #print(indx, acc)
        newvalue='Class : '+ str(indx) + ' - ' + ' Accuracy : '+  str(acc)
        newclass.append(newvalue)

    print(newclass)

    print('average accuracy', acc_av)
    return acc, acc_av


k = 5
accuracy= kNN(train_images, train_labels, test_images, test_labels, k)
print('a', accuracy)



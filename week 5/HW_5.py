import numpy as np
import cv2.cv2
from skimage import feature
import matplotlib.pyplot as plt
import os,glob
from scipy import ndimage as nd
from skimage.filters import gabor_kernel
import sklearn.model_selection as model_selection
import re
from sklearn.preprocessing import minmax_scale
import pandas as pd
import statsmodels.api as sm
np.random.seed(0)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


"""
You have to implement an image classification solution for the same dataset as in HT4. There are 16 folders and
16 corresponding classes. You should reconfigure [manually or by a script] this dataset and split it into three folders
(train, validation, test). Feel free to reduce the number of examples for each class in your dataset, if it takes too
long to compute on your hardware.
For this task, you should use extracted features from HT4 and try several classifiers from scikit learn.
We encourage you to use such ML tools like ensemble voting, boost, dimensionality reduction, gridsearch for
hyperparameters deeptuning™ etc. The more trials you take the more you learn.

As an output, you should provide your dataset, and as much precision metrics [calculated on test images] as you will
google (precision, recall, and confusion matrix are required).
"""

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius
    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
            self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
            bins=np.arange(0, self.numPoints + 3),
            range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype(np.float32)
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


def compute_feats(image, kernels):
    # feats = np.zeros((len(kernels), 2), dtype=np.double)
    descriptors = []
    for k, kernel in enumerate(kernels):
        filtered = nd.convolve(image, kernel, mode='wrap')
        descriptors.append(filtered.mean())
        descriptors.append(filtered.var())
        # feats[k, 0] = filtered.mean()
        # feats[k, 1] = filtered.var()
    return descriptors


class Gradient_histogram:
    def __init__(self, numPoints):
        # store the number of points and radius
        self.numPoints = numPoints


    def get_grad_features(self,grad_mag, grad_ang):
        # Init
        angles = grad_ang[grad_mag > 5]
        hist, bins = np.histogram(angles,self.numPoints)

        return hist

    def describe(self, pattern_img, eps=1e-7):
        # Calculate Sobel gradient
        grad_x = cv2.Sobel(pattern_img, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(pattern_img, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag, grad_ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
        hist = self.get_grad_features(grad_mag, grad_ang).astype(np.float32)
        return hist


def extract_features(folder_name):
    # Create dataset
    dataset = []

    # prepare filter bank kernels
    kernels = []
    for theta in range(4):
        theta = theta / 4. * np.pi
        for sigma in (1, 3):
            for frequency in (0.05, 0.25):
                kernel = np.real(gabor_kernel(frequency, theta=theta,
                                              sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    for i in range(16):

        folder_path = r'{}/{}'.format(folder_name, i + 1)

        for filename in glob.glob(os.path.join(folder_path, '*.jpg')):
            descriptors = []

            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

            feature_extractor = LocalBinaryPatterns(256, 1)
            descriptor_template = feature_extractor.describe(img)
            descriptor_template_norm = minmax_scale(descriptor_template, feature_range=(-1, 1))
            descriptors.append(descriptor_template)

            feature_extractor_2 = Gradient_histogram(256)
            descriptor_template_2 = feature_extractor_2.describe(img)
            descriptor_template_2_norm = minmax_scale(descriptor_template_2, feature_range=(-1, 1))
            descriptors.append(descriptor_template_2_norm)

            descriptor_template_3 = compute_feats(img, kernels)
            descriptor_template_3_norm = minmax_scale(descriptor_template_3, feature_range=(-1, 1))
            descriptors.append(descriptor_template_3_norm)

            # add class
            descriptors.append([i + 1])

            flat_descriptors = [item for sublist in descriptors for item in sublist]

            dataset.append(flat_descriptors)

    dataset_arr = np.array([np.array(xi) for xi in dataset], dtype=object)

    y = dataset_arr[:, -1]
    X = dataset_arr[:, :-1]

    df_X = pd.DataFrame.from_records(X)  # Shape (1600, 546)
    series_y = pd.Series(y)

    return df_X, series_y


def h_w_5():
    validation_X, validation_y = extract_features('validation')
    print('validation = ', validation_X.shape, validation_y.shape)

    train_X, train_y = extract_features('train')
    print('train = ', train_X.shape, train_y.shape)

    test_X, test_y = extract_features('test')
    print('test = ', test_X.shape, test_y.shape)


    # RandomForest
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    # Train the Classifier to take the training features and learn how they relate
    clf.fit(np.asarray(train_X), np.asarray(train_y).astype('int'))
    preds_RandomForest = clf.predict(np.asarray(validation_X))
    # Create confusion matrix
    print('RandomForest confusion matrix')
    print(pd.crosstab(validation_y, preds_RandomForest, rownames=['Actual'], colnames=['Predicted']))

    recall_score_RandomForest = recall_score(y_true=np.asarray(validation_y).astype('int'), y_pred=preds_RandomForest,
                                             average='weighted')
    precision_score_RandomForest = precision_score(y_true=np.asarray(validation_y).astype('int'),
                                                   y_pred=preds_RandomForest, average='weighted')
    f1_score_RandomForest = f1_score(y_true=np.asarray(validation_y).astype('int'), y_pred=preds_RandomForest,
                                     average='weighted')
    print('RandomForest scores')
    print('recall_score = ', recall_score_RandomForest)
    print('precision_score = ', precision_score_RandomForest)
    print('f1_score = ', f1_score_RandomForest)
    print()

    # KNeighbors
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(np.asarray(train_X), np.asarray(train_y).astype('int'))
    y_pred = knn.predict(np.asarray(validation_X))
    print('KNeighbors confusion matrix')
    # Create confusion matrix
    print(pd.crosstab(validation_y, y_pred, rownames=['Actual'], colnames=['Predicted']))

    recall_score_KNeighbors = recall_score(y_true=np.asarray(validation_y).astype('int'), y_pred=y_pred,
                                           average='weighted')
    precision_score_KNeighbors = precision_score(y_true=np.asarray(validation_y).astype('int'), y_pred=y_pred,
                                                 average='weighted')
    f1_score_KNeighbors = f1_score(y_true=np.asarray(validation_y).astype('int'), y_pred=y_pred, average='weighted')
    print('KNeighbors scores')
    print('recall_score = ', recall_score_KNeighbors)
    print('precision_score = ', precision_score_KNeighbors)
    print('f1_score = ', f1_score_KNeighbors)
    print()

    # DecisionTree
    clf_DecisionTree = DecisionTreeClassifier()
    clf_DecisionTree.fit(np.asarray(train_X), np.asarray(train_y).astype('int'))
    predicted_DecisionTree = clf_DecisionTree.predict(np.asarray(validation_X))
    print('DecisionTree confusion matrix')
    # Create confusion matrix
    print(pd.crosstab(validation_y, predicted_DecisionTree, rownames=['Actual'], colnames=['Predicted']))

    recall_score_DecisionTree = recall_score(y_true=np.asarray(validation_y).astype('int'),
                                             y_pred=predicted_DecisionTree, average='weighted')
    precision_score_DecisionTree = precision_score(y_true=np.asarray(validation_y).astype('int'),
                                                   y_pred=predicted_DecisionTree, average='weighted')
    f1_score_DecisionTree = f1_score(y_true=np.asarray(validation_y).astype('int'), y_pred=predicted_DecisionTree,
                                     average='weighted')
    print('DecisionTree scores')
    print('recall_score = ', recall_score_DecisionTree)
    print('precision_score = ', precision_score_DecisionTree)
    print('f1_score = ', f1_score_DecisionTree)
    print()

    # SVC
    clf_SVC = SVC(kernel='linear')
    clf_SVC.fit(np.asarray(train_X), np.asarray(train_y).astype('int'))
    predicted_SVC = clf_SVC.predict(np.asarray(validation_X))

    print('SVC confusion matrix')
    # Create confusion matrix
    print(pd.crosstab(validation_y, predicted_SVC, rownames=['Actual'], colnames=['Predicted']))

    recall_score_SVC = recall_score(y_true=np.asarray(validation_y).astype('int'), y_pred=predicted_SVC,
                                    average='weighted')
    precision_score_SVC = precision_score(y_true=np.asarray(validation_y).astype('int'), y_pred=predicted_SVC,
                                          average='weighted')
    f1_score_SVC = f1_score(y_true=np.asarray(validation_y).astype('int'), y_pred=predicted_SVC, average='weighted')
    print('SVC scores')
    print('recall_score = ', recall_score_SVC)
    print('precision_score = ', precision_score_SVC)
    print('f1_score = ', f1_score_SVC)


if __name__ == '__main__':
    h_w_5()


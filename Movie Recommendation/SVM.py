
# coding: utf-8

# In[ ]:


import csv
import math
import numpy as np
import pandas
import sys
import warnings
from collections import Counter
from csv import reader
from sklearn import tree
from sklearn import decomposition
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import check_random_state
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

def load_csv(fileName):
    file = open(fileName, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

def stringColumnToFloat(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def projection_simplex(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

class MulticlassSVM(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1, max_iter=50, tol=0.05, random_state=None, verbose=0):
        self.C = C
        self.max_iter = max_iter
        self.tol = tol,
        self.random_state = random_state
        self.verbose = verbose

    def _partial_gradient(self, X, y, i):
        # Partial gradient for the ith sample.
        g = np.dot(X[i], self.coef_.T) + 1
        g[y[i]] -= 1
        return g

    def _violation(self, g, y, i):
        # Optimality violation for the ith sample.
        smallest = np.inf
        for k in range(g.shape[0]):
            if k == y[i] and self.dual_coef_[k, i] >= self.C:
                continue
            elif k != y[i] and self.dual_coef_[k, i] >= 0:
                continue

            smallest = min(smallest, g[k])

        return g.max() - smallest

    def _solve_subproblem(self, g, y, norms, i):
        Ci = np.zeros(g.shape[0])
        Ci[y[i]] = self.C
        beta_hat = norms[i] * (Ci - self.dual_coef_[:, i]) + g / norms[i]
        z = self.C * norms[i]
        beta = projection_simplex(beta_hat, z)
        return Ci - self.dual_coef_[:, i] - beta / norms[i]

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._label_encoder = LabelEncoder()
        y = self._label_encoder.fit_transform(y)
        n_classes = len(self._label_encoder.classes_)
        self.dual_coef_ = np.zeros((n_classes, n_samples), dtype=np.float64)
        self.coef_ = np.zeros((n_classes, n_features))
        norms = np.sqrt(np.sum(X ** 2, axis=1))
        rs = check_random_state(self.random_state)
        ind = np.arange(n_samples)
        rs.shuffle(ind)
        violation_init = None
        for it in range(self.max_iter):
            violation_sum = 0
            for ii in range(n_samples):
                i = ind[ii]
                if norms[i] == 0:
                    continue
                g = self._partial_gradient(X, y, i)
                v = self._violation(g, y, i)
                violation_sum += v
                if v < 1e-12:
                    continue
                delta = self._solve_subproblem(g, y, norms, i)
                self.coef_ += (delta * X[i][:, np.newaxis]).T
                self.dual_coef_[:, i] += delta
            if it == 0:
                violation_init = violation_sum
            vratio = violation_sum / violation_init
            if vratio < self.tol:
                break
        return self

    def predict(self, X):
        decision = np.dot(X, self.coef_.T)
        pred = decision.argmax(axis=1)
#         print(pred)
        return pred

def calculateTimeStampWeight(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def TFIDFProductValue(row):
    return row['tf']*row['idf']

def CalculateMovieTF(row):
    return row['tag_weightage'] / row['total_movie_weightage']

def calculateIDFData(row, total_movies):
    return math.log10(total_movies / row['count_of_movies'])

def calculateTFIDFData(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(TFIDFProductValue, axis=1)
    return tfidfdata[['movieid','tagid','tfidf']]

def fetchMoviesTagsData():
    allmoviesTagsData =pandas.read_csv("data/mltags.csv")
    min_timestamp = pandas.to_datetime(min(allmoviesTagsData['timestamp']))
    max_timestamp = pandas.to_datetime(max(allmoviesTagsData['timestamp']))
    allmoviesTagsData['timestamp_weightage'] = allmoviesTagsData.apply(calculateTimeStampWeight, axis=1, args=(min_timestamp, max_timestamp))
    allmoviesTagsData['tag_weightage'] = allmoviesTagsData.groupby(['movieid','tagid'])['timestamp_weightage'].transform('sum')
    allmoviesTagsData = allmoviesTagsData[['movieid','tagid','tag_weightage']].drop_duplicates(subset=['movieid','tagid'])
    allmoviesTagsData['total_movie_weightage'] = allmoviesTagsData.groupby(['movieid'])['tag_weightage'].transform('sum')
    allmoviesTagsData['tf'] = allmoviesTagsData.apply(CalculateMovieTF, axis=1)
    taglist = allmoviesTagsData['tagid'].tolist()
    alltagsdata = pandas.read_csv("data/mltags.csv")
    specifictagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]
    specifictagsdata.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
    specifictagsdata['count_of_movies'] = specifictagsdata.groupby('tagid')['movieid'].transform('count')
    specifictagsdata.drop_duplicates(subset=['tagid'], inplace=True)
    moviesdata = pandas.read_csv("data/mlmovies.csv")
    total_movies = moviesdata.shape[0]
    specifictagsdata['idf'] = specifictagsdata.apply(calculateIDFData, axis=1, total_movies=total_movies)
    tfidfdata = calculateTFIDFData(allmoviesTagsData, specifictagsdata[['tagid', 'idf']])
    return tfidfdata

def fetchMoviesDetails(movielist):
    moviedetails = pandas.read_csv("data/mlmovies.csv")
    moviedetails = moviedetails[moviedetails['movieid'].isin(movielist)]
    movienamelist = moviedetails.values.tolist()
    movienamelist = sorted(movienamelist, key=lambda x: x[0])
    return movienamelist

def fetchLabelDetails():
    labeldetails = pandas.read_csv("data/mllabels.csv")
    labellist = labeldetails.values.tolist()
    return labellist

def classificationUsingSVM():
    print("Assigning labels using SVM classifier")
    labelListDetails = fetchLabelDetails()
    modifiedList =[]
    labelNames = sorted(np.unique([item[1] for item in labelListDetails]))
    for i in range (0, len(labelListDetails)):
        modifiedList.append((labelListDetails[i][0],labelListDetails[i][1],labelNames.index(labelListDetails[i][1])))
    modifiedList = sorted(modifiedList, key=lambda x: x[0])
    LabelledListOfMovies = sorted(np.unique([item[0] for item in modifiedList]))
    listOfLabelledIDs =[]
    for ele in modifiedList:
        listOfLabelledIDs.append(ele[2])
    tempDataFrame = pandas.DataFrame(listOfLabelledIDs)
    tempDataFrame.to_csv('ListOfLabelledIDs.csv', index=False, header=False)
    moviesTagsData = fetchMoviesTagsData()
    moviesList = sorted(np.unique(moviesTagsData['movieid'].tolist()))
    movieTagMatrix = moviesTagsData.pivot_table(index='movieid', columns='tagid',values='tfidf', fill_value=0)
    data = []
    for movieID in LabelledListOfMovies:
        i = moviesList.index(movieID)
        data.append(movieTagMatrix.values[i])
    tempDataFrame = pandas.DataFrame(data)
    tempDataFrame.to_csv('LabelledMovieTagData.csv', index=False, header=False)
    tempDataFrame = pandas.DataFrame(movieTagMatrix.values)
    tempDataFrame.to_csv('MovieTagData.csv', index=False, header=False)
    fileName = 'LabelledMovieTagData.csv'
    data = load_csv(fileName)
    for i in range(len(data[0])):
        stringColumnToFloat(data, i)
    fileName = 'ListOfLabelledIDs.csv'
    labels = load_csv(fileName)
    fileName = 'MovieTagData.csv'
    testData = load_csv(fileName)
    for i in range(len(testData[0])):
        stringColumnToFloat(testData, i)
    label = []
    for i in labels:
        label.append(i[0])
    clf = MulticlassSVM(C=0.1, tol=0.01, max_iter=100, random_state=0, verbose=1)
    clf.fit(np.array(data), label)
    labelledResults = clf.predict(np.array(testData))
    labelled =[]
    for i in range(0,len(moviesList)):
        index = int(labelledResults[i])
        labelled.append([moviesList[i],labelNames[index]])
    newLabelsDataFrame = pandas.DataFrame(labelled,columns=['movieid','label'])
#     print(newLabelsDataFrame)
    newLabelsDataFrame.to_csv('ClassificationUsingSVM.csv', index=False)
    print("Classification using n-ary SVM completed. Please check the csv file.")


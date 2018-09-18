
# coding: utf-8

# In[1]:


import sys
import pandas
import numpy
import math
import warnings
from sklearn import tree
from sklearn import decomposition
from collections import Counter
warnings.filterwarnings("ignore")
from random import seed
from random import randrange
from csv import reader

def load_csv(filename):
    file = open(filename, "r")
    lines = reader(file)
    dataset = list(lines)
    return dataset

def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())

def cross_validation_split(dataset, n_folds):
    dataset_split = list()
    dataset_copy = list(dataset)
    fold_size = int(len(dataset) / n_folds)
    for i in range(n_folds):
        fold = list()
        while len(fold) < fold_size:
            index = randrange(len(dataset_copy))
            fold.append(dataset_copy.pop(index))
        dataset_split.append(fold)
    return dataset_split

def accuracy_metric(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == predicted[i]:
            correct += 13
    return correct / float(len(actual)) * 100.0

def test_algo(dataset, test_data, algorithm,*args):
    predicted = algorithm(dataset, test_data, *args)
    #print (predicted)
    return predicted

def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        # print (predicted)
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores

def test_split(index, value, dataset):
    left, right = list(), list()
    for row in dataset:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left, right

def gini_index(groups, classes):
    # count all samples at split point
    n_instances = float(sum([len(group) for group in groups]))
    # sum weighted Gini index for each group
    gini = 0.0
    for group in groups:
        size = float(len(group))
        # avoid divide by zero
        if size == 0:
            continue
        score = 0.0
        # score the group based on the score for each class
        for class_val in classes:
            p = [row[-1] for row in group].count(class_val) / size
            score += p * p
        # weight the group score by its relative size
        gini += (1.0 - score) * (size / n_instances)
    return gini

def get_split(dataset):
    class_values = list(set(row[-1] for row in dataset))
    b_index, b_value, b_score, b_groups = 999, 999, 999, None
    for index in range(len(dataset[0])-1):
        for row in dataset:
            groups = test_split(index, row[index], dataset)
            gini = gini_index(groups, class_values)
            if gini < b_score:
                b_index, b_value, b_score, b_groups = index, row[index], gini, groups
    return {'index':b_index, 'value':b_value, 'groups':b_groups}

def to_terminal(group):
    outcomes = [row[-1] for row in group]
    return max(set(outcomes), key=outcomes.count)

def split(node, max_depth, min_size, depth):
    left, right = node['groups']
    del(node['groups'])
    if not left or not right:
        node['left'] = node['right'] = to_terminal(left + right)
        return
    if depth >= max_depth:
        node['left'], node['right'] = to_terminal(left), to_terminal(right)
        return
    if len(left) <= min_size:
        node['left'] = to_terminal(left)
    else:
        node['left'] = get_split(left)
        split(node['left'], max_depth, min_size, depth+1)
    if len(right) <= min_size:
        node['right'] = to_terminal(right)
    else:
        node['right'] = get_split(right)
        split(node['right'], max_depth, min_size, depth+1)

def build_tree(train, max_depth, min_size):
    root = get_split(train)
    split(root, max_depth, min_size, 1)
    return root

def predict(node, row):
    if row[node['index']] < node['value']:
        if isinstance(node['left'], dict):
            return predict(node['left'], row)
        else:
            return node['left']
    else:
        if isinstance(node['right'], dict):
            return predict(node['right'], row)
        else:
            return node['right']

def decision_tree(train, test, max_depth, min_size):
    tree = build_tree(train, max_depth, min_size)
    predictions = list()
    for row in test:
        prediction = predict(tree, row)
        predictions.append(prediction)
    return(predictions)


def CalculateTimestampWeights(row, min_timestamp, max_timestamp):
    return ((pandas.to_datetime(row['timestamp'])-min_timestamp).days + 1)/((max_timestamp-min_timestamp).days+1)

def CalculateTFIDF(row):
    return row['tf']*row['idf']

def CalculateMovieTF(row):
    return row['tag_weight'] / row['total_movie_weight']

def CalculateMoviesIDF(row, total_movies):
    return math.log10(total_movies / row['count_of_movies'])

def ProcessMovieTFandIDFtoTFIDF(tfdata, idfdata):
    tfidfdata = tfdata.merge(idfdata, on='tagid')
    tfidfdata['tfidf'] = tfidfdata.apply(CalculateTFIDF, axis=1)
    return tfidfdata[['movieid','tagid','tfidf']]

def GetMoviesTagsData():

    mltags =pandas.read_csv("data/mltags.csv")

    min_timestamp = pandas.to_datetime(min(mltags['timestamp']))
    max_timestamp = pandas.to_datetime(max(mltags['timestamp']))
    #
    # mltags = mltags[mltags['movieid'].isin(movieslist)]

    mltags['timestamp_weight'] = mltags.apply(CalculateTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    mltags['tag_weight'] = mltags.groupby(['movieid','tagid'])['timestamp_weight'].transform('sum')

    mltags = mltags[['movieid','tagid','tag_weight']].drop_duplicates(subset=['movieid','tagid'])

    mltags['total_movie_weight'] = mltags.groupby(['movieid'])['tag_weight'].transform('sum')

    # print(mltags)

    mltags['tf'] = mltags.apply(CalculateMovieTF, axis=1)

    taglist = mltags['tagid'].tolist()

    alltagsdata = pandas.read_csv("data/mltags.csv")

    specifictagsdata = alltagsdata[alltagsdata['tagid'].isin(taglist)]

    specifictagsdata.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
    specifictagsdata['count_of_movies'] = specifictagsdata.groupby('tagid')['movieid'].transform('count')
    specifictagsdata.drop_duplicates(subset=['tagid'], inplace=True)

    mlmovies = pandas.read_csv("data/mlmovies.csv")
    total_movies = mlmovies.shape[0]

    specifictagsdata['idf'] = specifictagsdata.apply(CalculateMoviesIDF, axis=1, total_movies=total_movies)

    # print(total_actors)
    # print(specificactortagsdata)

    tfidfdata = ProcessMovieTFandIDFtoTFIDF(mltags, specifictagsdata[['tagid', 'idf']])


    return tfidfdata

def GetMoviesDetails(movielist):

    moviedetails = pandas.read_csv("data/mlmovies.csv")
    moviedetails = moviedetails[moviedetails['movieid'].isin(movielist)]
    movienamelist = moviedetails.values.tolist()
    movienamelist = sorted(movienamelist, key=lambda x: x[0])
    return movienamelist

def GetLabelDetails():

    labeldetails = pandas.read_csv("data/mllabels.csv")
    labellist = labeldetails.values.tolist()
    return labellist

def DecisionTreeClasssifier():
    print("Assigning labels using Decision Tree classifier")

    labeldetailslist = GetLabelDetails()

    modifiedlist =[]

    labelnames = sorted(numpy.unique([item[1] for item in labeldetailslist]))

    for i in range (0, len(labeldetailslist)):
        modifiedlist.append((labeldetailslist[i][0],labeldetailslist[i][1],labelnames.index(labeldetailslist[i][1])))

    modifiedlist = sorted(modifiedlist, key=lambda x: x[0])

    labelledmovieslist = sorted(numpy.unique([item[0] for item in modifiedlist]))

    movietagsdata = GetMoviesTagsData()

    templabellist = []
    maxtagid = max(movietagsdata['tagid'])

    for movieid in labelledmovieslist:
        j =labelledmovieslist.index(movieid)
        templabellist.append([movieid,maxtagid+1,modifiedlist[j][2]])

    labelsdf = pandas.DataFrame(templabellist,columns=['movieid','tagid','tfidf'])

    # print(labelsdf)

    concatmovietagsdata = pandas.concat([movietagsdata,labelsdf])

    movieslist = sorted(numpy.unique(movietagsdata['movieid'].tolist()))

    # moviedetaillist = GetMoviesDetails(movieslist)

    movie_tag_matrix = concatmovietagsdata.pivot_table(index='movieid', columns='tagid',values='tfidf', fill_value=0)
    data = []

    for movieid in labelledmovieslist:
        i =movieslist.index(movieid)
        data.append(movie_tag_matrix.values[i])

    my_df = pandas.DataFrame(data)
    my_df.to_csv('LabelledMovies.csv', index=False, header=False)

    my_df = pandas.DataFrame(movie_tag_matrix.values)
    my_df.to_csv('UnLabelledMovies.csv', index=False, header=False)

    seed(1)

    filename = 'LabelledMovies.csv'
    dataset = load_csv(filename)
    #print(dataset)

    for i in range(len(dataset[0])):
        str_column_to_float(dataset, i)

    n_folds = 5
    max_depth = 5
    min_size = 10
    scores = evaluate_algorithm(dataset, decision_tree, n_folds, max_depth, min_size)

    test_file = 'UnLabelledMovies.csv'
    test_data = load_csv(test_file)

    for i in range(len(test_data[0])):
        str_column_to_float(test_data, i)

    labelresults = test_algo(dataset, test_data, decision_tree, max_depth, min_size)

    labelled =[]

    for i in range(0,len(movieslist)):
        index = int(labelresults[i])
        labelled.append([movieslist[i],labelnames[index]])

    newlabelsdf = pandas.DataFrame(labelled,columns=['movieid','label'])

    print("Classification using Decision Tree completed. Please check the csv file.")


    newlabelsdf.to_csv('ClassificationUsingDecisionTree.csv', index=False)


DecisionTreeClasssifier()


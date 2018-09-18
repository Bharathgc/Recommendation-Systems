
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

    mlTags =pandas.read_csv("data/mltags.csv")

    min_timestamp = pandas.to_datetime(min(mlTags['timestamp']))
    max_timestamp = pandas.to_datetime(max(mlTags['timestamp']))

    mlTags['timestamp_weight'] = mlTags.apply(CalculateTimestampWeights, axis=1, args=(min_timestamp, max_timestamp))

    mlTags['tag_weight'] = mlTags.groupby(['movieid','tagid'])['timestamp_weight'].transform('sum')

    mlTags = mlTags[['movieid','tagid','tag_weight']].drop_duplicates(subset=['movieid','tagid'])

    mlTags['total_movie_weight'] = mlTags.groupby(['movieid'])['tag_weight'].transform('sum')

    mlTags['tf'] = mlTags.apply(CalculateMovieTF, axis=1)

    taglist = mlTags['tagid'].tolist()
    alltagsdata = pandas.read_csv("data/mltags.csv")
    RequiredTagData = alltagsdata[alltagsdata['tagid'].isin(taglist)]

    RequiredTagData.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
    RequiredTagData['count_of_movies'] = RequiredTagData.groupby('tagid')['movieid'].transform('count')
    RequiredTagData.drop_duplicates(subset=['tagid'], inplace=True)

    moviesdata = pandas.read_csv("data/mlmovies.csv")
    total_movies = moviesdata.shape[0]

    RequiredTagData['idf'] = RequiredTagData.apply(CalculateMoviesIDF, axis=1, total_movies=total_movies)

    tfidfdata = ProcessMovieTFandIDFtoTFIDF(mlTags, RequiredTagData[['tagid', 'idf']])

    return tfidfdata

def GetMoviesDetails(movielist):

    mlMovies = pandas.read_csv("data/mlmovies.csv")
    mlMovies = mlMovies[mlMovies['movieid'].isin(movielist)]
    MovieMameList = mlMovies.values.tolist()
    MovieMameList = sorted(MovieMameList, key=lambda x: x[0])
    return MovieMameList

def GetLabels():
    mlLabels = pandas.read_csv("data/mllabels.csv")
    labellist = mlLabels.values.tolist()
    labellist = sorted(labellist, key=lambda x: x[0])
    return labellist

def rNNClassifier(k):
    print("Assigning labels using rNN classifier")

    labelsdetaillist = GetLabels()

    labelledmovieslist = sorted(numpy.unique([movie[0] for movie in labelsdetaillist]))
    #print(labelledmovieslist)

    tfidf = GetMoviesTagsData()

    movieslist = sorted(numpy.unique(tfidf['movieid'].tolist()))

    moviedetaillist = GetMoviesDetails(movieslist)

    #print(moviedetaillist)

    movie_tag_matrix = tfidf.pivot_table(index='movieid', columns='tagid', values='tfidf', fill_value=0)

    #print(movie_tag_matrix)

    newlabellist =[]

    inputmovielist = list(set(movieslist) - set(labelledmovieslist))
    #print(inputmovielist)

    #inputmovielist = [8171]
    for movieid in inputmovielist:

        givenmovieindex = movieslist.index(movieid)

        #print(givenmovieindex)

        given_movie_tags = movie_tag_matrix.values[givenmovieindex]
       # print("{0}-{1} \n{2}\n".format(givenmovieindex,movieslist[givenmovieindex],given_movie_tags))

        relatedmovies = []
        topklabels = []

        for i in range (0, len(labelsdetaillist)):

            labeledmovieindex = movieslist.index(labelsdetaillist[i][0])
            labeled_movie_tags = movie_tag_matrix.values[labeledmovieindex]
            #print("{0}-{1}\n {2}\n".format(labeledmovieindex,movieslist[labeledmovieindex],labeled_movie_tags))
            moviemovielatentsimilarity = numpy.matmul(given_movie_tags, labeled_movie_tags.transpose());
            relatedmovies.append((moviedetaillist[labeledmovieindex][0], moviedetaillist[labeledmovieindex][1], moviemovielatentsimilarity, labelsdetaillist[i][1]))

        relatedmovies = sorted(relatedmovies, reverse=1, key = lambda x:x[2])

#        print(relatedmovies)
 #       print("Top {0} neighbours to movie[{1}]\n".format(k, movieid))

        for i in range(0, min(len(relatedmovies),k)):
  #          print("ID: {0}, Name: {1}, Similarity: {2} Label: {3}".format(relatedmovies[i][0],relatedmovies[i][1],relatedmovies[i][2],relatedmovies[i][3]))
            topklabels.append(relatedmovies[i][3])

        toplabel,times= Counter(topklabels).most_common(1)[0]
   #     print("The movie {0} is classified to label [{1}]\n".format(movieid, toplabel))
        newlabellist.append([movieid,toplabel])

    newlabels = pandas.DataFrame(newlabellist,columns=['movieid','label'])

    #print(newlabels)
    print("Classification using r Nearest Neighbor completed. Please check the csv file.")
    newlabels.to_csv('ClassificationUsingrNN.csv', index=False)


# rNNClassifier(8171, 5)


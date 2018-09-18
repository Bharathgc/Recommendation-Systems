
# coding: utf-8

# In[77]:


import pandas
import numpy
import sys
import math
import datetime

genomeTags = pandas.read_csv("genome-tags.csv")
imdbActorInfo = pandas.read_csv("imdb-actor-info.csv")
mlmovies = pandas.read_csv("mlmovies.csv")
mlratings = pandas.read_csv("mlratings.csv")
mltags = pandas.read_csv("mltags.csv")
mlusers = pandas.read_csv("mlusers.csv")
movieActor = pandas.read_csv("movie-actor.csv")

def appendTimestampWeight(row, oldTagTime, recentTagTime):
    return ((pandas.to_datetime(row['timestamp']) - oldTagTime).days + 1) / ((recentTagTime - oldTagTime).days + 1)

def calculateTF(row):
    return row['totalTagWeight'] / row ['TFDenForEachMovie']

def calculateIDF(row, IDFnumValue):
    return math.log10(IDFnumValue / row ['IDFDenValue'])

def calculateTFIDF(row):
    return row['TFValue'] * row['IDFValue']

def appendTFIDFValue(moviesTagUniqueData, idfdata):
    TFIDFData = moviesTagUniqueData.merge(idfdata, on='tagid')
    TFIDFData['TFIDFValue'] = TFIDFData.apply(calculateTFIDF, axis=1)
    return TFIDFData[['movieid','tagid','TFIDFValue']]

def firstASVD(inputGenre):
    movieData = mlmovies[mlmovies['genres'].str.contains(inputGenre)]
    movieIDList = movieData['movieid'].tolist()

    moviesTagData = mltags[mltags['movieid'].isin(movieIDList)]
    moviesTagData.is_copy = False

    oldTagTime = pandas.to_datetime(min(moviesTagData['timestamp']))
    recentTagTime = pandas.to_datetime(max(moviesTagData['timestamp']))
    moviesTagData['weight'] = moviesTagData.apply(appendTimestampWeight, axis=1, args=(oldTagTime, recentTagTime))

    moviesTagData['totalTagWeight'] = moviesTagData.groupby(['movieid','tagid'])['weight'].transform('sum')
    moviesTagUniqueData = moviesTagData[['movieid','tagid','totalTagWeight']].drop_duplicates(subset=['movieid','tagid'])
    moviesTagUniqueData['TFDenForEachMovie'] = moviesTagUniqueData.groupby('movieid')['totalTagWeight'].transform('sum')
    moviesTagUniqueData['TFValue'] = moviesTagUniqueData.apply(calculateTF , axis=1)

    IDFnumValue = mlmovies.shape[0]
    moviesTagUniqueDataList = moviesTagUniqueData['tagid'].tolist()

    moviesTagDetailsData = mltags[mltags['tagid'].isin(moviesTagUniqueDataList)]
    moviesTagDetailsData.is_copy = False
    moviesTagDetailsData.drop_duplicates(subset=['tagid', 'movieid'], inplace=True)
    moviesTagDetailsData['IDFDenValue'] = moviesTagDetailsData.groupby('tagid')['movieid'].transform('count')
    moviesTagDetailsData.drop_duplicates(subset=['tagid'], inplace=True)
    moviesTagDetailsData['IDFValue'] = moviesTagDetailsData.apply(calculateIDF, axis=1, IDFnumValue=IDFnumValue)
    finalTFIDFValue = appendTFIDFValue(moviesTagUniqueData[['movieid' , 'tagid', 'TFValue']], moviesTagDetailsData[['tagid', 'IDFValue']])

    movieTagMatrix = finalTFIDFValue.pivot_table(index='movieid', columns='tagid', values='TFIDFValue', fill_value=0)
    u, s, vT = numpy.linalg.svd(movieTagMatrix, full_matrices=False)

    tagList = finalTFIDFValue['tagid'].tolist()
    tagIDNameData = genomeTags[genomeTags['tagId'].isin(tagList)]
    tagIDNameList = tagIDNameData.values.tolist()
    tagIDNameList.sort(key=lambda x:x[0])

    for i in range(0, min(s.size, 4)):
        print("Latent Semantics {0}:" .format(i+1))
        latentSemanticTagMatrixRow = vT[i]
        meanValue = numpy.mean(latentSemanticTagMatrixRow)
        for j in range(0, len(tagIDNameList)):
            if(latentSemanticTagMatrixRow[j] >= meanValue):
                result = "%s %s %s" % (str(tagIDNameList[j][0]), " " , str(tagIDNameList[j][1]))
                print(result)
        print("")

#firstASVD("Adventure")


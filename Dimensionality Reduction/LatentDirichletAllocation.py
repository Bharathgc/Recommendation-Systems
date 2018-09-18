
# coding: utf-8

# In[34]:


import sys
import math
import datetime
import operator
import pandas
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import warnings
warnings.filterwarnings('ignore')

genomeTags = pandas.read_csv("genome-tags.csv")
imdbActorInfo = pandas.read_csv("imdb-actor-info.csv")
mlmovies = pandas.read_csv("mlmovies.csv")
mlratings = pandas.read_csv("mlratings.csv")
mltags = pandas.read_csv("mltags.csv")
mlusers = pandas.read_csv("mlusers.csv")
movieActor = pandas.read_csv("movie-actor.csv")

def firstaLDA(inputGenre):
    inputGenreMoviesList = mlmovies[mlmovies['genres'].str.contains(inputGenre)]['movieid'].tolist()
    inputGenreMoviesTagsList = mltags[mltags['movieid'].isin(inputGenreMoviesList)]
    spaceSeparatedTagDetails = []
    inputGenreMoviesListWithoutEmptyTags = numpy.unique(inputGenreMoviesTagsList[['movieid','tagid']]['movieid'].tolist())
    totalMovies = len(inputGenreMoviesListWithoutEmptyTags)
    for i in (inputGenreMoviesListWithoutEmptyTags):
        tagList = inputGenreMoviesTagsList[inputGenreMoviesTagsList['movieid'] == i]['tagid'].tolist()
        tempTagList = []
        for singleTag in tagList:
            tempTagList.append(str(singleTag))
        spaceSeparatedTagDetails.append(' '.join(tempTagList))

    vectorizer = CountVectorizer(max_df=0.95, min_df=0.05, max_features=totalMovies)
    TFValue = vectorizer.fit_transform(spaceSeparatedTagDetails)
    featureNames = vectorizer.get_feature_names()
    LDAValue = LatentDirichletAllocation(n_components=4, max_iter=5, learning_method='online', learning_offset=50., random_state=0)
    LDAValue.fit(TFValue)
#     print(genomeTags[genomeTags['tagId'].isin([19])]['tag'].values[0])

    for i, topic in enumerate(LDAValue.components_):
        print("Latent Semantics {0}:" .format(i+1))
        print(" ".join([featureNames[i]
                        for i in topic.argsort()[:-5:-1]]))
        print(" ")

#firstaLDA("Thriller")


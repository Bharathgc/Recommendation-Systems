
# coding: utf-8

# In[127]:

from __future__ import division 
from operator import itemgetter
from collections import OrderedDict
from datetime import datetime
from collections import Counter
import pandas
import numpy as np
import datetime as dt
import math
import itertools



# In[128]:


ml_tags = pandas.read_csv("mltags.csv")
ml_usrs = pandas.read_csv("movie-actor.csv")
ml_movies = pandas.read_csv("mlmovies.csv")
ml_ratings = pandas.read_csv("mlratings.csv")
ml_genome_tags = pandas.read_csv("genome-tags.csv")


# In[129]:


ml_urs_tags = pandas.merge(ml_tags,ml_usrs, on = "movieid")
len_col = ml_urs_tags.shape[0]
tag_freq = Counter(ml_urs_tags['tagid'])
ml_genre_tags = pandas.merge(ml_movies,ml_tags, on ="movieid")
ml_ratings_tags = pandas.merge(ml_ratings,ml_tags, on = "movieid")
uniq_actors = len(ml_urs_tags['actorid'].unique())
uniq_genres = len(ml_genre_tags['genres'].unique())
uniq_users = len(ml_ratings_tags['userid_x'].unique())


# In[138]:


def actor_TF(inp_actid):
    ml_new = ml_urs_tags[ml_urs_tags['actorid'] == inp_actid]
    if(ml_new.empty):
        print("Actor does not have any tags")
        return(ml_new)
    #*******weighted-rank*******
    ml_new_rank =  ml_new.sort_values('actor_movie_rank',ascending=True)
    actor_highest_rank = ml_new_rank.loc[ml_new_rank.index[0], 'actor_movie_rank']
    actor_lowest_rank = ml_new_rank.loc[ml_new_rank.index[-1], 'actor_movie_rank']
    #print(actor_highest_rank)
    #print(actor_lowest_rank)
    denom_rank = (actor_lowest_rank - actor_highest_rank) + 1
    ml_new_rank['weighted_rank'] = ( actor_lowest_rank - ml_new_rank['actor_movie_rank']  + 1 )/denom_rank
    ml_new_rank =  ml_new_rank.sort_values('weighted_rank',ascending=False)
    #print(ml_new_rank)
    ml_new = ml_new_rank
    #print(ml_new_rank)
    #***************************
    
    #***********weighted-time*******
    ml_new_time =  ml_new.sort_values('timestamp',ascending=False)
    actor_newest_time = ml_new_time.loc[ml_new_time.index[0], 'timestamp']
    actor_oldest_time = ml_new_time.loc[ml_new_time.index[-1], 'timestamp']
    actor_newest_time = datetime.strptime(actor_newest_time , '%Y-%m-%d %H:%M:%S')
    actor_oldest_time = datetime.strptime(actor_oldest_time , '%Y-%m-%d %H:%M:%S')
    denom_time = (actor_newest_time - actor_oldest_time).total_seconds() + 1
    #print(actor_oldest_time)
    #print(actor_newest_time)
    ml_new['timestamp'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    ml_new['timestamp'] = pandas.to_datetime(ml_new['timestamp'])
    ml_new['timestamp'] = (ml_new['timestamp'] - actor_oldest_time)
    weighted_time = []
    for coloumn in ml_new['timestamp']:
        coloumn = (coloumn.total_seconds() + 1)/denom_time
        weighted_time.append(coloumn)
    ml_new['weighted_time'] = weighted_time
    #print(ml_new)
     #*************************************************************************#    
    ml_new['avg_weight'] = (ml_new['weighted_time'] + ml_new['weighted_rank'])/2
    #print(ml_new)
    tags = ml_new['tagid']
    uniq_tag_id = ml_new['tagid'].unique()
    freq = Counter(tags)
    #print (freq)
    col_sum = sum(ml_new['avg_weight'])
    #print(col_sum)
    weighted_table = []
    #print(uniq_tag_id)
    for i in uniq_tag_id:
        ml_gen_tags = ml_genome_tags[ml_genome_tags['tagId'] == i]
        ml_gen_tags = list(ml_gen_tags['tag'])
        ml_unique_tag = ml_new[ml_new['tagid'] ==i]
        tag_weight_sum = sum(ml_unique_tag['avg_weight'])
        weighted_table.append([inp_actid,i,freq[i],(tag_weight_sum/col_sum),ml_gen_tags])
        
    columns = ['actorid','tagid','freq','tag_weight','tagname']
    weighted_table = pandas.DataFrame(weighted_table,columns=columns)
    weighted_table = weighted_table.sort_values('tag_weight',ascending=False)
    #print (weighted_table)
    return weighted_table

#*****************************************************************************************************#


#*************************actor_IDF*******************************************************************#
def actor_IDF(inp_tagid):
    filtered_tags = ml_urs_tags[ml_urs_tags['tagid'] == inp_tagid]
    filtered_tags = filtered_tags['actorid'].unique()
    idf = uniq_actors/len(filtered_tags)
    idf = math.log10(idf)
    return idf
#******************************************************************************************************#


#***********************Print_actor_vector*************************************************************#
def print_actor_vector(actorid,modleid):
    tf_modle = actor_TF(actorid)
    if(tf_modle.empty):
        quit()
    elif(modleid == "TF"):
        print(tf_modle[['tagname','tag_weight']])
    elif(modleid == "TFIDF"):
        tf_modle['IDF'] = tf_modle['tagid'].apply(actor_IDF)
        tf_modle['TFIDF'] = tf_modle['tag_weight'] * tf_modle['IDF']
        tf_modle = tf_modle.sort_values('TFIDF',ascending = False)
        print(tf_modle[['tagname','TFIDF']])
        
#********************************************************************************************************#


#**********************GENRE_TF**************************************************************************#
def genre_TF(genre):
    count = 0
    ml_new = pandas.DataFrame(columns=list(ml_genre_tags.columns))
    for i in ml_genre_tags['genres']:
        if(i.find(genre) == -1):
            count = count + 1
            continue
        else:
            ml_new = ml_new.append(ml_genre_tags.iloc[count])
            #print(ml_genre_tags.iloc[count])
            count = count + 1
    #print(ml_new.shape)
    #print(ml_new)
    #print(count)
    #ml_new = ml_genre_tags[ml_genre_tags['genres'] == genre]
    if(ml_new.empty):
        print("Actor does not have any tags")
        return(ml_new)
    ml_new_time =  ml_new.sort_values('timestamp',ascending=False)
    actor_newest_time = ml_new_time.loc[ml_new_time.index[0], 'timestamp']
    actor_oldest_time = ml_new_time.loc[ml_new_time.index[-1], 'timestamp']
    actor_newest_time = datetime.strptime(actor_newest_time , '%Y-%m-%d %H:%M:%S')
    actor_oldest_time = datetime.strptime(actor_oldest_time , '%Y-%m-%d %H:%M:%S')
    denom_time = (actor_newest_time - actor_oldest_time).total_seconds() + 1
    ml_new['timestamp'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    ml_new['timestamp'] = pandas.to_datetime(ml_new['timestamp'])
    ml_new['timestamp'] = (ml_new['timestamp'] - actor_oldest_time)
    weighted_genre = []
    for coloumn in ml_new['timestamp']:
        coloumn = (coloumn.total_seconds() + 1)/denom_time
        weighted_genre.append(coloumn)
    ml_new['weighted_genre'] = weighted_genre
    #print(ml_new)
    tags = ml_new['tagid']
    uniq_tag_id = ml_new['tagid'].unique()
    freq = Counter(tags)
    col_sum = sum(ml_new['weighted_genre'])
    #print(len(uniq_tag_id))
    weighted_table = []
    for i in uniq_tag_id:
        ml_gen_tags = ml_genome_tags[ml_genome_tags['tagId'] == i]
        ml_gen_tags = list(ml_gen_tags['tag'])
        ml_unique_tag = ml_new[ml_new['tagid'] ==i]
        tag_weight_sum = sum(ml_unique_tag['weighted_genre'])
        weighted_table.append([genre,i,freq[i],(tag_weight_sum/col_sum),ml_gen_tags])
    
    columns = ['genre','tagid','freq','Genre_tf','tagname']
    weighted_table = pandas.DataFrame(weighted_table,columns=columns)
    weighted_table = weighted_table.sort_values('Genre_tf',ascending=False)
    #print (weighted_table)
    return weighted_table
    
#********************************************************************************************************#


#*************************genre_IDF*******************************************************************#
def genre_IDF(inp_tagid):
    filtered_genres = ml_genre_tags[ml_genre_tags['tagid'] == inp_tagid]
    filtered_genres = filtered_genres['genres'].unique()
    idf = uniq_genres/len(filtered_genres)
    idf = math.log10(idf)
    return idf
#******************************************************************************************************#


#************************************print_genre_vector_main**************************************************#
def print_genre_vector(genre,modleid):
    tf_modle = genre_TF(genre)
    if(tf_modle.empty):
        quit()
    elif(modleid == "TF"):
        print(tf_modle[['tagname','Genre_tf']])
    elif(modleid == "TFIDF"):
        tf_modle['IDF'] = tf_modle['tagid'].apply(genre_IDF)
        tf_modle['TFIDF'] = tf_modle['Genre_tf'] * tf_modle['IDF']
        tf_modle = tf_modle.sort_values('TFIDF',ascending = False)
        print(tf_modle[['tagname','TFIDF']])
        
#*******************************************************************************************************#


#*****************************************USER_TF*********************************************************#
def user_TF(userid):
    rated_movies = ml_ratings[ml_ratings['userid'] == userid]
    tag_movies = ml_tags[ml_tags['userid'] == userid]
    rated_movies = rated_movies['movieid']
    tag_movies = tag_movies['movieid']
    rated_movies = rated_movies.append(tag_movies)
    all_movies = rated_movies.unique()
    ml_new = pandas.DataFrame(columns=list(ml_tags.columns))
    for i in all_movies:
         ml_new = ml_new.append(ml_tags[ml_tags['movieid'] == i])   
    ml_new_time =  ml_new.sort_values('timestamp',ascending=False)
    actor_newest_time = ml_new_time.loc[ml_new_time.index[0], 'timestamp']
    actor_oldest_time = ml_new_time.loc[ml_new_time.index[-1], 'timestamp']
    actor_newest_time = datetime.strptime(actor_newest_time , '%Y-%m-%d %H:%M:%S')
    actor_oldest_time = datetime.strptime(actor_oldest_time , '%Y-%m-%d %H:%M:%S')
    denom_time = (actor_newest_time - actor_oldest_time).total_seconds() + 1
    ml_new['timestamp'].apply(lambda x: dt.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))
    ml_new['timestamp'] = pandas.to_datetime(ml_new['timestamp'])
    ml_new['timestamp'] = (ml_new['timestamp'] - actor_oldest_time)
    weighted_user = []
    for coloumn in ml_new['timestamp']:
        coloumn = (coloumn.total_seconds() + 1)/denom_time
        weighted_user.append(coloumn)
    ml_new['weighted_user'] = weighted_user
    #print(ml_new)
    tags = ml_new['tagid']
    uniq_tag_id = ml_new['tagid'].unique()
    freq = Counter(tags)
    col_sum = sum(ml_new['weighted_user'])
    #print(len(uniq_tag_id))
    weighted_table = []
    for i in uniq_tag_id:
        ml_gen_tags = ml_genome_tags[ml_genome_tags['tagId'] == i]
        ml_gen_tags = list(ml_gen_tags['tag'])
        ml_unique_tag = ml_new[ml_new['tagid'] ==i]
        tag_weight_sum = sum(ml_unique_tag['weighted_user'])
        weighted_table.append([userid,i,freq[i],(tag_weight_sum/col_sum),ml_gen_tags])
    
    columns = ['userid','tagid','freq','User_tf','tagname']
    weighted_table = pandas.DataFrame(weighted_table,columns=columns)
    weighted_table = weighted_table.sort_values('User_tf',ascending=False)
    #print (weighted_table)
    return weighted_table

#*********************************************************************************************************#


#*********************************USER_IDF*****************************************************************#
def user_IDF(inp_tagid):
    filtered_users = ml_ratings_tags[ml_ratings_tags['tagid'] == inp_tagid]
    filtered_users = filtered_users['userid_x'].unique()
    idf = uniq_users/len(filtered_users)
    idf = math.log10(idf)
    return idf

#**********************************************************************************************************#


#**********************print_user_vectors_main*********************************************************#
def print_user_vector(userid,modleid):
    tf_modle = user_TF(userid)
    if(modleid == "TF"):
        print(tf_modle[['tagname','User_tf']])
    elif(modleid == "TFIDF"):
        tf_modle['IDF'] = tf_modle['tagid'].apply(user_IDF)
        tf_modle['TFIDF'] = tf_modle['User_tf'] * tf_modle['IDF']
        tf_modle = tf_modle.sort_values('TFIDF',ascending = False)
        print(tf_modle[['tagname','TFIDF']])
        
        
#********************************************************************************************************#

#********************************P-DIFF1_modle***********************************************#
def P_DIFF1(genre1,genre2):
    count = 0
    ml_new_genre1 = pandas.DataFrame(columns=list(ml_genre_tags.columns))
    for i in ml_genre_tags['genres']:
        if(i.find(genre1) == -1):
            count = count + 1
            continue
        else:
            ml_new_genre1 = ml_new_genre1.append(ml_genre_tags.iloc[count])
            count = count + 1
    
    ml_new_genre1.drop('moviename',1,inplace = True)        
    genre1_tags = ml_new_genre1['tagid'].unique()
    r1j_values = {}
    for i in genre1_tags:
        m = ml_new_genre1[ml_new_genre1['tagid'] == i]
        n = m['movieid'].unique()
        r1j_values[i] = len(n)
           
    count = 0
    ml_new_genre2 = pandas.DataFrame(columns=list(ml_genre_tags.columns))
    for i in ml_genre_tags['genres']:
        if(i.find(genre2) == -1):
            count = count + 1
            continue
        else:
            ml_new_genre2 = ml_new_genre2.append(ml_genre_tags.iloc[count])
            count = count + 1 
            
    common_1 = {}
    for i in genre1_tags:
        k = ml_new_genre1[ml_new_genre1['tagid'] == i]
        l = k['movieid'].unique()
        m = ml_new_genre2[ml_new_genre2['tagid'] == i]
        n = m['movieid'].unique()
        o = np.intersect1d(l,n)
        common_1[i] = len(o)
        
    m2j_values = {}
    for i in genre1_tags:
        m = ml_new_genre2[ml_new_genre2['tagid'] == i]
        n = m['movieid'].unique()
        m2j_values[i] = len(n)
        
#calculating the R,M values before you append g1 movies to g2            
    R = ml_new_genre1['movieid'].unique()
    S = ml_new_genre2['movieid'].unique()
    common = np.intersect1d(R,S)   
    M = len(R) + len(S) - len(common)

    ml_new_genre2.drop('moviename',1,inplace = True)
    ml_new_genre2 = ml_new_genre2.append(ml_new_genre1)
    m1j_values = {}
    for i in genre1_tags:
        m = ml_new_genre2[ml_new_genre2['tagid'] == i]
        n = m['movieid'].unique()
        m1j_values[i] = len(n)

    diff4  = {}
    diff1 = {}
    diff3 = {}
    diff2 = {}
    diff_final = {}
    wij = 0
    n1 = np.empty(0)
    d1 = np.empty(0)
    f1 = np.empty(0)
    f2 = np.empty(0)

    for i in genre1_tags:
        r = r1j_values[i]
        if(((m1j_values[i] != r) - (R != r)).any()):    
            n1 = r1j_values[i]/(R - r1j_values[i])
            d1 = (m1j_values[i] - r1j_values[i])/(M - m1j_values[i] - R + r1j_values[i])
            f1 = r1j_values[i]/R
            f2 = (m1j_values[i] - r1j_values[i])/(M - R)
        else:
            if(common_1[i] == 0 & m2j_values[i] == 0):
                diff1[i] = "This is most Discriminating Factor"
            elif(m1j_values[i] == 0 & m2j_values[i] == 0):
                diff4[i] =  "This is least discrimating Factor"
            elif(m2j_values[i] == 0):
                diff2[i] = common_1[i]

    for i in range(len(n1)):
        wij = math.log10(n1[i]) * abs(f1[i])
        diff3[i] = wij
    li = []    
    #df1 = pandas.DataFrame(columns= ['tagid','weight'])
    li1 = list(diff1.items())
    li2 = list(diff2.items())
    li3 = list(diff3.items())
    li4 = list(diff4.items())
    li = list(li.extend(li1))
    li = list(li.extend(li2))
    li = list(li.extend(li3))
    li = list(li.extend(li4))
    
    
    weight = pandas.DataFrame( li , columns=['Tagid','Weight'])
    print(weight)
      
#********************************************************************************************************#

#****************************************DIFFERENTIAT_GENRE**********************************************#

def differentiate_genre(genre1,genre2,modleid):
    if(modleid == "TFIDFDIFF"):
        count = 0
        tf_modle_genre1 = genre_TF(genre1) 
        ml_new = pandas.DataFrame(columns=list(ml_genre_tags.columns))
        for i in ml_genre_tags['genres']:
            if(i.find(genre1) == -1 & i.find(genre2) == -1):
                count = count + 1
                continue
            else:
                ml_new = ml_new.append(ml_genre_tags.iloc[count])
                count = count + 1
        unq_gen1_gen2 = len(ml_new['movieid'].unique())
        idf = []
        for i in tf_modle_genre1['tagid']:
            filtered_genres = ml_new[ml_new['tagid'] == i]
            filtered_genres = filtered_genres['movieid'].unique()
            idf_value = unq_gen1_gen2/len(filtered_genres)
            idf.append(math.log10(idf_value))
        tf_modle_genre1['IDF'] = idf
        tf_modle_genre1['TFIDF'] = tf_modle_genre1['Genre_tf'] * tf_modle_genre1['IDF']
        tf_modle_genre1 = tf_modle_genre1.sort_values('TFIDF',ascending = False)
        print(tf_modle_genre1[['tagname','TFIDF']])
        
        
#********************************************************************************************************#


# In[146]:
print(" Choose any 1 Fuction:")
print("Press 1 To print User Vector")
print("Press 2 To print Genre Vector")
print("Press 3 To printt Actor Vector")
print("Press 4 To print Differentiate Genre Vector")

ch = input("Enter an Option")
if(ch == "1"):
	uid = input("Enter a Userid")
	mod = input("Enter TF or TFIDF")
	print_user_vector(int(uid),mod)
elif(ch == "2"):
	gen = input("Enter a Genre")
	mod = input("Enter TF or TFIDF")
	print_genre_vector(gen,mod)
elif(ch == "3"):
	act = input("Enter an actorid")
	mod = input("Enter TF or TFIDF")
	print_actor_vector(int(act),mod)
elif(ch == "4"):
	di = input("Enter Genre1")
	di2 = input("Enter Genre2")
	mod = "TFIDFDIFF"
	differentiate_genre(di,di2,mod)

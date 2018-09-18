import sys
import pandas
import numpy
import math
import datetime

allactormoviesdata =pandas.read_csv("movie-actor.csv")

allactormoviesdata['value'] = 1

actor_movie_matrix = allactormoviesdata.pivot_table(index='actorid', columns='movieid', values='value', fill_value=0)

#print(actor_movie_matrix)

transpose_actor_movie_matrix = actor_movie_matrix.transpose()

coactorcoactorimilarity = numpy.matmul(actor_movie_matrix, transpose_actor_movie_matrix);
a = coactorcoactorimilarity.shape
#print(a)
numpy.fill_diagonal(coactorcoactorimilarity, 0)
#file = open('testfile1.csv','w') 
#for row in coactorcoactorimilarity:
  #  print(row)
   # file.write("%s\n" % row)
    
#file.close()

u, s, v = numpy.linalg.svd(coactorcoactorimilarity, full_matrices=False)

#print(u)
#print(s)
#print(v)

actorlist = sorted(allactormoviesdata['actorid'].tolist())
actorlist = sorted(list(set(allactormoviesdata['actorid'].tolist())))

actordetails = pandas.read_csv("imdb-actor-info.csv")

actordetails = actordetails[actordetails['id'].isin(actorlist)]

actornamelist = actordetails.values.tolist()
actornamelist = sorted(actornamelist, key = lambda x:x[0])

#print(actorlist)
#print(actornamelist)

#print("Total tag list {0}".format(actornamelist))

for i in range(0, min(10,len(s))):
    tagcontribution = v[i]
    mean = numpy.mean(tagcontribution)
    print("Total co-actors greater than mean {0} for latent semantic {1} with core value {2}\n".format(mean, i+1, s[i]))
    for j in range(0, len(actornamelist)):
        if(tagcontribution[j] >= mean):
            #print(actorlist[j])
            print(actornamelist[j])
    if( i == 2):
        break
l = v.shape[1]
#print(l)
resizev = numpy.delete(v, numpy.s_[2:l-1], 0)
#print(resizev)  
firstsemantic = []
secondsemantic = []
thirdsemantic = []
for j in range(0, len(actornamelist)):
    temp = resizev[:,j]
    val = temp.argmax()
    maxval = max(temp)
    #idx = numpy.where(val)
    #idx = row.index(val)
    if ( maxval == 0.0 and resizev[0][j] == 0.0 and resizev[1][j] == 0.0 and resizev[2][j] == 0.0 ):
        continue
    elif ( val == 0 ):
        firstsemantic.append(str(actornamelist[j]))
    elif ( val == 1 ):
        secondsemantic.append(str(actornamelist[j]))
    elif ( val == 2):
        thirdsemantic.append(str(actornamelist[j]))

#print("first" + firstsemantic)
print("\nfirst semantic with non - overlapping tags -----------------------------------------------------------------")
for i in range(0, len(firstsemantic)):
    print(firstsemantic[i])

print("\nSecond semantic with non - overlapping tags-----------------------------------------------------------------")
for i in range(0, len(secondsemantic)):
    print(secondsemantic[i])
print("\nThird semantic with non - overlapping tags------------------------------------------------------------------")
for i in range(0, len(thirdsemantic)):
    print(thirdsemantic[i])
    

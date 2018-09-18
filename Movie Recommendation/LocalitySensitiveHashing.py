import random
from collections import defaultdict
from operator import itemgetter
from task3input import VectorSpace
import numpy
import pandas as pd
from math import *

global uniqueMovies 
class LSHIndex:

    def __init__(self,hash_family,k,L,vector):
        self.hash_family = hash_family
        self.k = k
        self.L = 0
        self.hash_tables = []
        self.resize(L)
        self.vec = vector
        self.totalMovies = 0

    def resize(self,L):
        if L < self.L:
            self.hash_tables = self.hash_tables[:L]
        else:
            hash_funcs = [[self.hash_family.create_hash_func() for h in range(self.k)] for l in range(self.L,L)]		
            self.hash_tables.extend([(g,defaultdict(lambda:[])) for g in hash_funcs])
            

    def hash(self,g,p,inputQueryPoint):
        arr = [h.hash(p,inputQueryPoint) for h in g]

        #print(arr)
        combined =  self.hash_family.combine(arr)
        #print("combined="+str(combined))
        return combined

    def index(self,points,inputQueryPoint):
        """ index the supplied points """
        self.points = points
        for g,table in self.hash_tables:
            for ix,p in enumerate(self.points):
                #print(ix)
                #print(p)
                table[self.hash(g,p,inputQueryPoint)].append(ix)
        # reset stats
		#print "table"
		#print table
        self.tot_touched = 0
        self.num_queries = 0

    def query(self,q,metric,max_results,inputQueryPoint):
        """ find the max_results closest indexed points to q according to the supplied metric """
        candidates = set()
        for g,table in self.hash_tables:
            #print(str(table))
            matches = table.get(self.hash(g,q,inputQueryPoint),[])
            self.totalMovies += len(matches)
            #print(matches)
            candidates.update(matches)
            #print(candidates)
        # update stats
        self.tot_touched += len(candidates)
        self.num_queries += 1
        # rerank candidates
        candidates = [(ix,metric(q,self.points[ix])) for ix in candidates]
        candidates.sort(key=itemgetter(1))
            #print(candidates)
        return candidates

    def get_avg_touched(self):
        """ mean number of candidates inspected per query """
        return self.tot_touched/self.num_queries

class L2HashFamily:

    def __init__(self,w,d):
        self.w = w
        self.d = d

    def create_hash_func(self):
        # each L2Hash is initialised with a different random projection vector and offset
        return L2Hash(self.rand_vec(),self.rand_offset(),self.w)

    def rand_vec(self):
        return [random.gauss(0,1) for i in range(self.d)]

    def rand_offset(self):
        return random.uniform(0,self.w)

    def combine(self,hashes):
        return str(hashes)

class L2Hash:

    def __init__(self,r,b,w):
        self.r = r
        self.b = b
        self.w = w

    def hash(self,vec,inputQueryPoint):
        #print(vec)
        #print(self.r)
        #print(inputQueryPoint)
        sum = numpy.add(int((numpy.dot(vec,self.r)+self.b)/self.w) , int((numpy.dot(vec,inputQueryPoint)+self.b)/self.w))
        return sum
        #return int((numpy.dot(vec,self.r)+self.b)/self.w)
        #return str([int((vec[i]-inputQueryPoint[i])/self.w) for i,r in enumerate(self.r)])

def L2_norm(u,v):
    return sum((ux-vx)**2 for ux,vx in zip(u,v))**0.5

class LSHTester:


    def __init__(self,points,num_neighbours,VectorSpace,inputMovieIds):
        self.points = points
        self.num_neighbours = num_neighbours
        self.vec = VectorSpace
        self.inputMovieIds = inputMovieIds

    def run(self,name,metric,hash_family,k_vals,L_vals,movieId):
        #print(self.points.shape[0])
        outputMovies = " " 
        mlmovies =pd.read_csv("mlmovies.csv")
        genreMovie = mlmovies[mlmovies['movieid'] == (movieId)]['genres'].tolist()
        genre = genreMovie[0]
        #print(genre)
        total_candidates = set()
        candidates1 = set()
        TM = 0
        for inputMovie in self.inputMovieIds:
            inputQueryPoint = self.vec.getQueryPoint(inputMovie)
            #print(inputQueryPoint)
            for k in k_vals:        # concatenating more hash functions increases selectivity
                lsh = LSHIndex(hash_family,k,0,self.vec)
                for L in L_vals:    # using more hash tables increases recall
                    lsh.resize(L)
                    lsh.index(self.points, inputQueryPoint)
                    #print(movieId)
                    queryPoint = self.vec.getQueryPoint(movieId)
                    #print("Query Point")
                    #print(queryPoint)

                    #candidate =[moviepoint for moviepoint,dist in  lsh.query(queryPoint,metric,self.num_neighbours+1,inputQueryPoint)]
                    candidate =lsh.query(queryPoint,metric,self.num_neighbours+1,inputQueryPoint)
                    TM = TM+len(candidate)
                    #print(candidate)
                    #print(inputMovie)
                    #candidates.append(candidate)
                    total_candidates.update(candidate)
                    
                    #candidate_temp = candidate[0]
                    #ans = candidate_temp[:self.num_neighbours+1]
        candidates1 = [(moviepoint,dist) for moviepoint,dist in total_candidates]
        candidates1.sort(key=itemgetter(1))
        #print(candidates1)
        #print(a)
        candidate_temp = [moviepoint for moviepoint,dist in candidates1]        
        #ans = candidate_temp[:self.num_neighbours+1]
        d = []
        count = 0
        #moviesContainInputGenre = 
        for m in candidate_temp:
            #movie = df[df['moviename'] == self.vec.movieIndexVsName[m]]
            movieid = self.vec.indexVsmovieId[m]
            
            #newmoviename = moviename.replace("*","")
            #print(self.vec.movieIndexVsName[m])
            #movie = mlmovies[mlmovies['moviename'].str.contains(newmoviename)]['movieid'].tolist()
            #print(movieid)
            #movieid = movie[0]
            candidateGenreMovie = mlmovies[mlmovies['movieid'] == (movieid)]['genres'].tolist()
            
            candidateGenre = candidateGenreMovie[0]
            #print("aaaaa     "+candidateGenre)
            #print(genre)
            #print(movie)
            if genre in candidateGenre or candidateGenre in genre:
                #print('hi')
                d.append({'MovieId': movieid, 'Name': self.vec.movieIndexVsName[m]})
                count += 1
            if count == self.num_neighbours+1:
                break
            #print(outputMovies)
            #print(m)
            #print(self.vec.movieIndexVsName[m])
            #print(d)
        return d, {'TM':TM, 'UM':len(candidate_temp)}

    def linear(self,q,metric,max_results):
        """ brute force search by linear scan """
        candidates = [(ix,metric(q,p)) for ix,p in enumerate(self.points)]
        return sorted(candidates,key=itemgetter(1))[:max_results]

class relevance_feedback:
    def __init__(self, data, num_neighbours, l_vals, k_vals, radius, points, vec, d, movieId, inputMovieIds):
        self.data = data
        self.num_neighbours = num_neighbours
        self.k_vals = k_vals
        self.l_vals = l_vals
        self.radius = radius
        self.points = points
        self.vec = vec
        self.d = d
        self.inputMovieId = movieId
        self.relevantData = 0
        self.inputMovieIds = inputMovieIds
    def userInput(self, num_neighbours):       
        print("\nEnter feedback for all output movies as R for relevant movies, I for irrelevant movies and blank for no feedback:\n")
        feedback = []
        for d in self.data:
            if self.inputMovieId == d['MovieId']:
                continue
            print(d['Name'])
            value = input()
            feedback.append({'MovieId': d['MovieId'], 'Name': d['Name'],'RF': value})
        return feedback
    
    def computeLSH(self, userFeedback):
        tester = LSHTester(self.points,self.num_neighbours,self.vec,self.inputMovieIds) # the queries are the first 100 points
        feedbackLSH = []
        for movie in userFeedback:
            if(movie['RF'] == 'R' or movie['RF'] == 'r'):
                self.relevantData += 1
                args = {'name':'L2',
                    'metric':L2_norm,
                        'hash_family':L2HashFamily(10*self.radius,self.d),
                    'k_vals':[self.k_vals],
                    'L_vals':[self.l_vals],
                    'movieId': int(movie['MovieId'])}
                data, UM = tester.run(**args)
                feedbackLSH.append({'MovieId': movie['MovieId'],'Name': movie['Name'], 'Data': data})
        #print(feedbackLSH)
        return feedbackLSH
    
    def printRelevantMovies(self, feedbackLSH, userFeedback):
        #noofmovies = round( self.num_neighbours / self.relevantData )
        #print(noofmovies)
        #print(feedbackLSH)
        listOfRelMovies = []
        for feedbackMovie in feedbackLSH:
            temp = feedbackMovie['Data']
            listOfRelMovies.append({'MovieId': feedbackMovie['MovieId'], 'Name': feedbackMovie['Name'],'Data': temp})
        #print("Revised list of movies after Relevance Feedback from User:")
        #print(listOfRelMovies)
        
        #irrelevantMovies = userFeedback[userFeedback['RF'].str.contains('I')]['movieid'].tolist()
        #print(irrelevantMovies)
        print("Revised list of movies after Relevance Feedback from User:")
        for relMovies in listOfRelMovies:
            #temp = relMovies['Data']
            print('\nRelated Movies for the movie id '+ str(relMovies['MovieId']))# +'\t'+ relMovies['Name'])
            print(relMovies['Data'])
        
        
        
        
def main():

    # create a test dataset of vectors of non-negative integers
    d = 500
    xmax = 20
    num_points = 4323
    #points = [[random.randint(0,xmax) for i in xrange(d)] for j in xrange(num_points)]
    vec = VectorSpace()
    points = vec.getLSHInput()

    # seed the dataset with a fixed number of nearest neighbours
    # within a given small "radius"
    print ("enter the number of layers, L ")
    l_vals = int(input())
    #l_vals = 5
    print ("enter the number of hashes per layer, K ")
    k_vals = int(input()) 
    #k_vals = 10
    print ("enter the list of movie ID's ")
    movies = input()
    inputMovieIds = list(map(int, movies.split(" ")))
    #print(movieIds)
    #movieId = 288
    print ("enter the number of output movies required, r ")
    num_neighbours = int(input()) 
    #num_neighbours = 10
    print ("enter a movie ID ")
    movieId = int(input()) 
    radius = 0.2
    
    
    tester = LSHTester(points,num_neighbours,vec,inputMovieIds)

    args = {'name':'L2',
            'metric':L2_norm,
            'hash_family':L2HashFamily(10*radius,d),
            'k_vals':[k_vals],
            'L_vals':[l_vals],
            'movieId': movieId}
    data,uniqueMovies = tester.run(**args)
    
    print("Related movie ids")
    
    for movie in data:
        print(str(movie['MovieId']) + '\t' + movie['Name'])
    
    print("\nTotal no. of movies considered: "+ str(uniqueMovies['TM']))
    
    print("\nTotal no. of unique movies considered: "+ str(uniqueMovies['UM']))
    
    rf = relevance_feedback(data, num_neighbours, l_vals, k_vals, radius, points, vec, d, movieId, inputMovieIds)
    userFeedback = rf.userInput(num_neighbours)
    feedbackLSH = rf.computeLSH(userFeedback)
    rf.printRelevantMovies(feedbackLSH, userFeedback)

main()
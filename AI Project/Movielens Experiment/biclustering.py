# -*- coding: utf-8 -*-
"""
    A program that carries out biclustering (specifically spectral coclustering) on Movielens datatset"""

#make more similar to Python 3
from __future__ import print_function, division, absolute_import, unicode_literals
from decimal import *

#other stuff we need to import
import csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import bipartite
from scipy.sparse import csc_matrix
from sklearn.cluster import KMeans
from sklearn.cluster.bicluster import SpectralCoclustering
from sklearn.metrics.cluster import v_measure_score
from math import *

# Input: M scipy.sparse.csc_matrix
# Output: NetworkX Graph
def nx_graph_from_biadjacency_matrix(M):
    # Give names to the nodes in the two node sets
    U = [ "u{}".format(user_ids[i]) for i in range(M.shape[0]) ]
    V = [ "v{}".format(movie_ids[i]) for i in range(M.shape[1]) ]
    
    # Create the graph and add each set of nodes
    G = nx.Graph()
    G.add_nodes_from(U, bipartite=0)
    G.add_nodes_from(V, bipartite=1)
    
    # Find the non-zero indices in the biadjacency matrix to connect
    # those nodes
    G.add_edges_from([ (U[i], V[j]) for i, j in zip(*M.nonzero()) ])
    
    return G

#beginning of main program

#read in u1.base
training_file = open('ml-100k/u1.base','r')
rows = training_file.readlines()
training_file.close()
training_data=[]
for row in rows:
    list = row.split('\t')
    int_list = [int(item) for item in list]
    training_data.append(int_list)

#read in u1.test
test_file = open('ml-100k/u1.test','r')
rows = test_file.readlines()
test_file.close()
test_data=[]
for row in rows:
    list = row.split('\t')
    int_list = [int(item) for item in list]
    test_data.append(int_list)

print(len(training_data))
print(len(test_data))

user_ids = [row[0] for row in training_data]
user_ids = set(user_ids)
user_ids = sorted(user_ids)
number_of_users = len(user_ids)
#print(user_ids)
print(number_of_users)

movie_ids = [row[1] for row in training_data]
movie_ids = set(movie_ids)
movie_ids = sorted(movie_ids)
number_of_movies = len(movie_ids)
#print(movie_ids)
print(number_of_movies)

#create a user movie matrix
#pre-processing could be in two ways :
# a. either ignore ratings <= 3 so rating of 4 or 5 = 1 in matrix and <=3 is 0
# b. calculate a mean for each user
# c. or simply give 1 if rated and 0 if not rated
user_movie_matrix = np.zeros((number_of_users,number_of_movies))
#user_movie_matrix.fill(0.001)
for row in training_data:
    user_id = user_ids.index(row[0])
    movie_id = movie_ids.index(row[1])
    user_movie_matrix[user_id,movie_id] = row[2]
    #user_movie_matrix[user_id,movie_id] = row[2]

#print(user_movie_matrix[0])
#print(user_movie_matrix[942][1])
#print(user_movie_matrix[942][8])

print(len(user_movie_matrix))
print(len(user_movie_matrix[0]))
#print(user_movie_matrix)
print(type(user_movie_matrix))

#find number of users and movies in each bicluster
'''G = nx_graph_from_biadjacency_matrix(user_movie_matrix)
nx.draw(G)
plt.show()'''

#initialize and carry out clustering
K=10

#km = KMeans(n_clusters = K)
#km.fit(user_movie_matrix)
scc = SpectralCoclustering(n_clusters = K,svd_method='arpack')
scc.fit(user_movie_matrix)

#labels
row_labels = scc.row_labels_
column_labels = scc.column_labels_

bicluster_num_users=np.zeros(K)
bicluster_num_movies=np.zeros(K)
#maintain a list of users per bicluster
bicluster_list_users=[]
#maintain a list of movies per bicluster
bicluster_list_movies=[]

for i in range(K):
    bicluster_list_users.append([])
    bicluster_list_movies.append([])

print(bicluster_list_users)
print(bicluster_list_movies)

for i in range(len(row_labels)):
    bicluster_num_users[row_labels[i]]+=1
    list_of_users = []
    list_of_users = bicluster_list_users[row_labels[i]]
    list_of_users.append(i)
    bicluster_list_users[row_labels[i]]=list_of_users


for i in range(len(column_labels)):
    bicluster_num_movies[column_labels[i]]+=1
    list_of_movies = []
    list_of_movies = bicluster_list_movies[column_labels[i]]
    list_of_movies.append(i)
    bicluster_list_movies[column_labels[i]]=list_of_movies

#print(str(row_labels))
#print(len(row_labels))
print('\n--------Number of users and movies in each bicluster--------')
print('{:<15}\t{}\t{}'.format('Cluster','Users','Movies'))
temp=0
for i in range(K):
    print('{:<15}\t{}\t{}'.format(i,bicluster_num_users[i],bicluster_num_movies[i]))
print(sum(bicluster_num_users))
print(sum(bicluster_num_movies))

f=open('cluster_num_users_biclsutering','w')
for i in range(K):
    f.write(str(i))
    f.write('\t')
    f.write(str(bicluster_num_users[i]))
    f.write('\n')
f.close()

#predicting rating for a movie by a user
print('Number of test data ')
print(len(test_data))
accuracy=0
root_mean_accuracy=0

for row in test_data:
    print('Testing for 1st user and movie in test : ' + str(row))
    movie = row[1]
    rating = row[2]
    print('Bi Cluster for this user : ')
    user = row[0]
    print(user)
    user_id = user_ids.index(user)
    print(user_id)
    #print(labels)
    bicluster_index = row_labels[user_id]
    print(bicluster_index)
    
    print('Other user ids  in this cluster : ')
    print(bicluster_num_users[bicluster_index])
    print(len(bicluster_list_users[bicluster_index]))
    other_user_ids_in_same_cluster=bicluster_list_users[bicluster_index]
    print(other_user_ids_in_same_cluster)
    print('Have they rated movie ')
    print(movie)
    if movie in movie_ids:
        movie_id=movie_ids.index(movie)
    else:
        continue
    
    number_of_users_who_rated_movie=0
    sum_total_rating=0
    for i in other_user_ids_in_same_cluster:
        if user_movie_matrix[i][movie_id] > 0:
            print(i)
            print('index has rated movie ')
            print(movie_id)
            print(user_movie_matrix[i][movie_id])
            number_of_users_who_rated_movie+=1
            sum_total_rating+=user_movie_matrix[i][movie_id]
    print('Predicted Rating for this movie :')
    print(sum_total_rating)
    if(number_of_users_who_rated_movie > 0):
        rating_predicted = sum_total_rating/number_of_users_who_rated_movie
        print(rating_predicted)
        #rating_predicted = round(rating_predicted)
        root_mean_accuracy += Decimal(pow(Decimal(rating_predicted-rating),2))
        
        print(rating)
        if abs(Decimal(rating_predicted - rating)) <= Decimal(1.5):
            print("HERE")
            accuracy += 1

print(accuracy)
print('% accuracy')
print(accuracy*100/len(test_data))

root_mean_accuracy = root_mean_accuracy/len(test_data)
root_mean_accuracy = sqrt(root_mean_accuracy)
print(root_mean_accuracy)

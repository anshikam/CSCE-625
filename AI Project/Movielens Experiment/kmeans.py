# -*- coding: utf-8 -*-
"""
    A program that carries out mini batch k-means clustering on Movielens datatset"""

from __future__ import print_function, division, absolute_import, unicode_literals
from decimal import *

#other stuff we need to import
import csv
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.cluster import v_measure_score
from math import *

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

#Normalizing user-movie matrix
#Additional step
'''for i in range(number_of_users):
   tempList = []
   tempList = user_movie_matrix[i].tolist()
   print('templist')
   print(tempList)
   minVal = min(tempList)
   maxVal = max(tempList)
   for j in tempList:
        j=Decimal(Decimal(j-minVal)/Decimal(maxVal-minVal))
        j=j*5
   user_movie_matrix[i] = tempList'''

print(user_movie_matrix)

print(len(user_movie_matrix))
print(len(user_movie_matrix[0]))
#print(user_movie_matrix)
#initialize and carry out clustering
K=10

#km = KMeans(n_clusters = K)
#km.fit(user_movie_matrix)
#km = KMeans(n_clusters = K)
km = MiniBatchKMeans(n_clusters = K)
km.fit(user_movie_matrix)

#labels
labels = km.labels_
print(str(labels))

#find which cluster each user is in
cluster_num_users=np.zeros(K)
#maintain a list of users per cluster
cluster_list_users=[]
for i in range(K):
    cluster_list_users.append([])

print(cluster_list_users)
prediction = km.predict(user_movie_matrix)
print('\n--------Which cluster each user is in--------')
print('{:<15}\t{}'.format('User','Cluster'))
for i in range(len(prediction)):
    print('{:<15}\t{}'.format(user_ids[i],prediction[i]))
    cluster_num_users[prediction[i]]+=1
    list_of_users = []
    list_of_users = cluster_list_users[prediction[i]]
    list_of_users.append(i)
    cluster_list_users[prediction[i]]=list_of_users

f=open('cluster_num_users','w')
for i in range(K):
    f.write(str(i))
    f.write('\t')
    f.write(str(cluster_num_users[i]))
    f.write('\n')
f.close()
print(cluster_num_users)
print(cluster_list_users)
#Number of users in each cluster
print('\n--------Number of users in a cluster--------')
for i in range(K):
    print('{:<15}\t{}'.format(i,cluster_num_users[i]))
print(sum(cluster_num_users))

print('The total distance of the solution found is',sum((km.transform(user_movie_matrix)).min(axis=1)))

#predicting rating for a movie by a user
print('Number of test data ')
print(len(test_data))
accuracy=0
root_mean_accuracy=0

for row in test_data:
    print('Testing for 1st user and movie in test : ' + str(row))
    movie = row[1]
    rating = row[2]
    print('Cluster for this user : ')
    user = row[0]
    print(user)
    user_id = user_ids.index(user)
    print(user_id)
    #print(labels)
    cluster_index = labels[user_id]
    print(cluster_index)

    print('Other user ids  in this cluster : ')
    print(cluster_num_users[cluster_index])
    print(len(cluster_list_users[cluster_index]))
    other_user_ids_in_same_cluster=cluster_list_users[cluster_index]
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
        print(rating)
        #rating_predicted = round(rating_predicted)
        root_mean_accuracy += Decimal(pow(Decimal(rating_predicted-rating),2))
        
        if abs(Decimal(rating_predicted - rating)) <= Decimal(1.5):
            print("HERE")
            accuracy += 1
        '''elif Decimal(rating - rating_predicted) < Decimal(0.5):
            print("HERE")
            accuracy += 1'''

print(accuracy)
print('% accuracy')
print(accuracy*100/len(test_data))

root_mean_accuracy = root_mean_accuracy/len(test_data)
root_mean_accuracy = sqrt(root_mean_accuracy)
print(root_mean_accuracy)




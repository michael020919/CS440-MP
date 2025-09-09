# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Last Modified 8/23/2023


"""
This is the main code for this MP.
You only need (and should) modify code within this file.
Original staff versions of all other files will be used by the autograder
so be careful to not modify anything else.
"""


import reader
import math
from tqdm import tqdm
from collections import Counter
from collections import defaultdict


'''
util for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=False, lowercase=False, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with naive bayes.
    You can modify the default values for the Laplace smoothing parameter and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def naive_bayes(train_set, train_labels, dev_set, laplace=1.0, pos_prior=0.5, silently=False):
    print_values(laplace,pos_prior)
    #Building up the map
    pos_count = defaultdict(int)
    neg_count = defaultdict(int)
    for i in range(len(train_set)):
        if train_labels[i] == 1:
            for w in train_set[i]:
                pos_count[w] += 1
        else:
            for w in train_set[i]:
                neg_count[w] += 1
    n_pos = 0
    n_neg = 0
    for w in pos_count:
        n_pos += pos_count[w]
    
    total_neg = 0
    for w in neg_count:
        n_neg += neg_count[w]
    
    #Train the train set
    pos_prob = {}
    neg_prob = {}
    vocab = set(list(pos_count.keys())+list(neg_count.keys()))
    V = len(vocab)
    for w in vocab:
        pos_prob[w] = math.log((pos_count[w] + laplace)/(n_pos + laplace * (V + 1)))
        neg_prob[w] = math.log((neg_count[w] + laplace)/(n_neg + laplace * (V + 1)))

    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        yhats.append(-1)

    i = 0
    for rev in dev_set:
        pos_pos = math.log(pos_prior)
        neg_pos = math.log(1.0-pos_prior)
        for w in rev:
            if w in pos_prob:
                pos_pos += pos_prob[w]
            else:
                pos_pos += math.log(laplace/(n_pos + laplace * (V + 1)))
        
        for w in rev:
            if w in neg_prob:
                neg_pos += neg_prob[w]
            else:
                neg_pos += math.log(laplace/(n_neg + laplace * (V + 1)))

        if pos_pos > neg_pos:
            yhats[i] = 1
        else:
            yhats[i] = 0
        i +=1

    return yhats
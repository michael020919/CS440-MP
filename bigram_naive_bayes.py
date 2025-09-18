# bigram_naive_bayes.py
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
utils for printing values
'''
def print_values(laplace, pos_prior):
    print(f"Unigram Laplace: {laplace}")
    print(f"Positive prior: {pos_prior}")

def print_values_bigram(unigram_laplace, bigram_laplace, bigram_lambda, pos_prior):
    print(f"Unigram Laplace: {unigram_laplace}")
    print(f"Bigram Laplace: {bigram_laplace}")
    print(f"Bigram Lambda: {bigram_lambda}")
    print(f"Positive prior: {pos_prior}")

"""
load_data loads the input data by calling the provided utility.
You can adjust default values for stemming and lowercase, when we haven't passed in specific values,
to potentially improve performance.
"""
def load_data(trainingdir, testdir, stemming=True, lowercase=True, silently=False):
    print(f"Stemming: {stemming}")
    print(f"Lowercase: {lowercase}")
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(trainingdir,testdir,stemming,lowercase,silently)
    return train_set, train_labels, dev_set, dev_labels


"""
Main function for training and predicting with the bigram mixture model.
    You can modify the default values for the Laplace smoothing parameters, model-mixture lambda parameter, and the prior for the positive label.
    Notice that we may pass in specific values for these parameters during our testing.
"""
def bigram_bayes(train_set, train_labels, dev_set, unigram_laplace=0.78, bigram_laplace=1.8, bigram_lambda=0.26, pos_prior=0.42, silently=False):
    print_values_bigram(unigram_laplace,bigram_laplace,bigram_lambda,pos_prior)
    ##The code in MP1
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
        pos_prob[w] = math.log((pos_count[w] + unigram_laplace)/(n_pos + unigram_laplace * (V + 1)))
        neg_prob[w] = math.log((neg_count[w] + unigram_laplace)/(n_neg + unigram_laplace * (V + 1)))

    bi_pos_count = defaultdict(int)
    bi_neg_count = defaultdict(int)
    for i in range(len(train_set)):
        tokens = train_set[i]
        bigrams = []
        for j in range(len(tokens) - 1):
            bigrams.append((tokens[j], tokens[j+1]))
        if train_labels[i] == 1:
            for w in bigrams: 
                bi_pos_count[w] += 1
        else:
            for w in bigrams: 
                bi_neg_count[w] += 1

    bi_n_pos = 0
    bi_n_neg = 0
    for w in bi_pos_count:
        bi_n_pos += bi_pos_count[w]
    
    total_neg = 0
    for w in bi_neg_count:
        bi_n_neg += bi_neg_count[w]

    bi_vocab = set(list(bi_pos_count.keys())+list(bi_neg_count.keys()))
    bi_V = len(bi_vocab)

    bi_pos_prob = {}
    bi_neg_prob = {}
    for w in bi_vocab:
        bi_pos_prob[w] = math.log((bi_pos_count[w] + bigram_laplace)/(bi_n_pos + bigram_laplace * (bi_V + 1)))
        bi_neg_prob[w] = math.log((bi_neg_count[w] + bigram_laplace)/(bi_n_neg + bigram_laplace * (bi_V + 1)))
    
    yhats = []
    for doc in tqdm(dev_set, disable=silently):
        yhats.append(-1)

    i = 0
    log_pos_prior = math.log(max(1e-12, pos_prior))
    log_neg_prior = math.log(max(1e-12, 1.0 - pos_prior))
    for rev in dev_set:
        uni_pos = 0.0
        uni_neg = 0.0
        bi_pos = 0.0
        bi_neg = 0.0
        for w in rev:
            if w in pos_prob:
                uni_pos += pos_prob[w]
            else:
                uni_pos += math.log(unigram_laplace/(n_pos + unigram_laplace * (V + 1)))

        for w in rev:
            if w in neg_prob:
                uni_neg += neg_prob[w]
            else:
                uni_neg += math.log(unigram_laplace/(n_neg + unigram_laplace * (V + 1)))

        bis = []
        for j in range(len(rev) - 1):
            bis.append((rev[j], rev[j+1]))
        for b in bis:
            if b in bi_pos_prob:
                bi_pos += bi_pos_prob[b]
            else:
                bi_pos += math.log(bigram_laplace/(bi_n_pos + bigram_laplace * (bi_V + 1)))

        for b in bis:
            if b in bi_neg_prob:
                bi_neg += bi_neg_prob[b]
            else:
                bi_neg += math.log(bigram_laplace/(bi_n_neg + bigram_laplace * (bi_V + 1)))

        score_pos = log_pos_prior + (1 - bigram_lambda)*uni_pos + bigram_lambda*bi_pos
        score_neg = log_neg_prior + (1 - bigram_lambda)*uni_neg + bigram_lambda*bi_neg

        if score_pos > score_neg:
            yhats[i] = 1
        else:
            yhats[i] = 0
        i += 1
    return yhats




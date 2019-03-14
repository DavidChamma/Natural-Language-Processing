from __future__ import division

import argparse
import pandas as pd
import tarfile


import numpy as np
import random
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import pickle
from math import e

__authors__ = ['david_chamma','romain_marsal','jean_roure', 'susan_saal']
__emails__  = ['b00707281@essec.edu','romain.marsal@student.ecp.fr','b00742787@essec.edu', 'b00708560@essec.edu']

PATH_TO_DATA = "C:/Users/33667/Documents/ESSEC/Natural Language Processing/Assignment 1/data/1-billion-word-language-modeling-benchmark-r13output.tar.gz"

#We were working with billion corpus as tar.gz so we used this function all along  
def text2sentences2(path, file_max):
    # feel free to make a better tokenization/pre-processing
    i = 0
    with tarfile.open(path, 'r') as t:
        for filename in t.getnames():
            if i < file_max:
                i = i + 1
                f = t.extractfile(filename)
                if f is not None:
                    data = f.read()
                    data = data.decode()
                    sentences = nltk.sent_tokenize(data)
    return(sentences)
    
def text2sentences(path):
    # feel free to make a better tokenization/pre-processing
    sentences = []
    
    with open(path) as f:
        for l in f:
            #sentences.append( l.lower().split("\n") )
            sentences.append(nltk.sent_tokenize(l)[0])
            
    return sentences


def loadPairs(path):
    data = pd.read_csv(path,delimiter='\t')
    pairs = zip(data['word1'],data['word2'],data['similarity'])
    return pairs


class SkipGram:
    def __init__(self, sentences, nEmbed=100, negativeRate=5, winSize = 5, minCount = 5):
        self.sentences = sentences
        self.nEmbed = nEmbed #dimension of the embedding we consider
        self.negativeRate = negativeRate # k-negative sampling: How many false examples for 1 true example
        self.winSize = winSize # Window size, full length so if 5 we have -2:+2
        self.minCount = minCount #Ignore words appearing less than 5 times
        self.stepsize = None
        self.coeff = None
         

    
    def create_vocab(self, minCount): #To add : how to remove words with occurence less than MinCount
        self.minCount = minCount
        vocabulary = {}
        translator=str.maketrans('','',string.punctuation) #filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        stpwds = stopwords.words("english")
        stpwds.append('would')
        
        for sent in self.sentences:
            sent = sent.translate(translator) #Remove punctuations
            filtered_words = [word for word in word_tokenize(sent.lower()) if word not in stpwds and word.isalpha()] #Remove stopwords + lower case
    
            for word in filtered_words:
                if word in vocabulary:
                    vocabulary[word] = vocabulary[word] + 1
                    
                else:
                    vocabulary[word] = 1
                    
        vocabulary_to_keep = {k: v for k, v in vocabulary.items() if v >= self.minCount + 1}
        vocabulary_to_remove = {k: v for k, v in vocabulary.items() if v < self.minCount + 1}
        return(vocabulary_to_keep, vocabulary_to_remove)
            
    
    def train(self, stepsize = 1e-1, epochs = 5):
        
        self.stepsize = stepsize
        
        def sigmoid(c, w):
            
            dot_prod = np.dot(c, w)
            return 1/(1 + e**-(dot_prod))
                
       
        def objective_function(matrix, negativeRate): #To check
            
            f = - np.log(sigmoid(matrix[1, :], matrix[0, :]))
            for k in range(2, negativeRate + 2):
                f = f - np.log(sigmoid(-matrix[k, :], matrix[0, :]))
            return f
            
        
        def target_gradient(matrix, negativeRate):
            
            f = -matrix[1, :] * sigmoid(-matrix[1, :], matrix[0, :]) 
            return f
        
        
        def positive_gradient(matrix, negativeRate):
            
            f = -matrix[0, :] * sigmoid(-matrix[1, :], matrix[0, :])  
            return f
        
        
        def negative_k_gradient(matrix, negativeRate, k):
            
            f = matrix[0, :] * sigmoid(matrix[k, :], matrix[0, :])  
            return f 
            
        ##Training function            
        vocabulary = self.create_vocab(self.minCount)[0]
        to_remove = list(self.create_vocab(self.minCount)[1].keys())
        
        tableau = list(vocabulary.keys())
        
        self.coeff = {k: np.random.uniform(low=-0.5/self.nEmbed, high=0.5/self.nEmbed, size = self.nEmbed) for k, v in vocabulary.items()} #Initialization of coefficients
        window = self.winSize//2
        lr = self.stepsize
        data = []
        
        translator=str.maketrans('','',string.punctuation) #filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
        stpwds = stopwords.words("english")
        stpwds.append('would')

        for epoch in range(epochs):
            for sent in self.sentences:
                sent = sent.translate(translator) #Remove punctuations
                filtered_words = [word for word in word_tokenize(sent.lower()) if word not in stpwds and word.isalpha() and word not in to_remove] #Remove stopwords
                word_index = 0
    
                for word in filtered_words:
        
                    for context in range(1, window + 1):  
                        # Matrix containing the embedding in 0:Target, 1:context, 2:6 the five negative examples
                        matrix = np.zeros((self.negativeRate + 2, self.nEmbed))
                        # Try is in case there is no neighbor for the target word on the right (overpass the size of the list)
                        try:
                            pos_word = filtered_words[word_index + context]
                            positive = ((word, pos_word), 1) #Positive examples
                            data.append(positive)
                            matrix[0, :] = self.coeff[word]
                            matrix[1, :] = self.coeff[pos_word]
                
                            neg_word = []
                            for k in range(self.negativeRate):
                                neg_word.append(tableau[random.randint(0, len(tableau)-1)])
                                neg = ((word, neg_word[k]), 0) #negative examples
                                data.append(neg)
                                matrix[k + 2, :] = self.coeff[neg_word[k]]
                    
                            #Updating the embedding of the words with SGD 
                            self.coeff[word] = self.coeff[word] - lr*target_gradient(matrix, self.negativeRate)
                            self.coeff[pos_word] = self.coeff[pos_word] - lr*positive_gradient(matrix, self.negativeRate)
                            for k in range(self.negativeRate):
                                self.coeff[neg_word[k]] = self.coeff[neg_word[k]] - lr*negative_k_gradient(matrix, self.negativeRate, k + 2)
                    
                    
                        except IndexError:
                            pass
           
                        # Matrix containing the embedding in 0:Target, 1:context, 2:6 the five negative examples
                        matrix = np.zeros((self.negativeRate + 2, self.nEmbed))
                        # Prevent to read the list in the other way
                        if max(word_index - context, 0) == word_index - context:
                            pos_word = filtered_words[word_index - context]
                            positive = ((word, pos_word), 1)
                            data.append(positive)
                            matrix[0, :] = self.coeff[word]
                            matrix[1, :] = self.coeff[pos_word]
                
                            neg_word = []
                            for k in range(self.negativeRate):
                                neg_word.append(tableau[random.randint(0, len(tableau)-1)])
                                neg = ((word, neg_word[k]), 0)
                                data.append(neg)
                                matrix[k + 2, :] = self.coeff[neg_word[k]]
                
                            self.coeff[word] = self.coeff[word] - lr*target_gradient(matrix, self.negativeRate)
                            self.coeff[pos_word] = self.coeff[pos_word] - lr*positive_gradient(matrix, self.negativeRate)
                            for k in range(self.negativeRate):
                                self.coeff[neg_word[k]] = self.coeff[neg_word[k]] - lr*negative_k_gradient(matrix, self.negativeRate, k + 2)
                    
            
                        else:
                            pass 
            
                    word_index = word_index + 1 #We move to the next word in the sentence
        
    

    def save(self,path):
        with open(path + '\\W', 'wb') as fichier:
            mypickle = pickle.Pickler(fichier)
            mypickle.dump(self.coeff)

    def similarity(self,word1,word2):
        """
            computes similiarity between the two words. unknown words are mapped to one common vector
        :param word1:
        :param word2:
        :return: a float \in [0,1] indicating the similarity (the higher the more similar)
        """
        model = self.coeff
        words_embedded = list(model.keys())
        default_emb = np.ones(100) - 0.5 #default embedding for words not in vocab, another may be set
        
        if word1 in words_embedded:
            w1_emb = model[word1]
        else:
            w1_emb = default_emb
        
        if word2 in model:
            w2_emb = model[word2]
        else:
            w2_emb = default_emb
        
        cosine = np.dot(w1_emb, w2_emb) / (np.linalg.norm(w1_emb) * (np.linalg.norm(w2_emb)))
        return(cosine)
        
        
    @staticmethod
    def load(path):
        with open(path + '\\W', 'rb') as fichier:
            my_depickler = pickle.Unpickler(fichier)
            coeff = my_depickler.load()
        return coeff
        



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--text', help='path containing training data', required=True)
    parser.add_argument('--model', help='path to store/read model (when training/testing)', required=True)
    parser.add_argument('--test', help='enters test mode', action='store_true')

    opts = parser.parse_args()

    if not opts.test:
        sentences = text2sentences(opts.text)
        sg = SkipGram(sentences)
        sg.train(epochs = 10)
        sg.save(opts.model)

    else:
        pairs = loadPairs(opts.text)

        sg = SkipGram.load(opts.model)
        for a,b,_ in pairs:
            print(sg.similarity(a,b))
            


    
# naive_bayes_mixture.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018
# Modified by Jaewook Yeom 02/02/2020

"""
This is the main entry point for Part 2 of this MP. You should only modify code
within this file for Part 2 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""

import math


def naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda,unigram_smoothing_parameter, bigram_smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    bigram_lambda - float between 0 and 1

    unigram_smoothing_parameter - Laplace smoothing parameter for unigram model (between 0 and 1)

    bigram_smoothing_parameter - Laplace smoothing parameter for bigram model (between 0 and 1)

    pos_prior - positive prior probability (between 0 and 1)
    """

    # TODO: Write your code here
    # return predicted labels of development set
    #bigram_probs should be dictionary w tuple key mapping to [P(tuple|ham), P(tuple|spam)]

    #Implement Naive Bayes over a bigram
    bigram_occurences = {}

    #Keep track of the number of bigrams and whether theyre ham or spam
    num_ham_bigram = 0
    num_spam_bigram = 0
    unique_bigrams = 0

    for index in range(len(train_set)):
        for word_index in range(len(train_set[index]) - 1):
            #Test if a new slot needs to be made for the word
            this_bigram = tuple([train_set[index][word_index], train_set[index][word_index + 1]])
            if this_bigram not in bigram_occurences:
                #test if it's ham
                unique_bigrams += 1
                if train_labels[index] == 1:
                    bigram_occurences[this_bigram] = [1,0]
                    num_ham_bigram += 1
                else:
                    bigram_occurences[this_bigram] = [0,1]
                    num_spam_bigram += 1
            else:
                current = bigram_occurences[this_bigram]
                if train_labels[index] == 1:
                    current[0] += 1
                    num_ham_bigram += 1
                else:
                    current[1] += 1
                    num_spam_bigram += 1
                bigram_occurences[this_bigram] = current

            

    ham_bigram_probabilities = {}
    spam_bigram_probabilities = {}

    for bigram in bigram_occurences:
        #Implementing the given algorithtm although I'm a bit confused on X
        #Maybe count the number of times we add a new word to the dict... Not sure though
        #Including +1 for the unrtecognized case

        #instead of num_ham -> get length of ham dictionary
        ham_bigram_probabilities[bigram] = (bigram_occurences[bigram][0] + bigram_smoothing_parameter) / (num_ham_bigram + bigram_smoothing_parameter * (unique_bigrams + 1))
        spam_bigram_probabilities[bigram] = (bigram_occurences[bigram][1] + bigram_smoothing_parameter) / (num_spam_bigram + bigram_smoothing_parameter * (unique_bigrams + 1))
    
    #Calculate probabilities of unrecognized words
    DNE_in_ham_bigram = bigram_smoothing_parameter / (num_ham_bigram + bigram_smoothing_parameter * (unique_bigrams + 1))
    DNE_in_spam_bigram = bigram_smoothing_parameter / (num_spam_bigram + bigram_smoothing_parameter * (unique_bigrams + 1))


#############################################################

    #Setup probabilities for the word classifier

    word_occurences = {}
    #keep track of # of words that are ham and spam
    #To use in given ham what is prob of word
    num_ham = 0
    num_spam = 0
    unique_words = 0

    count_types = {}

    for index in range(len(train_set)):
        for word in train_set[index]:
            #Test if a new slot needs to be made for the word
            if word not in word_occurences:
                #test if it's ham
                unique_words += 1
                if train_labels[index] == 1:
                    word_occurences[word] = [1,0]
                    num_ham += 1
                else:
                    word_occurences[word] = [0,1]
                    num_spam += 1
            else:
                current = word_occurences[word]
                if train_labels[index] == 1:
                    current[0] += 1
                    num_ham += 1
                else:
                    current[1] += 1
                    num_spam += 1
                word_occurences[word] = current

    ham_probabilities = {}
    spam_probabilities = {}

    #Computing P(woird | class) with Laplace smoothing
    for word in word_occurences:
        #Including +1 for the unrtecognized case
        ham_probabilities[word] = (word_occurences[word][0] + unigram_smoothing_parameter) / (num_ham + unigram_smoothing_parameter * (unique_words + 1))
        spam_probabilities[word] = (word_occurences[word][1] + unigram_smoothing_parameter) / (num_spam + unigram_smoothing_parameter * (unique_words + 1))
    
    #Calculate probabilities of unrecognized words
    DNE_in_ham = unigram_smoothing_parameter / (num_ham + unigram_smoothing_parameter * (unique_words + 1))
    DNE_in_spam = unigram_smoothing_parameter / (num_spam + unigram_smoothing_parameter * (unique_words + 1))




    dev_set_labels = []
    #Iterating through dev-set emails
    for email in dev_set:
        #Setup pos_prior
        probability_ham_bigram = math.log(pos_prior)
        probability_spam_bigram = math.log(1-pos_prior) 

        #Evaluate for bigram
        for word_index in range(len(email) - 1):
            this_bigram = tuple([email[word_index],email[word_index + 1]])

            if this_bigram in ham_bigram_probabilities:
                probability_ham_bigram += math.log(ham_bigram_probabilities[this_bigram])
            else:
                probability_ham_bigram += math.log(DNE_in_ham_bigram)
            if this_bigram in spam_bigram_probabilities:
                probability_spam_bigram += math.log(spam_bigram_probabilities[this_bigram])
            else:
                probability_spam_bigram += math.log(DNE_in_spam_bigram)

        #Evaluate for word
        probability_ham = math.log(pos_prior)
        probability_spam = math.log(1-pos_prior)

        for word in email:
            if word in ham_probabilities:
                probability_ham += math.log(ham_probabilities[word])
            else:
                probability_ham += math.log(DNE_in_ham)
            if word in spam_probabilities:
                probability_spam += math.log(spam_probabilities[word])
            else:
                probability_spam += math.log(DNE_in_spam)
        
        final_probability_ham = (bigram_lambda) * probability_ham_bigram + (1-bigram_lambda) * probability_ham
        final_probability_spam = (bigram_lambda) * probability_spam_bigram + (1-bigram_lambda)  * probability_spam

        if final_probability_ham > final_probability_spam:
            dev_set_labels.append(1)
        else:
            dev_set_labels.append(0)
        


    return dev_set_labels
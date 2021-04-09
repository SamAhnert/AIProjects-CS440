# naive_bayes.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 09/28/2018

"""
This is the main entry point for Part 1 of this MP. You should only modify code
within this file for Part 1 -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import math


def naiveBayes(train_set, train_labels, dev_set, smoothing_parameter, pos_prior):
    """
    train_set - List of list of words corresponding with each email
    example: suppose I had two emails 'like this movie' and 'i fall asleep' in my training set
    Then train_set := [['like','this','movie'], ['i','fall','asleep']]

    train_labels - List of labels corresponding with train_set
    example: Suppose I had two emails, first one was ham and second one was spam.
    Then train_labels := [1, 0]

    dev_set - List of list of words corresponding with each email that we are testing on
              It follows the same format as train_set

    smoothing_parameter - The smoothing parameter --laplace (1.0 by default)
    pos_prior - positive prior probability (between 0 and 1)
    """
    # TODO: Write your code here
    # return predicted labels of development set

    #Create a dictionary that maps a word to it's number of occurences in a list (num_ham, num_spam)
    word_occurences = {}
    #keep track of # of words that are ham and spam
    #To use in given ham what is prob of word
    num_ham = 0
    num_spam = 0
    unique_words = 0

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

    #Maps word to their probability of appearing in ham emails and spam emails
    ham_probabilities = {}
    spam_probabilities = {}

    #Computing P(woird | class) with Laplace smoothing
    for word in word_occurences:
        #Implementing the given algorithtm although I'm a bit confused on X
        #Maybe count the number of times we add a new word to the dict... Not sure though
        #Including +1 for the unrtecognized case
        ham_probabilities[word] = (word_occurences[word][0] + smoothing_parameter) / (num_ham + smoothing_parameter * (unique_words + 1))
        spam_probabilities[word] = (word_occurences[word][1] + smoothing_parameter) / (num_spam + smoothing_parameter * (unique_words + 1))
    
    #Calculate probabilities of unrecognized words
    DNE_in_ham = smoothing_parameter / (num_ham + smoothing_parameter * (unique_words + 1))
    DNE_in_spam = smoothing_parameter / (num_spam + smoothing_parameter * (unique_words + 1))

    dev_set_labels = []
    #Iterating through dev-set emails
    for email in dev_set:
        #Setup pos_prior
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

        if probability_ham > probability_spam:
            dev_set_labels.append(1)
        else:
            dev_set_labels.append(0)

    print(unique_words)

    return dev_set_labels
    
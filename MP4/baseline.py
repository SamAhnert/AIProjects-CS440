# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""

def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    #key is a word, value is a dictionary where the key is a tag and the value is the number of times that tag has been on the corresponding word
    words = {}
    #key is a particular tag, value is the total number of instances of a tag in the train set
    tags = {}


    for sentence in train:
        for w in sentence:
            word = w[0]
            tag = w[1]
            #check that it is a valid trainable input
            if word == 'START' or word == 'END':
                continue
            #add to tags total
            if tag not in tags:
                tags[tag] = 1
            else:
                tags[tag] = tags[tag] + 1
            #check if the word has appeared before
            if (word in words):
                # check if corresponding tag has appeared with the word before
                if tag in words[word]:
                    words[word][tag] = words[word][tag] + 1
                else:
                    words[word][tag] = 1
            else:
                words[word] = {tag : 1}
    
    mostLikelyTag = max(tags, key=tags.get)

    taggedSentences = []

    for sentence in test:
        thisSentence = []
        for word in sentence:
            #include an indicator that a sentence is starting or ending
            if word == 'START' or word == 'END':
                thisSentence.append((word,word))
                continue
            #take the most common tag for the word if it's was seen in the training data
            #otherwise just assign the most common overall tag
            if word in words:
                bestTagForWord = max(words[word], key=words[word].get)
            else:
                bestTagForWord = mostLikelyTag
            #append tagged word to the sentence
            thisSentence.append( (word, bestTagForWord) )
        taggedSentences.append(thisSentence)

    return taggedSentences

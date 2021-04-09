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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 2: This is the simplest version of viterbi that doesn't do anything special for unseen words
but it should do better than the baseline at words with multiple tags (because now you're using context
to predict the tag).
"""
import math

def viterbi_1(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    # emmissions are tag to word, transition are tag to tag
    # emmision probability- {tag : {word : prob that the tag produces the observed word}}
    # transition probability- {current_tag: {prev_tag : probab that prev_tag transitions to current_tag}}
    emissionProbabilities, transitionProbabilities, tagOccurenceProbability = train_probabilities(train)

    #implement viterbi algorithm
    taggedSentences = []

    # sentence = test[3]
    # # print(sentence)
    for sentence in test:
        tagProbabilities = {}

        # {index : {tag : {PROB: {num}, PREVTAG: tag}}}
        for index,word in enumerate(sentence):
            # print(word)
            tagProbabilities[index] = {}
            #hardcode probability that START and END correspond to tags START and END wit a probability of 1
            if word=='START':
                tagProbabilities[index][word] = {'PROB' : math.log(.99999), 'PREVTAG' : None}
                continue

            # For each of the possible states you can transition into from each of the possible previous states
            # find the probability of landing in those states    
            for tag in tagOccurenceProbability:
                #contains the probabilities of the state:tag given the probabilities of each of the possible previous tags
                possibleNodeProbabilities = {}
            
                for prevTag in tagProbabilities[index - 1]:
                    if prevTag in transitionProbabilities[tag]:
                        # print('other')
                        if word in emissionProbabilities[tag]:
                            edgeProbability = (transitionProbabilities[tag][prevTag]) + (emissionProbabilities[tag][word])
                        else:
                            edgeProbability = (transitionProbabilities[tag][prevTag]) + (emissionProbabilities[tag]['UNK'])
                            ############
                    elif word in emissionProbabilities[tag]:
                        # print('other')
                        edgeProbability = (transitionProbabilities[tag]['UNK']) + (emissionProbabilities[tag][word])
                    else:
                        # print('UK')
                        edgeProbability = transitionProbabilities[tag]['UNK'] + emissionProbabilities[tag]['UNK']
                    # print(prevTag,tagProbabilities[index-1][prevTag]['PROB'])
                    nodeProbability = tagProbabilities[index-1][prevTag]['PROB']

                    possibleNodeProbabilities[prevTag] = edgeProbability + nodeProbability
                # choose the previous tag which gives the current tag the max probability
                MLprevTag = max(possibleNodeProbabilities, key=possibleNodeProbabilities.get)
                # print(MLprevTag, possibleNodeProbabilities[MLprevTag])
                tagProbabilities[index][tag] = {
                                                'PROB': possibleNodeProbabilities[MLprevTag],
                                                'PREVTAG': MLprevTag
                                                }

        # print(max(tagProbabilities, key=possibleNodeProbabilities.get))
        probabilities = {}
        index = len(sentence) - 1
        for tag in tagProbabilities[index]:
            probabilities[tag] = tagProbabilities[index][tag]['PROB']
        prevTag = max(probabilities, key=probabilities.get)

        reverseTags = []
        indexIntagProbabilities = len(sentence) - 2
        while prevTag != None:
            reverseTags.append(prevTag)
            prevTag = tagProbabilities[index][prevTag]['PREVTAG']
            index -= 1

        # print((reverseTags))

        taggedSentence = []
        
        index = len(sentence) - 1
        for i, word in enumerate(sentence):
            taggedSentence.append((word, reverseTags[index]))
            index -=1

        taggedSentences.append(taggedSentence)
    

    return taggedSentences


def train_probabilities(train):
    #How often does a particular tag follow another tag
    #takes a tag and returns a dictionary that takes a tag and returns how many times the first tag came after the second tag
    tagPairs = {}

    #keep track of the number of times a tag yiels a word w
    #dict with tag as key which returns a dict that takes a word as a key which returns the number of times the tag produced that word
    tagToWord = {}

    #key is a particular tag, value is the total number of instances of a tag in the train set
    tags = {'COUNT' : 0}

    first = True
    count = 0

    for sentence in train:
        for w in sentence:
            word = w[0]
            tag = w[1]
            #add to tags-> words counter
            #keep track of the total number of words produced by a tag via 'COUNT'
            if tag in tagToWord:
                tagToWord[tag]['COUNT'] += 1 
                if word in tagToWord[tag]:
                    tagToWord[tag][word] = tagToWord[tag][word] + 1
                else:
                    tagToWord[tag][word] = 1
            else:
                tagToWord[tag] = {word : 1}
                tagToWord[tag]['COUNT'] = 1 
            #add to tags total
            if tag not in tags:
                tags['COUNT'] += 1
                tags[tag] = 1
            else:
                tags['COUNT'] += 1
                tags[tag] += 1

            #The very first word will have no previous state to build off of
            #Therefore ignore it to instantiate the HMM
            if first:
                prevTag = tag
                first = False
                continue

            #add tag pair to TagPairs
            if (tag in tagPairs):
                tagPairs[tag]['COUNT'] += 1 
                # check if the previous tag has come before the current tag before
                if prevTag in tagPairs[tag]:
                    tagPairs[tag][prevTag] = tagPairs[tag][prevTag] + 1
                else:
                    tagPairs[tag][prevTag] = 1
            else:
                tagPairs[tag] = {prevTag : 1}
                tagPairs[tag]['COUNT'] = 1 
            prevTag = tag

    #Calculate probabilities
    emissionProbabilities = {}
    transitionProbabilities = {}
    tagOccurenceProbability = {}

    #smoothing parameter
    k = 1e-10

    for tag in tagToWord:
        emissionProbabilities[tag] = {}
        for word in tagToWord[tag]:
            if word == 'COUNT':
                continue
            emissionProbabilities[tag][word] = math.log((tagToWord[tag][word] + 1 * k) / (tagToWord[tag]['COUNT'] + k * len(tagToWord[tag])))
        #Probability that a tag produces an unknown word
        emissionProbabilities[tag]['UNK'] = math.log((k) / (tagToWord[tag]['COUNT'] + k * len(tagToWord[tag])))
    
    for tag in tagPairs:
        transitionProbabilities[tag] = {}
        for prevTag in tagPairs[tag]:
            if prevTag == 'COUNT':
                continue
            transitionProbabilities[tag][prevTag] = math.log((tagPairs[tag][prevTag] + k*1) / (tagPairs[tag]['COUNT'] + k*len(tagPairs[tag])))
        #Probability that a tag follows a tag it hasn't seen before
        transitionProbabilities[tag]['UNK'] = math.log((k) / (tagPairs[tag]['COUNT'] + k * len(tagPairs[tag])))

    for tag in tags:
        if tag == 'COUNT':
            continue
        tagOccurenceProbability[tag] = math.log((tags[tag]) / (tags['COUNT']))
    # tagOccurenceProbability[tag]
    return emissionProbabilities, transitionProbabilities, tagOccurenceProbability
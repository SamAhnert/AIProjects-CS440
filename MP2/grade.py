import numpy as np
import time
import os
import sys
import reader
import naive_bayes as nb
import naive_bayes_mixture as nbm
from sklearn.metrics import confusion_matrix
import pprint
import argparse
import pickle
import json
import mp2


BIGRAM_PENALTY = 0.25


def bigram_check():
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(
        "data/bigram_check/train",
        "data/bigram_check/dev",
        stemming=False,
        lower_case=False,
        use_tqdm=False,
    )
    predicted_labels = nbm.naiveBayesMixture(
        train_set,
        train_labels,
        dev_set,
        unigram_smoothing_parameter=1.0,
        bigram_smoothing_parameter=1.0,
        bigram_lambda=1.0,
        pos_prior=0.5,
    )
    if isinstance(predicted_labels, np.ndarray):
        predicted_labels = list(predicted_labels.reshape(-1))
    if predicted_labels == [1, 1]:
        return BIGRAM_PENALTY
    else:
        return 1.0


def test_unigram_dev_stem_false_lower_false():
    print("Running unigram test..."+'\n')
    train_set, train_labels, dev_set, dev_labels = reader.load_dataset(
    	"data/spam_data/train",
    	"data/spam_data/dev",
        stemming=False,
        lower_case=False,
        use_tqdm=False
    )
    predicted_labels = nb.naiveBayes(
        train_set, train_labels, dev_set, smoothing_parameter=1.0, pos_prior=0.5)

    if len(predicted_labels) != len(dev_labels):
    	print("The length of the list of predictions is not equivalent to the length of the list of development labels.")
    	errorDict = {
    	'name': 'Unigram test on dev set without stemming and without lowercase',
		'score': 0,
		'max_score': 20,
		'visibility': 'visible'
    	}
    	return json.dumps(errorDict, indent=1)
    (
        accuracy,
        f1,
        precision,
        recall,
    ) = mp2.compute_accuracies(predicted_labels, dev_set, dev_labels)
    print("Accuracy:",accuracy)
    print("F1-Score:",f1)
    print("Precision:",precision)
    print("Recall:",recall)

    total_score = 0
    if accuracy >= 0.81:
        total_score += 5
        print("+ 5 points for accuracy  above " + str(0.81))
    else:
        print("Accuracy needs to be above " + str(0.81))
    if accuracy >= 0.86:
        total_score += 5
        print("+ 5 points for accuracy above " + str(0.86))
    else:
        print("Accuracy needs to be above " + str(0.86))
    if accuracy >= 0.91:
        total_score += 5
        print("+ 5 points for accuracy above " + str(0.91))
    else:
        print("Accuracy needs to be above " + str(0.91))
    if accuracy >= 0.95:
        total_score += 5
        print("+ 5 points for accuracy above " + str(0.95))
    else:
        print("Accuracy needs to be above " + str(0.95))
    resultDict = {
    	'name': 'Unigram test on dev set without stemming and without lowercase',
		'score': total_score,
		'max_score': 20,
		'visibility': 'visible'
    }
    return json.dumps(resultDict, indent=1)


def test_bigram_dev_stem_false_lower_false():
        print("Running mixture model test..."+'\n')
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset(
            "data/spam_data/train",
            "data/spam_data/dev",
            stemming=False,
            lower_case=False,
            use_tqdm=False
        )
        predicted_labels = nbm.naiveBayesMixture(train_set, train_labels, dev_set, bigram_lambda=0.05, unigram_smoothing_parameter=1,
            bigram_smoothing_parameter=0.005, pos_prior=0.5)
        if len(predicted_labels) != len(dev_labels):
            print("The length of the list of predictions is not equivalent to the length of the list of development labels.")
            errorDict = {
            'name': 'Mixture model test on dev set without stemming and without lowercase',
            'score': 0,
            'max_score': 5,
            'visibility': 'visible'
            }
            return json.dumps(errorDict, indent=1)
        (
            accuracy,
            f1,
            precision,
            recall
        ) = mp2.compute_accuracies(predicted_labels, dev_set, dev_labels)
        print("Accuracy:",accuracy)
        print("F1-Score:",f1)
        print("Precision:",precision)
        print("Recall:",recall)
        total_score = 0
        if accuracy >= 0.80:
            total_score += 1.25
            print("+ 1.25 points for accuracy  above " + str(0.80))
        else:
            print("Accuracy needs to be above " + str(0.80))
        if accuracy >= 0.85:
            total_score += 1.25
            print("+ 1.25 points for accuracy above " + str(0.85))
        else:
            print("Accuracy needs to be above " + str(0.85))
        if accuracy >= 0.90:
            total_score += 1.25
            print("+ 1.25 points for accuracy above " + str(0.90))
        else:
            print("Accuracy needs to be above " + str(0.90))
        if accuracy >= 0.95:
            total_score += 1.25
            print("+ 1.25 points for accuracy above " + str(0.95))
        else:
            print("Accuracy needs to be above " + str(0.95))

        if bigram_check() == BIGRAM_PENALTY:
            print(f"We hypothesize that your implementation of naiveBayesMixture is not correct. "
                  f"Therefore, we applied a penalty multiplier of {BIGRAM_PENALTY} to your score.")
            total_score *= BIGRAM_PENALTY
        resultDict = {
             'name': 'Mixture test on dev set without stemming and without lowercase',
             'score': total_score,
             'max_score': 5,
             'visibility': 'visible'
        }
        return json.dumps(resultDict, indent = 1)

def print_results():
	unigram_test = test_unigram_dev_stem_false_lower_false()
	print('\n'+unigram_test+'\n')
	mixture_test = test_bigram_dev_stem_false_lower_false()
	print('\n'+mixture_test+'\n')

if __name__ == '__main__':
    print_results()

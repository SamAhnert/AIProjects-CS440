import unittest
from gradescope_utils.autograder_utils.decorators import weight,visibility
from gradescope_utils.autograder_utils.files import check_submitted_files
from gradescope_utils.autograder_utils.decorators import partial_credit
import numpy as np
import time
from tests import reader
import neuralnet_part1 as p1
import neuralnet_part2 as p2
import torch

local_dir = "./tests"

def compute_accuracies(predicted_labels, dev_set, dev_labels):
    yhats = predicted_labels
    if len(yhats) != len(dev_labels):
        print("Lengths of predicted labels don't match length of actual labels", len(yhats),len(dev_labels))
        return 0.,0.,0.,0.

    accuracy = np.mean(yhats == dev_labels)

    tp = np.sum([yhats[i] == dev_labels[i] and yhats[i] == 1 for i in range(len(dev_labels))])
    fp = np.sum([yhats[i] != dev_labels[i] and yhats[i] == 1 for i in range(len(dev_labels))])
    fn = np.sum([yhats[i] != dev_labels[i] and yhats[i] == 0 for i in range(len(dev_labels))])

    precision = tp / (tp + fp)
    recall = tp / (fn + tp)
    f1 = 2 * (precision * recall) / (precision + recall)

    return accuracy, f1, precision, recall

class NetworkTest1(unittest.TestCase):

    def setUp(self):
        reader.init_seeds(42)
        start_time = time.time()
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset("mp3_data")
        train_set = torch.tensor(train_set, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.int64)
        mu = train_set.mean(axis=0,keepdims=True)
        std = train_set.std(axis=0,keepdims=True)
        stdize = lambda X: (X-mu)/std
        dev_set = torch.tensor(dev_set, dtype=torch.float32)
        dev_loss, dev_predicted_labels, net = p1.fit(train_set, train_labels, dev_set, 500, 100)

        self.assertEquals(len(dev_predicted_labels), len(dev_labels),f"Incorrect number of labels. Got {len(dev_predicted_labels)} labels when there should be {len(dev_labels)} labels")
        self.net = net
        self.assertEqual(net(dev_set).shape[1], 2, "Network doesn't have output size 2.")
        yhats = np.argmax(net(dev_set).detach().numpy(), axis=1)
        acc_dev_std, f1_dev, _, _ = compute_accuracies(yhats, dev_set, dev_labels)
        yhats = np.argmax(net(stdize(dev_set)).detach().numpy(), axis=1)
        acc_dev, f1_dev, _, _ = compute_accuracies(yhats, dev_set, dev_labels)
        self.dev_acc = np.maximum(acc_dev,acc_dev_std)

        self.time_spend = time.time() - start_time
        print("Time Taken: {0:.2f} secs".format(self.time_spend))
        print("Dev Accuracy: {0:.4f}".format(self.dev_acc))

    @visibility('visible')
    @partial_credit(25)
    def test_dev_set(self,set_score=None):
        """Check if part 1 dev accuracy reaches threshold"""
        # normal test
        total_score = 0
        for threshold in [0.6,0.75,0.794,0.82,0.83]:
            if(self.dev_acc >= threshold):
                total_score += 5
                print("+5 points for dev accuracy above", str(threshold))
            else:
                print("Dev accuracy needs to be above", str(threshold))
                break
        set_score(total_score)

    @visibility('visible')
    @weight(5)
    def test_parameters(self):
        """Test number of parameters"""
        # normal test
        num_parameters = sum([ np.prod(w.shape) for w  in self.net.parameters()])
        threshold = 500000
        self.assertLess(num_parameters, threshold, "Num_parameters: " + str(num_parameters) + ". This should be below " + str(threshold)+ "!")

    @visibility('visible')
    @weight(5)
    def test_running_time(self):
        """predict with neuralnet.py run within 2min"""
        # normal test
        threshold = 120
        self.assertGreater(threshold, self.time_spend, "Runtime too long: " + str(self.time_spend) + ". This should be within 2 min!")

class NetworkTest2(unittest.TestCase):

    def setUp(self):
        reader.init_seeds(42)
        start_time = time.time()
        train_set, train_labels, dev_set, dev_labels = reader.load_dataset("mp3_data")
        train_set = torch.tensor(train_set, dtype=torch.float32)
        train_labels = torch.tensor(train_labels, dtype=torch.int64)
        mu = train_set.mean(axis=0,keepdims=True)
        std = train_set.std(axis=0,keepdims=True)
        stdize = lambda X: (X-mu)/std
        dev_set = torch.tensor(dev_set, dtype=torch.float32)
        dev_loss, dev_predicted_labels, net = p2.fit(train_set, train_labels, dev_set, 500, 100)

        self.assertEquals(len(dev_predicted_labels), len(dev_labels),f"Incorrect number of labels. Got {len(dev_predicted_labels)} when there should be {len(dev_labels)} labels")
        self.net = net
        self.assertEqual(net(dev_set).shape[1], 2, "Network doesn't have output size 2.")
        yhats = np.argmax(net(dev_set).detach().numpy(), axis=1)
        acc_dev_std, f1_dev, _, _ = compute_accuracies(yhats, dev_set, dev_labels)
        yhats = np.argmax(net(stdize(dev_set)).detach().numpy(), axis=1)
        acc_dev, f1_dev, _, _ = compute_accuracies(yhats, dev_set, dev_labels)
        self.dev_acc = np.maximum(acc_dev,acc_dev_std)

        self.time_spend = time.time() - start_time
        print("Time Taken: {0:.2f} secs".format(self.time_spend))
        print("Dev Accuracy: {0:.4f}".format(self.dev_acc))

    @visibility('visible')
    @partial_credit(15)
    def test_dev(self,set_score=None):
        """Check dev accuracy"""
        # normal test
        total_score = 0
        for threshold in [0.845,0.86,0.87]:
            if(self.dev_acc >= threshold):
                total_score += 5
                print("+5 points for dev accuracy above", str(threshold))
            else:
                print("Dev accuracy needs to be above", str(threshold))
                break
        set_score(total_score)

    @visibility('visible')
    @weight(10)
    def test_parameters(self):
        """Test number of parameters"""
        # normal test
        num_parameters = sum([ np.prod(w.shape) for w  in self.net.parameters()])
        threshold = 500000
        self.assertLess(num_parameters, threshold, "Num_parameters: " + str(num_parameters) + ". This should be below " + str(threshold)+ "!")

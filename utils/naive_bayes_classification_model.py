"""
Source code for the Naive Bayes sentiment classification model.
"""
# Import for defaultdict
import collections
# Imports for sentiment analyis utilities
from sentiment_analysis_utils import *

class NaiveBayes:
    def __init__(self, unique_classes):
        self.classes = unique_classes
        self.bag = [collections.defaultdict(int) for class in self.classes]
        # NOTE: for nonexistent key k, the key-value pair (k,0) is automatically
        # generated and added to the dictionary

    def add_to_bag(self, example, label):
        for word in example.split():
            self.bag[label][word] += 1

    def train(self, data, labels):
        # Construct bag of word for each label category
        for index, label in enumerate(self.classes):
            class_examples = data[labels==label] # get all examples with specified label
            cleaned_class_examples = remove_punctuation_and_lower(class_examples) # clean
            cleaned_class_examples = pd.DataFrame(data=cleaned_class_examples) # convert to dataframe

            np.apply_along_axis(func1d=self.add_to_bag, axis=1,
                                arr=cleaned_class_examples, index) # add to corresponding bag

        class_probabilities, class_word_counts = np.empty(len(self.classes)), np.empty(len(self.classes))
        all_words = []
        for index, label in enumerate(self.classes):
            # Calculate Prior Probability P(C) for each class
            class_probabilities[index] = np.sum(labels==label)/len(labels)
            # Calculate total counts of all the words for each class
            counts = np.array(self.bag[index].values())
            class_word_counts[index] = np.sum(counts)+1
            # Get all words of this class
            all_words += self.bag[index].keys()

        # Combine all words of every category & make them unique to get vocabulary -V- of entire training set
        self.vocabulary = np.unique(np.array(all_words))
        self.vocabulary_length = self.vocab.shape[0]

        # Computing denominator value
        tmp = []
        for index, class in enumerate(self.classes):
            tmp.append(class_word_counts[index] + self.vocabulary_length + 1)
        denominators = np.array(tmp)
        # Continue from here!!!

        self.precomputed = np.array([(self.bag[index],
                                      class_probabilities[index],
                                      denominators[index]) for index, label in enumerate(self.classes)])

    def compute_test_probability(self, example):
        # Initialize the array to store probability for each class
        likelihood_probability = np.zeros(self.classes.shape[0])
        # Find probability for each class of the given test example
        for index, label in enumerate(self.classes):
            # Split the test example and get probability of each test word
            for word in example.split():
                # Get total count of this test token from it's respective training dict to get numerator value
                word_counts = self.precomputed[index][0].get(word, 0) + 1
                # Get likelihood of this word
                word_probability = word_counts / float(self.precomputed[index][2])
                # Take logarithm to prevent underflow
                likelihood_probability[index] += np.log(word_probability)

        # Calculate posterior probability
        posterior_probability = np.empty(self.classes.shape[0])
        for index, label in enumerate(self.classes):
            posterior_probability[index] = likelihood_probability[index] + np.log(self.precomputed[index][1])

        return posterior_probability

    def test(self, data):
        # Initiralize store prediction of each test example
        predictions = []
        for example in data:
            # Preprocess the test example the same way we did for training set exampels
            cleaned_example = remove_punctuation_and_lower(example)
            # Simply get the posterior probability of every example
            post_probability = self.compute_test_probability(cleaned_example)
            # Simply pick the max value and map against self.classes
            predictions.append(self.classes[np.argmax(post_probability)])
        return np.array(predictions)

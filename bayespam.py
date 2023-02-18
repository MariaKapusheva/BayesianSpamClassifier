import argparse
import os, string
from math import log
from enum import Enum

class MessageType(Enum):
    REGULAR = 1,
    SPAM = 2

class Counter():

    def __init__(self):
        self.counter_regular = 0
        self.counter_spam = 0

    def increment_counter(self, message_type):
        """
        Increment a word's frequency count by one, depending on whether it occurred in a regular or spam message.

        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            self.counter_regular += 1
        else:
            self.counter_spam += 1

class Bayespam():

    def __init__(self):
        self.regular_list = None
        self.spam_list = None
        self.vocab = {}
        self.prior_probability_regular = 0
        self.prior_probability_spam = 0
        self.class_conditional_regular = []
        self.class_conditional_spam = []
        self.posteriori_log_regular = 0
        self.posteriori_log_spam = 0
        self.msg_class = {}

    def list_dirs(self, path):
        """
        Creates a list of both the regular and spam messages in the given file path.

        :param path: File path of the directory containing either the training or test set
        :return: None
        """
        # Check if the directory containing the data exists
        if not os.path.exists(path):
            print("Error: directory %s does not exist." % path)
            exit()

        regular_path = os.path.join(path, 'regular')
        spam_path = os.path.join(path, 'spam')

        # Create a list of the absolute file paths for each regular message
        # Throws an error if no directory named 'regular' exists in the data folder
        try:
            self.regular_list = [os.path.join(regular_path, msg) for msg in os.listdir(regular_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'regular'." % path)
            exit()

        # Create a list of the absolute file paths for each spam message
        # Throws an error if no directory named 'spam' exists in the data folder
        try:
            self.spam_list = [os.path.join(spam_path, msg) for msg in os.listdir(spam_path)]
        except FileNotFoundError:
            print("Error: directory %s should contain a folder named 'spam'." % path)
            exit()

    def read_messages(self, message_type):
        """
        Parse all messages in either the 'regular' or 'spam' directory. Each token is stored in the vocabulary,
        together with a frequency count of its occurrences in both message types.
        :param message_type: The message type to be parsed (MessageType.REGULAR or MessageType.SPAM)
        :return: None
        """
        if message_type == MessageType.REGULAR:
            message_list = self.regular_list
        elif message_type == MessageType.SPAM:
            message_list = self.spam_list
        else:
            message_list = []
            print("Error: input parameter message_type should be MessageType.REGULAR or MessageType.SPAM")
            exit()

        for msg in message_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line)):
                        ## 1. Make each token case-insensitive, removes punctuation, and numerals
                        token = split_line[idx].lower()
                        token = token.translate(str.maketrans('','', string.punctuation))
                        token = token.translate(str.maketrans('','', '1234567890'))
                        ## 1. If the length of the token/word is less than 4 letters, skip it and continue looping
                        if len(token) < 4:
                            continue
                        if token in self.vocab.keys():
                            # If the token is already in the vocab, retrieve its counter
                            counter = self.vocab[token]
                        else:
                            # Else: initialize a new counter
                            counter = Counter()

                        # Increment the token's counter by one and store in the vocab
                        counter.increment_counter(message_type)
                        self.vocab[token] = counter
            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()

    def print_vocab(self):
        """
        Print each word in the vocabulary, plus the amount of times it occurs in regular and spam messages, and
        their respective log probabilities
        :return: None
        """

        count = 0
        for word, counter in self.vocab.items():
            # repr(word) makes sure that special characters such as \t (tab) and \n (newline) are printed.
            print("%s | Frequency in regular: %d | Probability log: %f | Frequency in spam: %d |  Probability log: %f" %
                  (repr(word), counter.counter_regular, self.class_conditional_regular[count], counter.counter_spam,
                   self.class_conditional_spam[count]))
            count += 1

    def compute_probabilities(self):
        """
        Compute priors, class conditional word likelihoods, and then convert the probabilites into log probabilites
        :return: None
        """
        # TODO: split into helper functions?

        ## 2.1
        n_messages_regular = len(self.regular_list)
        n_messages_spam = len(self.spam_list)
        n_messages_total = n_messages_regular + n_messages_spam
        ## finds the prior probabilities and converts them to log
        self.prior_probability_regular = log(n_messages_regular / n_messages_total)
        self.prior_probability_spam = log(n_messages_spam / n_messages_total)

        ## 2.2
        e = 0.60## tuning parameter
        n_words_regular = 0
        n_words_spam = 0
        ## counts the number of words in each set by looping through the vocabulary
        for word, counter in self.vocab.items():
            n_words_regular += counter.counter_regular
            n_words_spam += counter.counter_spam


        ## loops through every word in the vocabulary and computes the class conditional word likelihoods
        for word, counter in self.vocab.items():
            ## cases for zero probabilites / estimated by a small non-zero value
            if counter.counter_regular == 0:
                self.class_conditional_regular.append(e / (n_words_regular + n_words_spam))
                self.class_conditional_spam.append(counter.counter_spam / n_words_spam)
            elif counter.counter_spam == 0:
                self.class_conditional_spam.append(e / (n_words_regular + n_words_spam))
                self.class_conditional_regular.append(counter.counter_regular / n_words_regular)
            else:
                self.class_conditional_regular.append(counter.counter_regular / n_words_regular)
                self.class_conditional_spam.append(counter.counter_spam / n_words_spam)
        ## converting into log
        for i in range(len(self.class_conditional_regular)):
            self.class_conditional_regular[i] = log(self.class_conditional_regular[i])
            self.class_conditional_spam[i] = log(self.class_conditional_spam[i])

    def classify_test(self, bayespam):
        ## Creating lists with the messages and appending the regular and the spam messages to it
        msg_list = []

        for i in range(len(self.spam_list)):
            msg_list.append(self.spam_list[i])

        for i in range(len(self.regular_list)):
            msg_list.append(self.regular_list[i])

        ## Looping over the messages in the test and extracting the words
        for msg in msg_list:
            try:
                # Make sure to use latin1 encoding, otherwise it will be unable to read some of the messages
                f = open(msg, 'r', encoding='latin1')

                # Loop through each line in the message
                for line in f:
                    # Split the string on the space character, resulting in a list of tokens
                    split_line = line.split(" ")
                    # Loop through the tokens
                    for idx in range(len(split_line)):
                        ## 1. Make each token case-insensitive, removes punctuation, and numerals
                        token = split_line[idx].lower()
                        token = token.translate(str.maketrans('','', string.punctuation))
                        token = token.translate(str.maketrans('','', '1234567890'))
                        ## 1. If the length of the token/word is less than 4 letters, skip it and continue looping
                        if len(token) < 4:
                            continue
                        if token in bayespam.vocab.keys():
                            ## 3.1 Accessing the index of the probability of the word and adding it to the regular and spam posterior probability
                            ind = list(bayespam.vocab).index(token)
                            self.posteriori_log_regular += bayespam.class_conditional_regular[ind]
                            self.posteriori_log_spam += bayespam.class_conditional_spam[ind]
                ## 3.1 Adding the prior probabilities to the posterior
                self.posteriori_log_regular += bayespam.prior_probability_regular
                self.posteriori_log_spam += bayespam.prior_probability_spam
                ## 3.1 Comparing the spam and the regular posterior probability and appending the message and its type to a dictionary
                if self.posteriori_log_regular > self.posteriori_log_spam:
                    self.msg_class[msg] = MessageType.REGULAR
                    print("Regular")
                else:
                    self.msg_class[msg] = MessageType.SPAM
                    print("Spam")
                ## 3.1 Setting the posterior probabilities to 0
                self.posteriori_log_spam = 0
                self.posteriori_log_regular = 0



            except Exception as e:
                print("Error while reading message %s: " % msg, e)
                exit()


    ## 3.2 Function for calculating the confusion matrix
    def confusion_matrix(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        ## Counting the TP, TN, FP, FN
        for i in range(len(self.regular_list)):
            if self.msg_class[self.regular_list[i]] == MessageType.REGULAR:
                TP += 1
            else:
                FN += 1
        for j in range(len(self.spam_list)):
            if self.msg_class[self.spam_list[j]] == MessageType.SPAM:
                TN += 1
            else:
                FP += 1

        ## Printing the TP, FP, TN, FN in percentages
        print("TP: ", TP*100/len(self.regular_list))
        print("FP: ", FP*100/len(self.spam_list))
        print("TN: ", TN*100/len(self.spam_list))
        print("FN: ", FN*100/len(self.regular_list))
        print("Accuracy: %f %%" % (((TP + TN) / (TP + TN + FN + FP)) * 100))






    def write_vocab(self, destination_fp, sort_by_freq=False):
        """
        Writes the current vocabulary to a separate .txt file for easier inspection.

        :param destination_fp: Destination file path of the vocabulary file
        :param sort_by_freq: Set to True to sort the vocab by total frequency (descending order)
        :return: None
        """

        if sort_by_freq:
            vocab = sorted(self.vocab.items(), key=lambda x: x[1].counter_regular + x[1].counter_spam, reverse=True)
            vocab = {x[0]: x[1] for x in vocab}
        else:
            vocab = self.vocab

        try:
            f = open(destination_fp, 'w', encoding="latin1")

            for word, counter in vocab.items():
                # repr(word) makes sure that special  characters such as \t (tab) and \n (newline) are printed.
                f.write("%s | In regular: %d | In spam: %d\n" % (repr(word), counter.counter_regular, counter.counter_spam),)

            f.close()
        except Exception as e:
            print("An error occurred while writing the vocab to a file: ", e)



def main():
    # We require the file paths of the training and test sets as input arguments (in that order)
    # The argparse library helps us cleanly parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('train_path', type=str,
                        help='File path of the directory containing the training data')
    parser.add_argument('test_path', type=str,
                        help='File path of the directory containing the test data')
    args = parser.parse_args()

    # Read the file paths of the folder containing the training and testing set from the input arguments
    train_path = args.train_path
    test_path = args.test_path


    # Initialize a Bayespam object
    bayespam = Bayespam()
    # Initialize a list of the regular and spam message locations in the training folder
    bayespam.list_dirs(train_path)

    # Parse the messages in the regular message directory
    bayespam.read_messages(MessageType.REGULAR)
    # Parse the messages in the spam message directory
    bayespam.read_messages(MessageType.SPAM)
    bayespam.compute_probabilities()

    ## NEW: added printing of probabilities
   # bayespam.print_vocab()

    # Initialize a new Bayespam object for the test data
    test_bayespam = Bayespam()
    ## Initialize a list of the regular and spam message location in the test data
    test_bayespam.list_dirs(test_path)
    test_bayespam.classify_test(bayespam)
    ## Calculating and printing the confusion matrix
    test_bayespam.confusion_matrix()

    # bayespam.write_vocab("vocab.txt")

    print("N regular messages: ", len(bayespam.regular_list))
    print("N spam messages: ", len(bayespam.spam_list))


if __name__ == "__main__":
    main()

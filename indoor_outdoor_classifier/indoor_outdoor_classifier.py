"""
The MIT License (MIT)

Copyright (c) 2016 William Falcon

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from sklearn import svm
import pandas as pd
import numpy as np

TRAINING_DATA_PATH = './data/inOutTrng.json'

# Will use this percent of data to train. Will use 1-this to test
IN_TEST_PERCENT = 0.8


def run_demo():
    """
    Demo if used as main
    """
    clf = InOutClassifier(verbose=True)
    indoor_a = [130, 9, -1, -1]
    indoor_b = [6, 9, 2.0, -1]
    outdoor_a = [120, 4, 2.0, 20]
    outdoor_b = [40, 4, 2.0, 1.5]
    prediction = clf.predict([indoor_a, indoor_b, outdoor_a, outdoor_b])
    print prediction


class InOutClassifier(object):
    """
    Author: William Falcon

    Classifier capable of distinguishing between outside and inside
    using various parameters from GPS data.

    Prediction results:
    0 - Outdoors
    1 - Indoors

    Input vector order:
    [gpsAccuracyHor, gpsAccuracyVert, gpsCourse, gpsSpeed]
    """

    def __init__(self, verbose=False):
        print 'training...'
        self.clf = self.__train_in_out_svm(verbose)
        print 'training complete!...'


    def predict(self, data_points):
        """
        Wrapper around the prediction func of svm
        """
        # make three features binary for better accuracy
        self.__normalize_data_points(data_points)
        return self.clf.predict(data_points)


    def __normalize_data_points(self, data_points):
        """
        Puts binary restrictions on each data point
        """
        for data_point in data_points:
            data_point[0] = self.__vert_compare_dp(data_point[0])
            data_point[2] = 0 if data_point[2] <= -1 else 1
                
        
    def __train_in_out_svm(self, verbose):
        """
        Trains an svm using 80/20 from the json file path.
        Prints stats about accuracy on completion
        """
        clf = svm.SVC()

        # load tng and testing data
        X_train, Y_train, X_test, Y_test = self.__load_tng_data()

        # train classifier
        clf.fit(X_train, Y_train)

        # measure accuracy and print stats
        accuracy = self.__test_accuracy(clf, X_test, Y_test)

        # print stats if requested
        if verbose:
            self.print_svm_stats(clf, accuracy, len(X_train), len(X_test))

        return clf


    def print_svm_stats(self, clf, accuracy, train_size, test_size):
        print '-------------------------------'
        print 'SVM Config: '
        print clf
        print '\nAccuracy: %.2f' % (accuracy*100)
        print 'Trained with: ', train_size
        print 'Tested with: ', test_size
        print '-------------------------------'


    def __test_accuracy(self, clf, X_test, Y_test):
        """
        Predicts and counts against training data for accuracy
        """
        matches = 0
        for i, x in enumerate(X_test):
            prediction = clf.predict([x])

            if prediction[0] == Y_test[i]:
                matches += 1

        return matches / float(len(Y_test))


    def __load_tng_data(self):
        """
        Load data into frames, then to numpy arrays
        """

        # load json
        df = pd.read_json(TRAINING_DATA_PATH)
        df['gpsAccuracyHor'] = df.apply(self.__vert_compare, axis=1)
        self.__make_feature_binary(df, 'gpsCourse', -1)

        # generate testing and training data
        mask = np.random.rand(len(df)) < IN_TEST_PERCENT
        X_train = df[mask]
        X_test = df[~mask]

        # Get Y values on their own
        Y_train = X_train['inOut']
        Y_test = X_test['inOut']

        # Remove Y values from DataFrames
        del X_train['inOut']
        del X_test['inOut']

        # return as numpy arrays for easier input into the SVM
        return X_train.as_matrix(), Y_train.as_matrix(), X_test.as_matrix(), Y_test.as_matrix()
        
        
    def __vert_compare(self, s):
        """
        Compare function to normalize all gps horizontal accuracies
        """
        accu = s['gpsAccuracyHor']
        if accu <= 50:
            return 0
        elif accu <= 80:
            return 1
        else:
            return 2
    
    
    def __vert_compare_dp(self, accu):
        """
        Compare function to normalize all tessting points for predict
        """
        if accu <= 50:
            return 0
        elif accu <= 80:
            return 1
        else:
            return 2

                
    def __make_feature_binary(self, df, feature_name, bin_threshold):
        """
        Changes a numerical feature to binary to
        make decision boundary simpler
        """
        # change feature to binary given the threshold
        boolFunc = lambda s: 0 if (s[feature_name] <= bin_threshold) else 1
        df[feature_name] = df.apply(boolFunc, axis=1)


if __name__ == '__main__':
    run_demo()

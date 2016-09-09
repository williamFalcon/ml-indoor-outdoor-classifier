from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import cross_val_predict
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn import metrics
from sklearn import svm
import pandas as pd
import numpy as np
import os

TRAINING_DATA_PATH = '/data/train.csv'
TESTING_DATA_PATH = '/data/testa.csv'

# ranked in order as chosen by feature selection
X_FEATURES = [
'gps_speed', 
'gps_vertical_accuracy',
'gps_horizontal_accuracy'
]

Y_FEATURE = ['indoors']


class InOutClassifier(object):
    """
    Author: William Falcon

    Classifier capable of distinguishing between outside and inside
    using various parameters from GPS data.

    Prediction results:
    0 - Outdoors
    1 - Indoors
    """
    def __init__(self, print_tng_accuracy=False):
        print 'training...'
        self.clf = self.__train_in_out_svm(print_tng_accuracy)
        print 'training complete!...'

    #----------------------------------------------------
    # LEARNING AND PREDICTING
    #----------------------------------------------------
    def predict(self, data_points):
        """
        Wrapper around the prediction func of svm
        """
        return self.clf.predict(data_points)


    def __train_in_out_svm(self, print_tng_accuracy):
        """
        Prints stats about accuracy on completion
        """
        clf = svm.SVC()

        # load tng and testing data
        X_train, Y_train = self.__load_data(TRAINING_DATA_PATH)

        # train classifier
        clf.fit(X_train, Y_train)

        # print accuracy stats if requested
        if print_tng_accuracy:
            X_test, Y_test = self.__load_data(TESTING_DATA_PATH)
            self.__test_features(X_train, Y_train, X_test, Y_test)

            # measure accuracy and print stats
            predicted = cross_val_predict(clf, X_test, Y_test)
            test_scores = metrics.accuracy_score(Y_test, predicted)
            self.__print_svm_stats(clf, len(X_train), len(X_test), test_scores)

        return clf

    #----------------------------------------------------
    # FEATURES UTILS
    #----------------------------------------------------
    def __generate_features(self, df):
        # shuffle first
        df_shuffled = df.iloc[np.random.permutation(len(df))]
        df_shuffled = df_shuffled.reset_index(drop=True)

        # extract the necessary features
        X = df_shuffled[X_FEATURES]
        Y = df_shuffled[Y_FEATURE]

        # scale and generate new features

        # flatten and cast to proper sklearn format
        X = X.values.astype(np.float64)
        Y = Y.values.astype(np.int32).flatten()

        return X, Y

    def __test_features(self, X_train, Y_train, X_test, Y_test):
        """
        Runs feature selection. Pass in the dimensions you think are important
        and it will rank them by importance
        """
        print('running feature selection...')
        model = ExtraTreesClassifier()
        model.fit(X_train, Y_train)

        # display the relative importance of each attribute
        print(model.feature_importances_)
        print('---------------------------')
        model = LogisticRegression()

        # create the RFE model and select 3 attributes
        rfe = RFE(model, 3)
        rfe = rfe.fit(X_train, Y_train)

        # summarize the selection of the attributes
        print(rfe.support_)
        print(rfe.ranking_)

    #----------------------------------------------------
    # UTILS
    #----------------------------------------------------
    def __print_svm_stats(sefl, clf, train_size, test_size, test_scores):
        print('\n-------------------------------')
        print('SVM Config: ')
        print(clf)
        print('Trained with: ')
        print(train_size)
        print('Testing accuracy: ')
        print(test_scores)
        print('Tested with: %d' %(test_size))

        print('-------------------------------')


    def __load_data(self, path):
        """
        Load data into frames, then to numpy arrays
        """
        full_path = os.path.dirname(os.path.realpath(__file__)) + path

        # load json
        cols = X_FEATURES.extend(Y_FEATURE)
        df = pd.read_csv(full_path, usecols=cols)

        print('generating features...')
        X_train, Y_train = self.__generate_features(df)

        return X_train, Y_train

def run_demo():
    print('running demo...')
    clf = InOutClassifier(print_tng_accuracy=True)

if __name__ == '__main__':
    run_demo()
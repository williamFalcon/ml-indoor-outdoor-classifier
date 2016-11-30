from sklearn.ensemble import RandomForestClassifier
import pickle
import os

WORKING_DIR = os.path.dirname(os.path.realpath(__file__))


class InOutClassifier(object):
    """
    A classifier to predict whether someone is indoors or outdoors by using
    gps sensor data

    Example use:

    clf = InOutClassifier()

    x1 = [10, 200, 20.30, 12]
    x2 = [20, 100, 20.50, 30]
    X = [x1, x2]

    y_hat = clf.predict(X)
    # y_hat = [1,0]

    """

    def __init__(self):
        """
        On init, load classifier model
        :return:
        """
        self.clf = self.__load_model()

    def predict(self, X):
        """
        a single x contains for features in this order:
        gps_vertical_accuracy, gps_horizontal_accuracy, gps_course, gps_speed

        example:
        x1 = [10, 200, 20.30, 12]
        x2 = [20, 100, 20.50, 30]
        X = [x1, x2]

        :param X:
        :return: Array of predictions (1 if inside, 0 if outside)
        """
        return self.clf.predict(X)

    def __load_model(self):
        """
        Loads the trained Random forest classifier
        Classifier was trained using a different script
        :return: RF classifier
        """
        clf = pickle.load(open("%s/in_out_model.p" % WORKING_DIR, "rb"))
        return clf





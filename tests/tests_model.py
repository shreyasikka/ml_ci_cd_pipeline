import unittest
from sklearn.ensemble import LogisticRegression
import pickle

class TestModelTraining(unittest.TestCase):
    def test_model_training(self):
        model=pickle.load('iris_logistic_regression.pkl')
        self.assertisInstance(model, LogisticRegression)
        self.assertGreaterequal(len(model.feature_importances_),4)

if __name__=='__main__':
    unittest.main()
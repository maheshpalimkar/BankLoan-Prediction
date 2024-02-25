import pickle
import numpy as np
import config

class PredictModel:
    def __init__(self):
        # Load the Decision Tree model during class initialization
        with open(config.MODEL_FILE_PATH, 'rb') as f:
            self.dt_clf = pickle.load(f)
        
        # List of features used by the model
        self.features_list = ['Age', 'Experience', 'Income', 'Family', 'CCAvg',
                              'Education', 'Mortgage', 'Securities.Account',
                              'CD.Account', 'Online', 'CreditCard']

    def create_test_array(self, age, experience, income, family, cc_avg, education, mortgage, securities_account, cd_account, online, credit_card):
        test_array = np.array([age, experience, income, family, cc_avg, education, mortgage, securities_account, cd_account, online, credit_card], ndmin=2)
        return test_array

    def predict_class(self, test_array):
        # Predict the class using the loaded Decision Tree model
        predicted_class = self.dt_clf.predict(test_array)[0]

        return predicted_class


    
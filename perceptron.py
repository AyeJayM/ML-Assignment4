#-------------------------------------------------------------------------
# AUTHOR: Austin Martinez
# FILENAME: perceptron.py
# SPECIFICATION: Read in training data from optdigits.tra and test data from optdigits.tes
#                Both perception and multi-layer perceptrons follow this program logic:
#                Create the respective neural network classifier. Then, fit it to the 
#                training data. Now, we initialize a variable to hold the number of 
#                correct predictions. Now we make predictions by testing on an x_sample
#                and making a label prediction. We then compare to the truel label and
#                increment the correct predictions variable if we got it right. We divide
#                the total correct predicitons by the total amount of test data to get our
#                accuracy and if it is higher than the previously calculated accuracy, we print.
#                
# FOR: CS 4210- Assignment #4
# TIME SPENT: This program specifically took 4 hours.
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

#importing some Python libraries
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) #reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] #getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  #getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) #reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    #getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     #getting the last field to form the class label for test


highestAccuracyPerceptron = 0.0
perceptronParametersDict = {
       'learning_rate': None,
       'shuffle': None
       }

highestAccuracyMLP = 0
mlpParametersDict = {
       'learning_rate': None,
       'shuffle': None
       }

for learning_rate in n: #iterates over n
    for shuffle in r: #iterates over r
                

                ####################
                # PERCEPTRON LOGIC #
                ####################

                # Create a Neural Network classifier
                # clf = Perceptron()

                # #use those hyperparameters:
                # eta0 = learning rate, shuffle = shuffle the training data, max_iter=1000

                # Create a Perceptron classifier
                perceptron_clf = Perceptron(eta0=learning_rate, shuffle=shuffle, max_iter=1000)

                # Fit the Neural Network to the training data
                perceptron_clf.fit(X_training, y_training)

                # Initialize the variable to hold correct predictions
                perceptron_correct = 0
               
                # We will iterate through the X_test and y_test data
                for x_sample, y_sample in zip(X_test, y_test):
                    # If the predicted label for x_sample matches the true y_label, then increase correct predictions by 1
                    if (perceptron_clf.predict([x_sample])[0]) == y_sample:
                        perceptron_correct += 1
    
                        # Divide correct predictions by total amount of test data to calculate accuracy
                        perceptron_accuracy = perceptron_correct / len(y_test)
                        
                        # If the calculated accuracy is higher than the previous calculated accuracy, we update our highest accuracy and print!
                        if perceptron_accuracy > highestAccuracyPerceptron:
                            highestAccuracyPerceptron = perceptron_accuracy
                            perceptronParametersDict['learning_rate'] = learning_rate
                            perceptronParametersDict['shuffle'] = shuffle
                            print(f"Highest Perceptron accuracy so far: {highestAccuracyPerceptron}, Parameters: learning rate={learning_rate}, shuffle={shuffle}\n")

for learning_rate in n: #iterates over n
        for shuffle in r: #iterates over r
                    

                    ####################
                    #    MLP LOGIC     #
                    ####################               
                
                    #   clf = MLPClassifier() #use those hyperparameters: activation='logistic', learning_rate_init = learning rate,
                    #                          hidden_layer_sizes = number of neurons in the ith hidden layer - use 1 hidden layer with 25 neurons,
                    #                          shuffle = shuffle the training data, max_iter=1000
                    mlp_clf = MLPClassifier(activation='logistic', learning_rate_init=learning_rate,
                                    hidden_layer_sizes=(25,), shuffle=shuffle, max_iter=1000)

                    # Fit the Neural Network to the training data
                    mlp_clf.fit(X_training, y_training)

                    # Initialize the variable to hold correct predictions
                    mlp_correct = 0
                    
                    # We will iterate through the X_test and y_test data
                    for x_sample, y_sample in zip(X_test, y_test):
                         # If the predicted label for x_sample matches the true y_label, then increase correct predictions by 1
                        if (mlp_clf.predict([x_sample])[0]) == y_sample:
                            mlp_correct += 1
                
                        # Divide correct predictions by total amount of test data to calculate accuracy
                        mlp_accuracy = mlp_correct / len(y_test)
                
                        # If the calculated accuracy is higher than the previous calculated accuracy, we update our highest accuracy and print!
                        if mlp_accuracy > highestAccuracyMLP:
                            highestAccuracyMLP = mlp_accuracy
                            mlpParametersDict['learning_rate'] = learning_rate
                            mlpParametersDict['shuffle'] = shuffle
                            print(f"Highest MLP accuracy so far: {highestAccuracyMLP}, Parameters: learning rate={learning_rate}, shuffle={shuffle}\n")
                            

                    #make the classifier prediction for each test sample and start computing its accuracy
                    #hint: to iterate over two collections simultaneously with zip() Example:
                    #for (x_testSample, y_testSample) in zip(X_test, y_test):
                    #to make a prediction do: clf.predict([x_testSample])

                    #check if the calculated accuracy is higher than the previously one calculated for each classifier. If so, update the highest accuracy
                    #and print it together with the network hyperparameters
                    #Example: "Highest Perceptron accuracy so far: 0.88, Parameters: learning rate=0.01, shuffle=True"
                    #Example: "Highest MLP accuracy so far: 0.90, Parameters: learning rate=0.02, shuffle=False"
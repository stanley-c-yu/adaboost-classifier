# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:48:15 2019

@author: stany
"""
#----------------------------------------------------------------------------#
#Import Packages
import numpy as np 
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 

from tqdm import tqdm
#----------------------------------------------------------------------------# 
##Import the spambase dataset and adjust as necessary 
spambase = pd.read_csv('spambase.data',header=None)
spambase.rename(columns={0:"word_freq_make", 1:"word_freq_address", 2:"word_freq_all", 3:"word_freq_3d", 4:"word_freq_our", 
                    5:"word_freq_over", 6:"word_freq_remove", 7:"word_freq_internet", 8:"word_freq_order", 9:"word_freq_mail",
                    10:"word_freq_receive", 11:"word_freq_will", 12:"word_freq_people", 13:"word_freq_report", 14:"word_freq_addresses",
                    15:"word_freq_free", 16:"word_freq_business", 17:"word_freq_email", 18:"word_freq_you", 19:"word_freq_credit", 
                    20:"word_freq_your", 21:"word_freq_font", 22:"word_freq_000", 23:"word_freq_money", 24:"word_freq_hp", 
                    25:"word_freq_hpl", 26:"word_freq_george", 27:"word_freq_650", 28:"word_freq_lab", 29:"word_freq_labs", 
                    30:"word_freq_telnet", 31:"word_freq_857", 32:"word_freq_data", 33:"word_freq_415", 34:"word_freq_85", 
                    35:"word_freq_technology", 36:"word_freq_1999", 37:"word_freq_parts", 38:"word_freq_pm", 39:"word_freq_direct", 
                    40:"word_freq_cs", 41:"word_freq_meeting", 42:"word_freq_original", 43:"word_freq_project", 44:"word_freq_re",
                    45:"word_freq_edu", 46:"word_freq_table", 47:"word_freq_conference", 48:"char_freq_;", 49:"char_freq_(", 
                    50:"char_freq_[", 51:"char_freq_!", 52:"char_freq_$", 53:"char_freq_#", 54:"capital_run_length_average", 
                    55:"capital_run_length_longest", 56:"capital_run_length_total", 57:"is_spam"},inplace=True)
#inplace: Makes changes in original Data Frame if True.
#----------------------------------------------------------------------------#
##Split spambase into feature and response sets 
sb_features = spambase.iloc[:, 0:57]
sb_response = spambase.iloc[:, 57]
#SB_response2 = spambase[['is_spam']] this will only select the is_spam column
#spambase.drop(['is_spam'],axis=1)  this will select everything but the is_spam column by dropping 
#----------------------------------------------------------------------------#
##Split sb_features and sb_response into training and testing sets (75% and 25% respectively)
sbf_train, sbf_test, sbr_train, sbr_test = train_test_split(sb_features, sb_response, test_size=0.25, train_size=0.75, 
                                                            random_state = 0, stratify=sb_response)
#----------------------------------------------------------------------------#
##Standardize the dataset by first using preprocessing to compute the mean and standard deviation for future scaling
##Then scale the data sets 
sbf_train = preprocessing.StandardScaler().fit_transform(sbf_train.values)
sbf_test = preprocessing.StandardScaler().fit_transform(sbf_test.values)
print(np.shape(sbf_train))
#----------------------------------------------------------------------------#


class AdaBoost:
    ''' 
    AdaBoost Ensemble Classifier built using SciKit Learn Decision Stumps.
    '''
    
    def __init__(self, X_train, y_train, X_test, y_test, report_iterative=False, iterations=1):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test 
        self.y_test = y_test 
        self.report_iterative = report_iterative 
        self.iterations = iterations
        
        
    def decision_stump(self, X_train,y_train,X_test,y_test,weights,report_iterative=False,random_state=1): 
        ''' 
        Generates decision stumps (one level decision trees)
        '''
        dtc = DecisionTreeClassifier(criterion="entropy",splitter="random",max_depth=1)
        dtc = dtc.fit(X_train,y_train,sample_weight=weights) 
        y_pred = dtc.predict(X_train) 
        acc = metrics.accuracy_score(y_train,y_pred)
        err = 1-acc
        if report_iterative == True: 
            print("Accuracy:",acc)
            print("Error:",err)
        return y_pred, dtc 
    
    def error_compiler(self,y_pred, y_train):
        '''
        Compiles a binary list demarcating errors between y_pred and y_train.  
        
        Errors are listes as ones, and correct predictions are lists as zeroes. 
        
        This makes it easier to apply weights in reweight function.
        '''
        errors = []
        y_train = list(y_train)
        for i in range(0,len(y_pred)):
            if y_pred[i] != y_train[i]:
                errors.append(1) 
            else:
                errors.append(0)
        return errors
                
    def reweight(self,errors,weights,beta):
        ''' 
        Uses the errors list generated in error_compiler to selectively 
        reweight the weights based on whether predictions were correct or 
        incorrect.  
        '''
        for i in range(0,len(weights)):
            if errors[i] == 1: #incorrect 
                weights[i]=weights[i]*np.exp(beta)
            else: #correct 
                weights[i]=weights[i]*np.exp(-1*beta)
        return weights 
    
    def convert_to_minus_plus_ones(self,y_data, check_type_dim = False): 
        '''
        AdaBoost relies on a "sign" function (e.g., positive or negative) to 
        classify values, so we need to convert y_pred and y_train to 1 and -1 
        values from 1 and 0 values.  
        '''
        new_y_data = [] #empty list for transformed y_data array consisting of values in [-1 or 1]
        if check_type_dim == True: #for debugging
            print("y_data un-altered", np.shape(y_data))
            print("type of y_data un-altered", type(y_data))
        y_data = list(y_data) #convert y_data to a list for easy iterating
        if check_type_dim == True: #for debugging
            print("y_data list transformed shape:", np.shape(y_data))
            print("y_data list transformed type:", type(y_data))
        for i in range(0, len(y_data)): #iterate through y_data and convert
            if y_data[i] == 1: 
                new_y_data.append(1)
            else:
                new_y_data.append(-1)
        new_y_data = np.array(new_y_data) #re-convert y_data back into a numpy array
        if check_type_dim == True: #for debugging
            print("Shape of re-converted y_data", np.shape(new_y_data))
            print("type of re-converted y_data",type(new_y_data))
        return new_y_data
        
        
    def calculate_hypothesis(self, betas, stumps, feature_data): 
        '''
        Calculates hypotheses using beta weights applied to each classifier.  
        
        Beta weights weight classifiers by how accurately they are able to make
        predictions.
        
        It then sums all weighted predictions to take a weighted majority vote,
        and returns it.  
        '''
        weighted_predictions = []
        for i in range(0, len(stumps)): 
            dtc = stumps[i]
            beta = betas[i]
            y_pred = dtc.predict(feature_data)
            y_pred = self.convert_to_minus_plus_ones(y_pred) #convert y_pred to an array of values equal to [-1 or 1]
            tmp = beta*y_pred
            weighted_predictions.append(tmp)
        consolidated_weighted_predictions = np.sum(weighted_predictions,0)
        hx = np.sign(consolidated_weighted_predictions)
        return hx
    
    
    
    def adaboost(self):
        ''' 
        Main AdaBoost Classifier Function.  
        '''
        #Create container for T classifiers (T = tot num iterations)
        stumps = []
        #Create container for betas 
        betas = []
        #Initialize weights 
        n = len(self.X_train) 
        weights = np.ones(n)/n #init uniform normalized weights
        print("\nTraining stumps and computing betas...")
        for t in tqdm(range(0,self.iterations)): 
            y_pred, dtc = self.decision_stump(self.X_train,self.y_train,self.X_test,self.y_test,weights,self.report_iterative) #calc pred using decision stump
            y_pred = list(y_pred)
            y_train = list(self.y_train)
            errors = self.error_compiler(y_pred,self.y_train) #compile error list; 1=misclass,;0=correct pred.
            error = (errors*weights).sum() #compute error for "t"th iteration
            if error == 0: 
                return "Error computed as 0.  Log(0) = Undefined, cannot compute."
            beta = 0.5 * np.log((1-error)/error) #calculate beta (aka, alpha depending on notation preferences)
            weights = self.reweight(errors,weights,beta) #re-weight weights based on correctness of predictions
            weights = weights/weights.sum() #normalize weights to be a distribution
            betas.append(beta) #store the "t"th beta for use in final hypothesis func 
            stumps.append(dtc) #store the "t"th stump classifier for use in final hypothesis func
        y_train = self.convert_to_minus_plus_ones(self.y_train) #convert y_train to an array of values in [-1 or 1]
        y_test = self.convert_to_minus_plus_ones(self.y_test) #convert y_test to an array of values in [-1 or 1]
        print("Iterations:",self.iterations)
        print("------------------------------------------------------------")
        hx_train = self.calculate_hypothesis(betas, stumps, self.X_train)
        print("Data for metrics: Training Set")
        print("Unique values in final hypothesis array:",np.unique(hx_train))
        print("Accuracy:",metrics.accuracy_score(y_train,hx_train))
        print("Error:",1-metrics.accuracy_score(y_train,hx_train))
        print("Precision:",metrics.precision_score(y_train,hx_train)) #True Positive/(TP+FP), TP+FP = Tot. Predic. Pos.
        print("Recall:",metrics.recall_score(y_train,hx_train)) #True Positive/(TP+FN), TP+FN= Tot. Actual. Pos.
        print("------------------------------------------------------------")
        hx_test = self.calculate_hypothesis(betas, stumps, self.X_test)
        print("Data for metrics: Testing Set")
        print("Unique values in final hypothesis array:",np.unique(hx_test))
        print("Accuracy:",metrics.accuracy_score(y_test,hx_test))
        print("Error:",1-metrics.accuracy_score(y_test,hx_test))
        print("Precision:",metrics.precision_score(y_test,hx_test))
        print("Recall:",metrics.recall_score(y_test,hx_test))
        return hx_train, hx_test 

#1 Iterations       
adabooster = AdaBoost(sbf_train,sbr_train,sbf_test,sbr_test,report_iterative=False, iterations=1)
adabooster.adaboost()
  
#50 Iterations      
adabooster = AdaBoost(sbf_train,sbr_train,sbf_test,sbr_test,report_iterative=False, iterations=50)
adabooster.adaboost()
        
#100 Iterations 
adabooster = AdaBoost(sbf_train,sbr_train,sbf_test,sbr_test,report_iterative=False, iterations=100)
adabooster.adaboost()  

#150 Iterations 
adabooster = AdaBoost(sbf_train,sbr_train,sbf_test,sbr_test,report_iterative=False, iterations=150)
adabooster.adaboost()
        
        
    
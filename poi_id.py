import sys
import pickle
import numpy as np
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
######################################################################################
### Task 1: Select what features you'll use.
# The script on developing these features is presented in a separate program
#####################################################################################
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# A separate code is provieded for feature selection
features_list= ['poi','from_this_person_to_poi','shared_receipt_with_poi','loan_advances','deferred_income', 'expenses','other', 'director_fees','to_messages', 'deferral_payments','total_payments','restricted_stock_deferred','from_poi_to_this_person','from_messages']
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
###################################################################################
### Task 2: Remove outliers
from pprint import pprint
# To find outliers we are going to use a formula where anything that is two standard deviation 
# above or below the mean is considered as an outlier
data = featureFormat(data_dict, features_list)
outliers=[]
for k in range(len(features_list)-1):
    k=k+1
    mean=np.mean(data[:,k])
    std=np.std(data[:,k])
    outlier_thresold= 2*std
    for key in data_dict:
        val = data_dict[key][features_list[k]]
        if  val == 'NaN':
            continue
        elif val >= mean+outlier_thresold or val <= mean-outlier_thresold:
	    outliers.append((features_list[k],key,int(val)))

#pprint(outliers)    # if you want to see all the outliers 
   
# After careful review of the outliers, the key TOTAL is going to be removed as that is nothing but sum of all the data.
# As with the rest of the names/employees, one data that will be removed is BHATNAGAR SANJAY. It is indicated that he has 
# a restricted stock defferred of 15.5 million. That is way out of the chart when compared to all other employees and I think
# this is just a typo error and so will be removed.

# Removing the outliers

for k in range(len(outliers)):
    if outliers[k][1] == 'ToTAL'or outliers[k][1] == 'BHATNAGAR SANJAY':  
       data_dict.pop(outliers[k][1], 0 ) 
##################################################################################  
### Task 3: Create new feature(s)
# New feaure is created in a separate program. It had little impact on the success rate of the classsfier
# and so, it is not used in the final program.
###################################################################################
### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

##################################################################################
### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# spliting the data set to training and testing data
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

# Classifier algorithms- Naive Bayes, SVM and Decision tree classifier
from sklearn.metrics import precision_recall_fscore_support

# Decision Tree
from sklearn import grid_search
from sklearn.metrics import precision_score

parameters= {             
              'min_samples_split': [2, 4, 6,8,10],
              'criterion': ['entropy','gini']
             }
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier()
clf = grid_search.GridSearchCV(estimator=DT, param_grid=parameters,scoring='f1') #F1 score is used as it incorporates both precision and recall in the formula
clf.fit(features_train, labels_train)
print "Best parameters for Decision Tree classfier", clf.best_params_ 
print "F1 Score for Decision Tree Classifier",clf.best_score_ 

#RandomForestClassifier

parameters= {             
              'n_estimators': [10,50,100],
              'criterion': ['entropy','gini'],
              'min_samples_split': [2, 4, 6,8,10]
             }
from sklearn.ensemble import RandomForestClassifier
RF = RandomForestClassifier()
clf = grid_search.GridSearchCV(estimator=RF,param_grid=parameters,scoring='f1') #F1 score is used as it incorporates both precision and recall in the formula
clf.fit(features_train, labels_train)
print "Best parameters for Random forest classfier", clf.best_params_ 
print "F1 Score for Random Forest Classifier",clf.best_score_ 

#KNeighborsClassifier

parameters= {             
              'n_neighbors': [3,5,10]
           
            }
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
KN = KNeighborsClassifier()
clf = grid_search.GridSearchCV(estimator=KN,param_grid=parameters,scoring='f1') #F1 score is used as it incorporates both precision and recall in the formula
scaler=MinMaxScaler()
rescaled_features=scaler.fit_transform(features)
rescaled_features_train, rescaled_features_test, rescaled_labels_train, rescaled_labels_test = train_test_split(rescaled_features, labels, test_size=0.3, random_state=42)
clf.fit(rescaled_features_train, labels_train)
print "Best parameters for KNeighbors classfier", clf.best_params_ 
print "F1 Score for KNeighbors Classifier",clf.best_score_  

#Naivebayesian

parameters= {    
            }
from sklearn.naive_bayes import GaussianNB
GN = GaussianNB()
clf = grid_search.GridSearchCV(estimator=GN,param_grid=parameters,scoring='f1') #F1 score is used as it incorporates both precision and recall in the formula
clf.fit(features_train, labels_train)
print "Best parameters for Naive Bayes classfier", clf.best_params_ 
print "F1 Score for Naive bayes Classifier",clf.best_score_  

##############################################################################################

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
from sklearn import grid_search

#From the above test,Decision tree with the following parameter has provided us with the best results.
#Thus, it is the choosen classifier for the final nalysis

# Decision tree

clf = DecisionTreeClassifier(min_samples_split=6,criterion= 'entropy')


### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn import grid_search
from sklearn.cross_validation import StratifiedShuffleSplit
labels, features = targetFeatureSplit(data)
# 1000 folds are used to make it as similar as possible to tester.py.
folds = 1000
search_parameters={'min_samples_split': [6],
                   'criterion': ['entropy']
                   }

# We then store the split instance into cv and use it in our GridSearchCV.
clf = DecisionTreeClassifier()
cv = StratifiedShuffleSplit(labels, folds)
clf = grid_search.GridSearchCV(clf, search_parameters, cv = cv, scoring='f1')
clf.fit(features, labels)

print("The best parameters are %s with a score of %0.2f"
      % (clf.best_params_, clf.best_score_))

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
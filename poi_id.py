#!/usr/bin/python

import sys
import pickle
import matplotlib.pyplot
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 
                    'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 
                    'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']

email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 
                'from_this_person_to_poi', 'shared_receipt_with_poi', 'fraction_from_poi', 'fraction_to_poi']

features_list = ['poi'] + financial_features + email_features

# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

def computeFraction( poi_messages, all_messages ):
    """ given a number messages to/from POI (numerator) 
        and number of all messages to/from a person (denominator),
        return the fraction of messages to/from that person
        that are from/to a POI
   """
    fraction = 0.
    if all_messages!="NaN" and poi_messages!="NaN" and all_messages > poi_messages:
        fraction = float(poi_messages)/float(all_messages)

    return fraction

submit_dict = {}
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    fraction_from_poi = computeFraction( from_poi_to_this_person, to_messages )
    data_point["fraction_from_poi"] = fraction_from_poi


    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    fraction_to_poi = computeFraction( from_this_person_to_poi, from_messages )
    data_point["fraction_to_poi"] = fraction_to_poi
    submit_dict[name]={"from_poi_to_this_person":fraction_from_poi,
                       "from_this_person_to_poi":fraction_to_poi}
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


### Task 4: Try a varity of classifiers

### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# Provided to give you a starting point. Try a variety of classifiers.

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit


scaler = MinMaxScaler()
feature_select = SelectKBest(k=5)

#Decision Tree pipeline, classifier, and cross-validation:

#dtree =  tree.DecisionTreeClassifier()
#gs =  Pipeline(steps=[('scaling', scaler), ("select", feature_select), ("dt", dtree)])
#param_grid = {"select": [1,10],
            #'dt__criterion': ["gini", "entropy"],
            #"dt__min_samples_split": [2, 10, 20],
            #"dt__max_depth": [None, 2, 5, 10],
            #"dt__min_samples_leaf": [1, 5, 10],
            #"dt__max_leaf_nodes": [None, 5, 10, 20],
            #}

#sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
#dtcclf = GridSearchCV(gs, param_grid, scoring='recall', cv=sss)
#dtcclf.fit(features, labels)
#clf = dtcclf.best_estimator_

#Gaussian Naive Bayes pipeline, classifier, and cross-validation:

gaussianNB = GaussianNB()
gs = Pipeline(steps=[('scaling',scaler), ('select', feature_select), ('gnb', gaussianNB)])
param_grid = {'select': [1,10]}
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=42)
gnbclf = GridSearchCV(gs, param_grid, scoring='recall', cv=sss)

gnbclf.fit(features, labels)
clf = gnbclf.best_estimator_


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

selected_features = feature_select.fit_transform(features,labels)
feature_scores = ['%.2f' % elem for elem in feature_select.scores_]
features_selected_tuple=[(features_list[i+1], feature_scores[i]) for i in feature_select.get_support(indices=True)]
features_selected_tuple = sorted(features_selected_tuple, key=lambda feature: float(feature[1]) , reverse=True)
print(features_selected_tuple)

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
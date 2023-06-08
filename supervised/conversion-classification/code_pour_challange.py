# Concatenate our train and test set to train your best classifier on all data with labels
X = np.append(X_train,X_test,axis=0)
Y = np.append(Y_train,Y_test)
classifier.fit(X,Y)
# Read data without labels
data_without_labels = pd.read_csv('conversion_data_test.csv')
print('Prediction set (without labels) :', data_without_labels.shape)

# Warning : check consistency of features_list (must be the same than the features
# used by your best classifier)
features_list = ['total_pages_visited', 'country', 'age', 'new_user', 'source']
X_without_labels = data_without_labels.loc[:, features_list]

# On va doper total_pages_visited
X_without_labels.loc[:, 'total_pages_visited_sq'] = (X_without_labels.loc[:, 'total_pages_visited']*2)
display(X_without_labels.head())
# Convert pandas DataFrames to numpy arrays before using scikit-learn
print("Convert pandas DataFrames to numpy arrays...")
X_without_labels = X_without_labels#.values # TEST COnformité
print("...Done")

#print(X_without_labels[0:5,:])
# Make predictions and dump to file
# WARNING : MAKE SURE THE FILE IS A CSV WITH ONE COLUMN NAMED 'converted' AND NO INDEX !
# WARNING : FILE NAME MUST HAVE FORMAT 'conversion_data_testpredictions[name].csv'
# where [name] is the name of your team/model separated by a '-'
# For example : [name] = AURELIE-model1
data = {
    'converted': classifier.predict(X_without_labels)
}

Y_predictions = pd.DataFrame(columns=['converted'],data=data)
Y_predictions.to_csv('conversion_data_test_predictions_FREDERIC-4.csv', index=False)
conv = pd.read_csv('conversion_data_test_predictions_FREDERIC-4.csv')
conv.describe()

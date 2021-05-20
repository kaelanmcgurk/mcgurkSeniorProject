#%%
#import proper files here
#from getLuminaryDB import callLuminary, makeQueryDataFarme
import pandas as pd
import altair as alt
import numpy as np
from altair_saver import save
from matplotlib import pyplot as plt
#from repTouchPoints import stc
alt.data_transformers.disable_max_rows()

# Call the database
#luminary = callLuminary()

# %%
##########################################
# Make MySQL queries here
#
##########################################
leadsQ = '''
SELECT * FROM leads;
'''

cleanLeadsQ = '''
SELECT lead_source, 
    owner, 
    YEAR(date_created) AS yearCreated, 
    MONTH(date_created) AS monthCreated, 
    DAY(date_created) AS dayCreated, 
    DAYNAME(date_created) AS dayName,
    city,
    state,
    zip_code,
    electric_company,
    square_footage,
    status AS y
FROM leads
WHERE owner != 6585993;
'''
statusQ = '''
SELECT status 
FROM leads
WHERE owner != 6585993
'''
#%%
#########################################
# Load in the dataframes
#
#########################################
#leads = makeQueryDataFarme(luminary, leadsQ)
#leadsCleaned = makeQueryDataFarme(luminary, cleanLeadsQ)
#statusY = makeQueryDataFarme(luminary, statusQ)

FebCleanedLeads = pd.read_csv('dataSets/cleanedLeads_2_11_21.csv')
FebStatusHistory = pd.read_csv('dataSets/status_history_2_11_21.csv')
leadInfoML = pd.read_csv('dataSets/leadInfoML_2_11_21.csv')

leadsAppt = pd.read_csv('dataSets/leadsAppointments.csv')
leadsStatusHist = pd.read_csv('dataSets/leadsStatusHistory.csv')
leadsCallLogs = pd.read_csv('dataSets/leadsCallLogs.csv')
leadsStatus = pd.read_csv('dataSets/leadsStatusCurrent.csv')

stc = pd.read_csv('dataSets/stc.csv')
stc = stc.drop(['Unnamed: 0', 'callStarted'], axis = 1)

#newLeadInfo = leadInfoML.rename(columns = {'lead_id':'leadId'})

#%%
##########################################
# Wrangle the data
#
##########################################

FebCleanedLeads['zip_code'] = FebCleanedLeads['zip_code'].str.split('-').str.get(0).str.strip()

FebCleanedLeads['state'] = FebCleanedLeads['state'].str.strip()

FebCleanedLeads['electric_company'] = FebCleanedLeads['electric_company'].str.lower()
FebCleanedLeads['electric_company'] = FebCleanedLeads['electric_company'].str.strip()


FebCleanedLeads = FebCleanedLeads.assign(
    zip_code = lambda x: x.zip_code
        .replace('poasoejfpo', '0')
        .replace('na','0')
        .replace('????','0')
        .replace('',np.nan)
        .replace(' ', np.nan)
        .replace('none', np.nan)
        .replace('NA1234', np.nan)
        .replace('n', np.nan)
        .replace('lattitude', np.nan)
        .replace('hotmail.co', np.nan)
        .replace('here', np.nan)
        .replace('Elmore Cit', np.nan)
        .replace('AR', np.nan)
        .replace('A', np.nan)
        .replace('74118821 e', np.nan)
        .replace('72641 35.9', np.nan)
        .replace('72712yu', np.nan)
        .fillna('0'),
    state = lambda x: x.state
        .replace('VA', 'Virginia')
        .replace('TN', 'Tennessee')
        .replace('OK', 'Oklahoma')
        .replace('NJ', 'New Jersey')
        .replace('MO', 'Missouri')
        .replace('CO', 'Colorado')
        .replace('AR', 'Arkansas')
        .replace('TX', 'Texas')
        .replace('Select a state', np.nan)
        .replace('KS', 'Kansas')
        .replace('IL', 'Illinois')
        .replace('MD', 'Maryland')
        .replace('CA', 'California')
        .replace('M', np.nan)
        .replace('FL', 'Florida')
        .fillna('noState'),
    electric_company = lambda x: x.electric_company
        .fillna('noCompany'),
    square_footage = lambda x: x.square_footage
        .fillna(0)

)


#%%
zipCodeDict = {'zip_code': 'int64'}
FebCleanedLeads = FebCleanedLeads.astype(zipCodeDict)
FebCleanedLeads = FebCleanedLeads.query('zip_code < 100000 & zip_code > 9999')

# This is how to remove the NAs from a column and keep
#  it in the data frame
# combinedLeadProducts = combinedLeadProducts.dropna(subset = ['state'])

#%%
##########################
# Work with the new machine learning dataset here
#
##########################

leadInfoML = leadInfoML.dropna()

leadInfoML = leadInfoML.assign(
    zip_code = lambda x: x.zip_code
        .replace('poasoejfpo', '0')
        .replace('na','0')
        .replace('????','0')
        .replace('',np.nan)
        .replace(' ', np.nan)
        .replace('none', np.nan)
        .replace('NA1234', np.nan)
        .replace('n', np.nan)
        .replace('lattitude', np.nan)
        .replace('hotmail.co', np.nan)
        .replace('here', np.nan)
        .replace('Elmore Cit', np.nan)
        .replace('AR', np.nan)
        .replace('A', np.nan)
        .replace('74118821 e', np.nan)
        .replace('72641 35.9', np.nan)
        .replace('72712yu', np.nan)
        .fillna('0'),
    state = lambda x: x.state
        .replace('VA', 'Virginia')
        .replace('TN', 'Tennessee')
        .replace('OK', 'Oklahoma')
        .replace('NJ', 'New Jersey')
        .replace('MO', 'Missouri')
        .replace('CO', 'Colorado')
        .replace('AR', 'Arkansas')
        .replace('TX', 'Texas')
        .replace('Select a state', np.nan)
        .replace('KS', 'Kansas')
        .replace('IL', 'Illinois')
        .replace('MD', 'Maryland')
        .replace('CA', 'California')
        .replace('M', np.nan)
        .replace('FL', 'Florida')
        .fillna('noState'),
    electric_company = lambda x: x.electric_company
        .fillna('noCompany')
)

#%%
leadInfoML['state'] = leadInfoML['state'].str.strip()
leadInfoML['state'] = leadInfoML['state'].str.lower()

leadInfoML['electric_company'] = leadInfoML['electric_company'].str.lower()
leadInfoML['electric_company'] = leadInfoML['electric_company'].str.strip()

#%%

leadInfoML['isCust'] = [1 if x == 18 else 0 for x in leadInfoML['status']] 

#%%
combined = leadInfoML.merge(stc, left_on = 'lead_id', right_on = 'leadId')
combined = combined.query('state != "colorado" and state != "kansas" and state != "virginia" ').drop(['leadId'], axis = 1)

#%%
# Merge the new datasets so that they look better
leadsCombined = leadsAppt.merge(leadsCallLogs)
leadsData = leadsCombined.merge(leadsStatusHist)
leadsDat = leadsData.merge(stc, left_on = 'lead_id', right_on = 'leadId')
leadsFinal = leadsDat.merge(leadsStatus)
leadsFinal = leadsFinal.drop(['leadId'], axis = 1)


#%%
########################
# Clean up the leadsData 
#   just a bit
########################

#repLeadSource.isna().sum() 
#repLeadSource['lead_source'].value_counts()

leadsFinal = leadsFinal.assign(
    state = lambda x: x.state
        .replace('VA', 'Virginia')
        .replace('TN', 'Tennessee')
        .replace('OK', 'Oklahoma')
        .replace('NJ', 'New Jersey')
        .replace('MO', 'Missouri')
        .replace('CO', 'Colorado')
        .replace('AR', 'Arkansas')
        .replace('TX', 'Texas')
        .replace('Select a state', np.nan)
        .replace('KS', 'Kansas')
        .replace('IL', 'Illinois')
        .replace('MD', 'Maryland')
        .replace('CA', 'California')
        .replace('M', np.nan)
        .replace('FL', 'Florida')
        .fillna('noState') 
)

leadsFinal = leadsFinal.drop(['state'], axis = 1)
leadsFinal['isCust'] = [1 if x == 18 else 0 for x in leadsFinal['status']] 

#%%
#leadsFinal.to_csv(r'dataSets/leadsFinal.csv')

#%%
cols = ['dayName','state','electric_company','agentId']
leadInfoMLDum = pd.get_dummies(leadInfoML, columns = cols)

colsTwo = ['dayName','state','agentId']
combinedDum = pd.get_dummies(combined, columns = colsTwo)







#%%
# Random oversample the data to get an equal amount of 
#  customers to nonCustomers
from imblearn.over_sampling import RandomOverSampler
ro = RandomOverSampler()




#%%
##############################################
# Start the machine learning process here
#
##############################################

import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import ensemble, tree
from sklearn.metrics import accuracy_score, mean_squared_error



################################
# ShineCLF Ab
#
# The Ab model
################################
#%%

A = leadsFinal.drop(['status', 'year', 'numberOfAppointments', 'month', 'isCust'],axis = 1)
b = leadsFinal.filter(items = ['isCust'])

A_OS, b_OS = ro.fit_resample(A, b)

A_OS = pd.DataFrame(A_OS)
b_OS = pd.DataFrame(b_OS)

Atrain1, Atest1, bTrain, bTest = train_test_split(
    A_OS, 
    b_OS, 
    test_size=0.2
)

Atrain = Atrain1.drop(['lead_id'], axis = 1)
Atest = Atest1.drop(['lead_id'], axis = 1)


ShineTreeClfAB = tree.DecisionTreeClassifier(ccp_alpha = 0.0003)
ShineTreeClfAB.fit(Atrain, bTrain)

bPredict = ShineTreeClfAB.predict(Atest)

featureDat = pd.DataFrame({
    "values":ShineTreeClfAB.feature_importances_,
    "features":Atrain.columns})
featureChart = (alt.Chart(featureDat.query('values > .017'))
    .encode(
     alt.X("values"),
     alt.Y("features", sort = "-x"))
    .mark_bar()
)
featureChart
metrics.plot_confusion_matrix(ShineTreeClfAB, Atest,bTest)
print(' ')
print(metrics.classification_report(bTest, bPredict))
metrics.accuracy_score(bTest, bPredict)
featureChart

#%%
# Print out the predicted probabilities for my model
bProba = ShineTreeClfAB.predict_proba(Atest)
bProba = pd.DataFrame(bProba)

notCust = pd.DataFrame(bProba[0])
cust = pd.DataFrame(bProba[1])
#%%
leadGuess = Atest1['lead_id']
leadGuess = pd.DataFrame(leadGuess).reset_index(drop=True)

#%%
leadGuess['notCust'] = notCust
leadGuess['Cust'] = cust

#leadGuess = leadGuess.fillna("0")

#%%
leadGuess.to_csv(r'dataSets/PredictionForLeads.csv')

#%%
# Visualize the actual tree!

target = np.array(['notCust','Cust'])
features = ['numberOfCalls', 'callTimeMinute', 'numberofStatuses', 'hoursSinceCall']



fig = plt.figure(figsize=(50, 40))
_ = tree.plot_tree(ShineTreeClfAB, 
                   feature_names=features,  
                   class_names=target,
                   filled=True)

#fig.savefig("Ab_decistion_tree.png")





#%%
#################################
# Attempt to postprune the 
#   tree so that it is not 
#   overfitting
#
#  This worked... OK?
#   I mean, it helped stop overfitting,
#   but even my overfit model performed
#   over 90% recall when working
#   with test data.    
################################

# View the path post pruning would take
path = ShineTreeClfAB.cost_complexity_pruning_path(Atrain, bTrain)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")

#%%

# This shows how the effective alpha moves as we prune more
#    and more nodes

clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(Atrain, bTrain)
    clfs.append(clf)
print("Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
      clfs[-1].tree_.node_count, ccp_alphas[-1]))

#%%
# Watch how the number of nodes and the actual depth of the 
#   tree change as alpha increases    

clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker='o', drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("number of nodes")
ax[0].set_title("Number of nodes vs alpha")
ax[1].plot(ccp_alphas, depth, marker='o', drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("depth of tree")
ax[1].set_title("Depth vs alpha")
fig.tight_layout()

#%%

#Finally, we look at the training and testing data to see the
#   effect post pruning has on accuracy
    
train_scores = [clf.score(Atrain, bTrain) for clf in clfs]
test_scores = [clf.score(Atest, bTest) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("accuracy")
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(ccp_alphas, train_scores, marker='o', label="train",
        drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker='o', label="test",
        drawstyle="steps-post")
ax.legend()
plt.show()

#%%










































#%%
# Below are all the different models I tried, got close to, but
#  ultimately failed
'''


X = leadInfoMLDum.drop(['isCust','status','lead_id','zip_code'], axis = 1)
y = leadInfoMLDum.filter(items=['isCust'])


#%%
X_OS, y_OS = ro.fit_resample(X, y)

X_OS = pd.DataFrame(X_OS)
y_OS = pd.DataFrame(y_OS)


#%%
Xtrain, Xtest, yTrain, yTest = train_test_split(
    X_OS, 
    y_OS, 
    test_size=0.2
)


# %%
###################################
# Train the model:
#   Tree Model
# 
###################################

ShineTreeClf = tree.DecisionTreeClassifier()
ShineTreeClf.fit(Xtrain, yTrain)


#%%
##################################
# Test the model:
#   Tree Model
#
#################################
yPredict = ShineTreeClf.predict(Xtest)

featureDat = pd.DataFrame({
    "values":ShineTreeClf.feature_importances_,
    "features":Xtrain.columns})
featureChart = (alt.Chart(featureDat.query('values > .017'))
    .encode(
     alt.X("values"),
     alt.Y("features", sort = "-x"))
    .mark_bar()
)
featureChart
metrics.plot_confusion_matrix(ShineTreeClf, Xtest,yTest)
print(' ')
print(metrics.classification_report(yTest, yPredict))
metrics.accuracy_score(yTest, yPredict)
featureChart


#%%
###################################
# Start the process for the 
#   neural network
###################################
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, losses

# Build the architechture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 223, activation = 'sigmoid'),
    tf.keras.layers.Dense(units = 200, activation = 'relu'),
    tf.keras.layers.Dense(units = 1, activation= 'relu')  
])

# Compile the model
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))

# Fit the model
model.fit(Xtrain, yTrain, epochs=15, validation_split=.20, verbose = 2)

# Find the accuracy
test_loss, test_acc = model.evaluate(Xtest,  yTest, verbose=2)
print('\nTest accuracy:', test_acc)

#%%
# Make the predictions 
probability_model = tf.keras.Sequential([model, 
    tf.keras.layers.Softmax()])

# Print an array of probabilities that a lead will be a 
#  customer
predictions = probability_model.predict(Xtest)

#%%
#################
# Run the model on
#   the new data table
#################
# Build the architechture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units = 7, activation = 'sigmoid'),
    tf.keras.layers.Dense(units = 20, activation = 'relu'),
    tf.keras.layers.Dense(units = 1, activation= 'relu')  
])

# Compile the model
model.compile(loss=losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=tf.metrics.BinaryAccuracy(threshold=0.0))


model.fit(Atrain, bTrain, epochs=150, validation_split=.20, verbose = 2)

# Find the accuracy
test_loss, test_acc = model.evaluate(Atest,  bTest, verbose=2)
print('\nTest accuracy:', test_acc)

#%%
# Make the predictions 
probability_model = tf.keras.Sequential([model, 
    tf.keras.layers.Softmax()])

# Print an array of probabilities that a lead will be a 
#  customer
predictions = probability_model.predict(Atest)
#%%
#leadInfoML.to_csv(r'leadInfoML.csv')
# %%
#############################################
# Completely different kind of Neural Network
#   I am going off of the Structured Data
#   tutorial on TensorFlow
#
#############################################

from tensorflow import feature_column
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# From the top, just start with the data oversample because it smol

A = leadsFinal.drop(['status', 'isCust'],axis = 1)
b = leadsFinal.filter(items = ['isCust'])

A_OS, b_OS = ro.fit_resample(A, b)

A_OS = pd.DataFrame(A_OS)
b_OS = pd.DataFrame(b_OS)

A_OS['isCust'] = b_OS

#%%
# Just made the new dataset into a .CSV file
#A_OS.to_csv(r'newLeadNumbers.csv')

#%%
#######################################
# Split data into test/train/validation 
#######################################

train, test = train_test_split(A_OS, test_size=0.2)
train, val = train_test_split(train, test_size=0.2)

#%%
##########################
# Use tf.data to use feature columns
#   in this network
##########################

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('isCust')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

batch_size = 32 
train_ds = df_to_dataset(train, batch_size=batch_size)
val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

# Look at each feature batch and the list of targets (isCust)
for feature_batch, label_batch in train_ds.take(1):
  print('Every feature:', list(feature_batch.keys()))
  print()
  print('A batch of callTimes:', feature_batch['callTimeMinute'])
  print()
  print('A batch of targets:', label_batch )

#%%
# Make the columns into numeric feature columns 

feature_columns = []
listOfColumns = ['lead_id', 'month', 'year', 'numberOfAppointments', 'numberOfCalls', 'callTimeMinute', 'numberofStatuses', 'hoursSinceCall']

# numeric cols
for header in listOfColumns:
  feature_columns.append(feature_column.numeric_column(header))


# Make those feature_columns into keras readable layers
feature_layer = tf.keras.layers.DenseFeatures(feature_columns)

#%%
# Now it is time to run the model!!

model = tf.keras.Sequential([
  feature_layer,
  layers.Dense(128, activation='relu'),
  layers.Dense(32, activation = 'linear'),
  layers.Dropout(.1),
  layers.Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(train_ds,
          validation_data=val_ds,
          epochs=200)


print()
loss, accuracy = model.evaluate(test_ds)
print()
print("Accuracy", accuracy)


#%%
predictedLeads = model.predict_classes(test_ds, batch_size = batch_size)

#%%

#################################
# Try another model with the agent_id
#  not in there
#
##################################

cols2 = ['dayName','state','zip_code','electric_company']
leadInfoMLDum2 = pd.get_dummies(leadInfoML, columns = cols2)


X = leadInfoMLDum2.drop(['isCust','status','city','agentName','lead_id','agentId'], axis = 1)
y = leadInfoMLDum2.filter(items=['isCust'])

#%%
X_OS, y_OS = ro.fit_resample(X, y)

X_OS = pd.DataFrame(X_OS)
y_OS = pd.DataFrame(y_OS)

#%%
Xtrain, Xtest, yTrain, yTest = train_test_split(
    X_OS, 
    y_OS, 
    test_size=0.2
)


# %%
###################################
# Train the model:
#   Tree Model
# 
###################################

treeClf = tree.DecisionTreeClassifier()
treeClf.fit(Xtrain, yTrain)

#%%
##################################
# Test the model:
#   Tree Model
#
#################################
yPredict = treeClf.predict(Xtest)

featureDat = pd.DataFrame({
    "values":treeClf.feature_importances_,
    "features":Xtrain.columns})
featureChart = (alt.Chart(featureDat.query('values > .015'))
    .encode(
     alt.X("values"),
     alt.Y("features", sort = "-x"))
    .mark_bar()
)
featureChart
metrics.plot_confusion_matrix(treeClf, Xtest,yTest)
print(' ')
print(metrics.classification_report(yTest, yPredict))
metrics.accuracy_score(yTest, yPredict)
featureChart








# %%
###################################
# Maybe try an XGboost Classifier
#  
##################################

#cols2 = ['dayName','state']
#leadInfoMLDum2 = pd.get_dummies(leadInfoML, columns = cols2)


X = leadInfoML.drop(['isCust','status','lead_id'], axis = 1)
y = leadInfoML.filter(items=['isCust'])

#%%
X_OS, y_OS = ro.fit_resample(X, y)

X_OS = pd.DataFrame(X_OS)
y_OS = pd.DataFrame(y_OS)

#%%
Xtrain, Xtest, yTrain, yTest = train_test_split(
    X_OS, 
    y_OS, 
    test_size=0.2
)

XGBclf = XGBClassifier()
XGBclf.fit(Xtrain, yTrain)

# %%
xgbPredict = XGBclf.predict(Xtest)

featureDat = pd.DataFrame({
    "values":XGBclf.feature_importances_,
    "features":Xtrain.columns})
featureChart = (alt.Chart(featureDat.query('values > .025'))
    .encode(
     alt.X("values"),
     alt.Y("features", sort = "-x"))
    .mark_bar()
)
featureChart
#%%
metrics.plot_confusion_matrix(XGBclf, Xtest,yTest)
print(' ')
print(metrics.classification_report(yTest, xgbPredict))
metrics.accuracy_score(yTest, xgbPredict)
featureChart
'''
# %%

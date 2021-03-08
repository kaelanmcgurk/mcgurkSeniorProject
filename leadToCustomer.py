#%%
#import proper files here
#from getLuminaryDB import callLuminary, makeQueryDataFarme
import pandas as pd
import altair as alt
import numpy as np
from altair_saver import save
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
stc = pd.read_csv('dataSets/stc.csv')

#%%
combined = leadInfoML.merge(stc, left_on = 'lead_id', right_on = 'leadId')

#%%
##########################################
# Wrangle the data
#
##########################################

FebCleanedLeads['zip_code'] = FebCleanedLeads['zip_code'].str.split('-').str.get(0).str.strip()

FebCleanedLeads['city'] = FebCleanedLeads['city'].str.lower()
FebCleanedLeads['city'] = FebCleanedLeads['city'].str.strip()

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
    city = lambda x: x.city
        .replace('24 rannoch dr', np.nan)
        .replace('35147 county scarnegie', np.nan)
        .replace('call 10-30', np.nan)
        .replace('call 10/26', np.nan)
        .replace('call after 7', np.nan)
        .replace('call after work', np.nan)
        .replace('call at the end of the month', np.nan)
        .replace('call early afterno', np.nan)
        .replace('needs time with girl', np.nan)
        .replace('out of service area call her back when in st james mo', np.nan)
        .fillna('noCity'),
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
    city = lambda x: x.city
        .replace('24 rannoch dr', np.nan)
        .replace('35147 county scarnegie', np.nan)
        .replace('call 10-30', np.nan)
        .replace('call 10/26', np.nan)
        .replace('call after 7', np.nan)
        .replace('call after work', np.nan)
        .replace('call at the end of the month', np.nan)
        .replace('call early afterno', np.nan)
        .replace('needs time with girl', np.nan)
        .replace('out of service area call her back when in st james mo', np.nan)
        .fillna('noCity'),
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
leadInfoML['city'] = leadInfoML['city'].str.lower()
leadInfoML['city'] = leadInfoML['city'].str.strip()

leadInfoML['state'] = leadInfoML['state'].str.strip()
leadInfoML['state'] = leadInfoML['state'].str.lower()

leadInfoML['electric_company'] = leadInfoML['electric_company'].str.lower()
leadInfoML['electric_company'] = leadInfoML['electric_company'].str.strip()

#%%

leadInfoML['isCust'] = [1 if x == 18 else 0 for x in leadInfoML['status']] 

#%%
cols = ['dayName','state','electric_company','agentId']
leadInfoMLDum = pd.get_dummies(leadInfoML, columns = cols)

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


X = leadInfoMLDum.drop(['isCust','status','agentName','lead_id','city', 'zip_code'], axis = 1)
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



#%%
#leadInfoML.to_csv(r'leadInfoML.csv')
# %%
'''
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
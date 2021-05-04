#%%
import pandas as pd
import altair as alt
import numpy as np
alt.data_transformers.disable_max_rows()

#%%
########################
# SQL queries to get 
#  the data
########################

statusHistory = '''
SELECT
		 l.lead_id AS lead_id,
		 COUNT(sh.status) AS numberOfStatuses
FROM leads l
JOIN status_history sh ON l.lead_id = sh.lead
WHERE sh.agent  != 6585993 AND l.date_created > '2021/03/05' AND l.status != 18
GROUP BY  l.lead_id;
'''

callLogs = '''
SELECT
		 l.lead_id AS lead_id,
		 COUNT(cl.call_id) AS numberOfCalls,
		 ROUND(SUM(TIMESTAMPDIFF(SECOND, cl.call_started, cl.call_completed)) / 60, 2) AS callTimeMinute
FROM  leads l
JOIN call_logs cl ON l.lead_id = cl.lead
WHERE cl."user"  != 6585993 AND l.date_created > '2021/03/05' AND l.status != 18
GROUP BY  l.lead_id;
'''

stc = '''
SELECT
    l.lead_id AS leadId,
    l.date_created as dateCreated,
    cl.call_started AS callStarted,
    TIMESTAMPDIFF(HOUR, l.date_created, cl.call_started) AS hoursSinceCall
FROM leads l
JOIN call_logs cl ON l.lead_id = cl.lead
JOIN lead_sources ls ON ls.source_id = l.lead_source
WHERE cl.user  != 6585993 AND l.date_created > '2021/03/05' AND l.status != 18 AND cl.call_started > l.date_created AND LOCATE("Scheduled", ls.source_name) = 0;
'''
#%%
########################
# Import the new leads'
#  data
########################

stc = pd.read_csv('testingData/stcNew.csv')
callLogs = pd.read_csv('testingData/callLogsNew.csv')
statusHistory = pd.read_csv('testingData/statusHistoryNew.csv')

# %%
###############################
# Wrangle the data if needed
#
#
###############################
stcData = stc.dropna(subset = ['leadId'])
stcData = stc.dropna(subset = ['dateCreated'])

stcData['dateCreated'] = pd.to_datetime(stcData['dateCreated'])
stcData['callStarted'] = pd.to_datetime(stcData['callStarted'])

# %%
speedToContact = stcData.groupby(['leadId']).agg({'hoursSinceCall':np.min, 'callStarted':np.min}).reset_index()
stc = speedToContact.loc[speedToContact.groupby('leadId')['hoursSinceCall'].idxmin()]
# %%

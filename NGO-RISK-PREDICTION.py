
# coding: utf-8

# In[41]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


# In[43]:


train_data = pd.read_csv("NGOTransactionReportTrain.csv")


# In[5]:


test_data = pd.read_csv("NGOTransactionReportTest.csv")


# In[6]:


encoder = preprocessing.LabelEncoder()


# In[7]:


req = encoder.fit_transform(train_data['PREDICTION'])


# In[8]:


earr = encoder.classes_


# In[9]:


train_data['PREDICTION'] = req


# In[10]:


train_y_labels = train_data['PREDICTION']


# In[11]:


train_x_data = train_data.drop(['PREDICTION'],axis=1)


# In[12]:


X_train = train_x_data


# In[13]:


y_train = train_y_labels


# In[14]:


df = pd.DataFrame(test_data)


# In[15]:


df = df[(df['TXN_TYPE']=='SELF_ISSUANCE')|(df['TXN_TYPE']=='SETTLEMENT')].reset_index()


# In[16]:


df['SUM'] = df.groupby(['ORGANIZATION','TXN_TYPE'])['AMOUNT'].transform('sum')


# In[17]:


df = df.drop_duplicates(subset=['ORGANIZATION','TXN_TYPE'])


# In[18]:


df['SELF_ISSUANCE'] = df[(df['TXN_TYPE']=='SELF_ISSUANCE')].groupby('ORGANIZATION')['SUM'].transform('sum')


# In[19]:


df['SETTLEMENT'] = df[(df['TXN_TYPE']=='SETTLEMENT')].groupby('ORGANIZATION')['SUM'].transform('sum')


# In[20]:


df = df.fillna(0).reset_index()


# In[21]:


df = df.groupby('ORGANIZATION').sum()


# In[22]:


df['TOTAL']=df['SELF_ISSUANCE']-df['SETTLEMENT']


# In[23]:


df = df.reset_index()


# In[24]:


dframe = df[['ORGANIZATION','TOTAL']]


# In[25]:


X_test = dframe


# In[26]:


org = tf.feature_column.categorical_column_with_hash_bucket("ORGANIZATION", hash_bucket_size=52)
amount = tf.feature_column.numeric_column("TOTAL")


# In[27]:


feat_cols = [org,amount]


# In[28]:


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)


# In[29]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes = 10)


# In[30]:


model.train(input_fn=input_func,steps=5000)


# In[31]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_train, y = y_train,
                                                     batch_size = 10, num_epochs=1,
                                                     shuffle=False)


# In[32]:


results = model.evaluate(eval_input_func)


# In[33]:


results


# In[34]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[35]:


predictions = list(model.predict(input_fn=pred_fn))


# In[36]:


predictions


# In[37]:


final_preds = []
print('\n\t\t!!!NGO RISK PREDICTION -- REPORT!!!\n')
for index,pred in zip(range(0,len(X_test)),predictions):
    row = {}
    row['AMOUNT'] = X_test['TOTAL'][index]
    row['ORGANIZATION'] = X_test['ORGANIZATION'][index]    
    row['PREDICTION'] = earr[pred['class_ids'][0]]
    final_preds.append(row)
    print(row)
dataframe = pd.DataFrame.from_dict(final_preds)
dataframe.to_csv('NGO-RISK-PREDICTION.csv', index = False)


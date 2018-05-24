
# coding: utf-8

# In[58]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import tensorflow as tf


# In[87]:


ngo_data = pd.read_csv("NGOTransactionReport.csv")


# In[88]:


ngo_data.columns


# In[89]:


encoder = preprocessing.LabelEncoder()


# In[92]:


req = encoder.fit_transform(ngo_data['REQUIRED'])


# In[93]:


earr = encoder.classes_


# In[94]:


ngo_data['REQUIRED'] = req


# In[95]:


y_labels = ngo_data['REQUIRED']


# In[96]:


x_data = ngo_data.drop(['REQUIRED','TIMESTAMP'],axis=1)


# In[97]:


X_train, X_test, y_train, y_test = train_test_split(x_data,y_labels,test_size=0.3,random_state=101)


# In[98]:


country = tf.feature_column.categorical_column_with_hash_bucket("COUNTRY", hash_bucket_size=52)
city = tf.feature_column.categorical_column_with_hash_bucket("CITY", hash_bucket_size=52)
org = tf.feature_column.categorical_column_with_hash_bucket("ORGANIZATION", hash_bucket_size=52)
donor = tf.feature_column.categorical_column_with_hash_bucket("DONOR", hash_bucket_size=52)
txn = tf.feature_column.categorical_column_with_hash_bucket("TXN_TYPE", hash_bucket_size=52)
amount = tf.feature_column.numeric_column("AMOUNT")


# In[99]:


feat_cols = [country,city,org,donor,txn,amount]


# In[100]:


input_func=tf.estimator.inputs.pandas_input_fn(x=X_train,y=y_train,batch_size=100,num_epochs=None,shuffle=True)


# In[101]:


model = tf.estimator.LinearClassifier(feature_columns=feat_cols,n_classes = 10)


# In[102]:


model.train(input_fn=input_func,steps=1000)


# In[103]:


eval_input_func = tf.estimator.inputs.pandas_input_fn(x = X_test, y = y_test,
                                                     batch_size = 10, num_epochs=1,
                                                     shuffle=False)


# In[104]:


results = model.evaluate(eval_input_func)


# In[105]:


results


# In[106]:


pred_fn = tf.estimator.inputs.pandas_input_fn(x=X_test,batch_size=len(X_test),shuffle=False)


# In[107]:


predictions = list(model.predict(input_fn=pred_fn))


# In[108]:


predictions


# In[109]:


final_preds = []
for pred in predictions:
    final_preds.append(earr[pred['class_ids'][0]])


# In[112]:


final_preds


# In[119]:


ytst = encoder.inverse_transform(y_test)


# In[120]:


ytst


# In[122]:


from sklearn.metrics import classification_report


# In[179]:


print(classification_report(ytst,final_preds))


# In[186]:


import matplotlib.pyplot as plt
def plot_classification_report(cr, title='Classification report ', with_avg_total=False, cmap=plt.cm.Blues):

    lines = cr.split('\n')

    classes = []
    plotMat = []
    for line in lines[2 : (len(lines) - 3)]:
        #print(line)
        t = line.split()
        # print(t)
        classes.append(t[0])
        v = [float(x) for x in t[1: len(t) - 1]]
        print(v)
        plotMat.append(v)

    if with_avg_total:
        aveTotal = lines[len(lines) - 1].split()
        classes.append('avg/total')
        vAveTotal = [float(x) for x in t[1:len(aveTotal) - 1]]
        plotMat.append(vAveTotal)


    plt.imshow(plotMat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(3)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.tight_layout()
    plt.ylabel('Classes')
    plt.xlabel('Measures')


# In[187]:


classReport = classification_report(ytst,final_preds)


# In[188]:


plot_classification_report(classReport)


# In[137]:


#import collections


# In[138]:


#coll = collections.Counter(ytst)


# In[178]:


#print(classification_report(ytst,final_preds))


# In[167]:


#prediction_df = pd.DataFrame(ytst)
#ngo_data1 = pd.read_csv("NGOTransactionReport.csv")


# In[169]:


#req1 = ngo_data1['REQUIRED']


# In[148]:


#prediction_df.to_csv('prediction.csv')


# In[170]:


#expected = req1


# In[171]:


#predicted = ytst


# In[172]:


#y_actu = pd.Series(expected, name='Actual')


# In[173]:


#y_pred = pd.Series(predicted, name='Predicted')


# In[176]:


#prediction_df = pd.DataFrame(expected,predicted)


# In[177]:


#prediction_df


# In[189]:


def classifaction_report_csv(report):
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report.csv', index = False)


# In[190]:


classifaction_report_csv(classReport)


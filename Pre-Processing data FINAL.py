#!/usr/bin/env python
# coding: utf-8

# In[134]:


import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np
import seaborn as sns


# In[135]:


data = pd.read_csv('./orginized_data.csv')


# In[138]:


print(data.shape)


# In[142]:


## https://medium.com/analytics-vidhya/building-an-intrusion-detection-model-using-kdd-cup99-dataset-fb4cba4189ed
plt.figure(figsize=(10,10))
class_distribution = data['target'].value_counts()
class_distribution.plot(kind='pie')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
# ref: arg sort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# -(train_class_distribution.values): the minus sign will give us in decreasing order
sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1,':', class_distribution.values[i], '(', np.round((class_distribution.values[i]/data.shape[0]*100), 3), '%)')


# In[137]:


data.head(10)


# In[126]:


data['flag']


# In[127]:


data.iloc[494020] ## Get a specific row in a given Pandas DataFrame


# In[128]:


# describe dataset
data.describe()


# In[129]:


data['target'].unique() ## unique values in a particular column


# In[130]:


data['target'].value_counts()


# # Code Cleaning

# In[144]:


data.drop_duplicates(keep='first', inplace = True)


# In[145]:


data.shape


# In[146]:


data.isnull().sum() ## check if there is any missin value in the dataset


# In[147]:


data.notnull()


# In[148]:


## https://medium.com/analytics-vidhya/building-an-intrusion-detection-model-using-kdd-cup99-dataset-fb4cba4189ed
plt.figure(figsize=(10,10))
class_distribution = data['target'].value_counts()
class_distribution.plot(kind='pie')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
# ref: arg sort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# -(train_class_distribution.values): the minus sign will give us in decreasing order
sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1,':', class_distribution.values[i], '(', np.round((class_distribution.values[i]/data.shape[0]*100), 3), '%)')


# In[15]:


data['target'].replace(['ipsweep.','mscan.','nmap.','portsweep.','saint.','satan.'],'Probe',inplace=True)
data['target'].replace(['buffer_overflow.','loadmodule.','perl.','ps.','rootkit.','sqlattack.','xterm.'],'U2R',inplace=True)
data['target'].replace(['apache2.','back.','land.','neptune.','mailbomb.','pod.','processtable.','smurf.','teardrop.','udpstorm.','worm.'],'Dos',inplace=True)
data['target'].replace(['ftp_write.','guess_passwd.','httptunnel.','imap.','multihop.','named.','phf.','sendmail.','snmpgetattack.','snmpguess.','spy.','warezclient.','warezmaster.','xlock.','xsnoop.'],'R2L',inplace=True)


# In[16]:


data['target'].value_counts()


# In[17]:


data['target'].unique()


# In[18]:


## https://medium.com/analytics-vidhya/building-an-intrusion-detection-model-using-kdd-cup99-dataset-fb4cba4189ed
plt.figure(figsize=(20,15))
class_distribution = data['target'].value_counts()
class_distribution.plot(kind='bar')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
# ref: arg sort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# -(train_class_distribution.values): the minus sign will give us in decreasing order
sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1,':', class_distribution.values[i], '(', np.round((class_distribution.values[i]/data.shape[0]*100), 3), '%)')


# In[19]:


data.head()


# # Data Normalization

# In[20]:


#  libraries to normalize data
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# In[21]:


data.select_dtypes(include= 'object').columns


# In[22]:


countinous_features = data.select_dtypes(include= 'number').columns


# In[23]:


print((countinous_features))


# In[25]:


normalization = StandardScaler()


# In[26]:


def normalization(data, countinous_features):
  for i in countinous_features:
    arr = data[i]
    arr = np.array(arr)
    data[i] = normalize.fit_transform(arr.reshape(len(arr),1))
  return data


# In[27]:


data.head()


# In[28]:


data = normalization(data.copy(), countinous_features)


# In[29]:


data.head()


# # Data Encoding

# In[30]:


cat_col = ['protocol_type','service','flag']


# In[31]:


data['protocol_type'].value_counts()


# In[32]:


data['protocol_type'].unique()


# In[33]:


data['service'].unique()


# In[34]:


data['flag'].unique()


# In[35]:


data.shape


# In[36]:


categorical = data[cat_col]


# In[37]:


categorical.head()


# In[38]:


# one hot encoding to change categorical features 
categorical = pd.get_dummies(categorical,columns= cat_col)
categorical.head()


# In[39]:


categorical.shape


# In[40]:


data.columns


# In[41]:


categorical.columns


# # Binary classification

# In[42]:


# normal/abnormal classification
binary_label = pd.DataFrame(data['target'].map(lambda x:'normal' if x=='normal.' else 'abnormal'))


# In[43]:



binary_data = data.copy()
binary_data['label'] = binary_label


# In[44]:


binary_data


# In[47]:


#encoding labels 
label1 = preprocessing.LabelEncoder()
encoding_label = binary_label.apply(label1.fit_transform)
binary_data['intrusion'] = encoding_label


# In[48]:


## https://medium.com/analytics-vidhya/building-an-intrusion-detection-model-using-kdd-cup99-dataset-fb4cba4189ed
plt.figure(figsize=(10,10))
class_distribution = binary_data['label'].value_counts()
class_distribution.plot(kind='pie')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
# ref: arg sort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# -(train_class_distribution.values): the minus sign will give us in decreasing order
sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1,':', class_distribution.values[i], '(', np.round((class_distribution.values[i]/data.shape[0]*100), 3), '%)')


# In[50]:


binary_data 


# In[51]:


# encoding label
binary_data = pd.get_dummies(binary_data,columns=['label'],prefix="",prefix_sep="") 
binary_data['label'] = binary_label
binary_data


# # Multiclass Classification

# In[53]:



mul_data = data.copy()
mul_label = pd.DataFrame(mul_data['target'])


# In[55]:


## https://medium.com/analytics-vidhya/building-an-intrusion-detection-model-using-kdd-cup99-dataset-fb4cba4189ed
plt.figure(figsize=(10,10))
class_distribution = binary_data['target'].value_counts()
class_distribution.plot(kind='pie')
plt.xlabel('Class')
plt.ylabel('Data points per Class')
plt.title('Distribution of yi in train data')
plt.grid()
plt.show()
# ref: arg sort https://docs.scipy.org/doc/numpy/reference/generated/numpy.argsort.html
# -(train_class_distribution.values): the minus sign will give us in decreasing order
sorted_yi = np.argsort(-class_distribution.values)
for i in sorted_yi:
    print('Number of data points in class', i+1,':', class_distribution.values[i], '(', np.round((class_distribution.values[i]/data.shape[0]*100), 3), '%)')


# In[57]:



label2 = preprocessing.LabelEncoder()
encoding_label = mul_label.apply(label2.fit_transform)
mul_data['intrusion'] = encoding_label


# In[58]:



mul_data = pd.get_dummies(mul_data,columns=['target'],prefix="",prefix_sep="") 
mul_data['target'] = mul_label
mul_data


# In[59]:


data.columns


# # Feature Extraction

# In[63]:



numerical_binary = binary_data[countinous_features]
numerical_binary['intrusion'] = binary_data['intrusion']


# In[66]:



correlation= numerical_binary.corr()
correlation_y = abs(correlation['intrusion'])
highest_correlation = correlation_y [correlation_y  >0.5]
highest_correlation.sort_values(ascending=True)


# In[67]:



numerical_binary = binary_data[['count','srv_serror_rate','serror_rate','dst_host_serror_rate','dst_host_srv_serror_rate',
                         'logged_in','dst_host_same_srv_rate','dst_host_srv_count','same_srv_rate']]


# In[68]:


numerical_binary.shape


# In[69]:


categorical.shape


# In[70]:



numerical_binary = numerical_binary.join(categorical)
binary_data = numerical_binary.join(binary_data[['intrusion','abnormal','normal','label']])


# In[71]:


binary_data


# In[72]:


numerical_binary


# In[73]:


categorical.shape


# In[79]:



binary_data.to_csv ("binary_data.csv")


# In[81]:



numerical_multi = mul_data[countinous_features]
numerical_multi['intrusion'] = mul_data['intrusion']


# In[82]:


numerical_multi


# In[84]:



correlation2 = numerical_multi.corr()
correlation2_y = abs(correlation2['intrusion'])
highest_correlation2 = correlation2_y[correlation2_y >0.5]
highest_correlation2.sort_values(ascending=True)


# In[87]:



numerical_multi = mul_data[['dst_host_same_srv_rate','dst_host_srv_serror_rate','dst_host_srv_count','same_srv_rate','count','logged_in','srv_serror_rate','serror_rate','dst_host_serror_rate'
                        ]]


# In[88]:



numerical_multi = numerical_multi.join(categorical)

mul_data = numerical_multi.join(mul_data[['intrusion','Dos','Probe','R2L','U2R','normal.','target']])


# In[89]:


# final dataset for multi-class classification
mul_data


# In[90]:



mul_data.to_csv('multipule_data.csv')


# # PCA for Binary Classification
# 

# In[91]:


from sklearn.decomposition import PCA


# In[92]:


binary_data


# In[94]:


numerical_binary


# In[96]:


binary_data = numerical_binary.join(binary_data[['intrusion','abnormal','normal','label']])


# In[97]:


binary_data


# In[98]:


Y = binary_data['intrusion']


# In[99]:


Y


# In[100]:


binary_data.drop("label", axis=1, inplace=True)


# In[101]:


binary_data.drop("normal", axis=1, inplace=True)


# In[102]:


binary_data.drop("abnormal", axis=1, inplace=True)


# In[103]:


binary_data.drop("intrusion", axis=1, inplace=True)


# In[104]:


binary_data


# In[105]:


pca = PCA(n_components=20)


# In[106]:


data = pca.fit_transform(binary_data)


# In[107]:


data


# In[108]:


array = np.array(data)


# In[109]:


data = pd.DataFrame(array)


# In[110]:


data


# In[111]:


Y


# In[113]:


data.to_csv ("data_pca_binary_final.csv")


# In[115]:


Y.to_csv ("Y_binary_final.csv")


# # Multi Class PCA

# In[ ]:


####################### PLEASE SEE  Multi Class PCA_Final ################################


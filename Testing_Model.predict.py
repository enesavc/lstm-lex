#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Check for GPU
import tensorflow as tf
# if tf.test.gpu_device_name():
#     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
# else:
#     print('Please install GPU version of TF')
tf.test.is_gpu_available()


# In[2]:


#check keras and tf versions
import keras
print(keras.__version__)
import tensorflow as tf
print(tf.__version__)


# In[3]:


from keras.models import Model,load_model
#Preparing Indermediate model
model = load_model('Dorsal_LSTMModel.h5')


# In[4]:


import numpy as np
X = np.load("X.npy")
Y = np.load("Y.npy")
print(X.shape)
print(Y.shape)


# In[5]:


#Train/Test/Validation Set Splitting in Sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test   = train_test_split(X, Y, test_size=0.1, random_state=35)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, random_state=35)


# In[6]:


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])


# In[7]:


pred = model.predict(X_test) 


# In[8]:


import matplotlib.pyplot as plt
for i in range(pred.shape[0]):
    plt.matshow(pred[i,:,:].T, origin='lower', cmap='Blues', vmin=0, vmax=None)
    plt.colorbar()
plt.savefig("figure.png")


# In[10]:


y_pred = []
for i in range(pred.shape[0]):
    y_pred.append(np.argmax(pred[i,196,:])) #here I used the last time slice because above heatmaps seems ok

y_true = np.argmax(Y_test.sum(axis=1),axis=1)

y_pred = np.array(y_pred, dtype=np.float32)
y_true = np.array(y_true, dtype=np.float32)
print('Predicted labels:', y_pred)
print('True labels     :', y_true)


# In[11]:


#percent correct of model.predict
quotient = 8 / 15
percentage = quotient * 100
print(percentage)


# In[12]:


keras.metrics.categorical_accuracy(y_true, y_pred)


# In[13]:


keras.metrics.binary_accuracy(y_true, y_pred)


# In[15]:


#keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
#Can not squeeze dim[0], expected a dimension of 1, got 15 [Op:Squeeze]


# In[ ]:





#2021-10-14
#EA
#This script is created for plotting the cosine sim acc over epochs.

import warnings
warnings.filterwarnings("ignore")

import keras
import tensorflow as tf
import datagenerator as dg #ventral is ventral_datagenerator
import numpy as np
import matplotlib.pyplot as plt
import gc

#
EpochsNum = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000,3100,3200,3300,3400,3500,3600,3700,3800,3900,4000,4100,4200,4300,4400,4500,4600,4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,6300,6400,6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,7600,7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,9800,9900,10000]

AvgCosSim = []
PredWords = []
for f in EpochsNum:
  #load the model from checkpoint
  model = tf.keras.models.load_model('/autofs/space/euler_001/users/lstm/checkpoints_cochs10k_dorsal/_{}.hd5f'.format(f))
  #read the partition and labels
  partition = dg.read_dict('/autofs/space/euler_001/users/lstm/dicts/partition.csv')
  labels = dg.read_patterns('/autofs/space/euler_001/users/lstm/dicts/DorsalSparse.csv') #Change this line for dorsal
  exec("partition['train'] = " + partition['train'])
  exec("partition['validation'] = " + partition['validation'])
  exec("partition['test'] = " + partition['test'])
  partition['train'] = partition['train'] + partition['validation']
  partition['validation'] = partition['test']
  # Parameters
  params = {'dim': (226,211), #(124,92) for spects
            'batch_size': 100,
            'n_classes': 300, #ventral was 1007
            #'n_channels': 1,
            'shuffle': False}

  # Datasets
  partition = partition
  labels = labels

  # Generators
  training_generator = dg.DataGenerator(partition['train'], labels, **params) #9 token per word
  validation_generator = dg.DataGenerator(partition['validation'], labels, **params) #1 token per word
  # Predict
  #y_pred = model.predict(training_generator)
  y_pred = model.predict(validation_generator)
  print('Y_pred shape:',y_pred.shape)

  #this is the loop over on datagenerator to get the X and y.
  data = []     # store all the generated data batches
  y_true = []   # store all the generated label batches
  max_iter = 100  # maximum number of iterations, in each iteration one batch is generated; the proper value depends on batch size and size of whole data
  i = 0
  #for d, l in training_generator:
  for d, l in validation_generator:
      data.append(d)
      y_true.append(l)
      i += 1
      if i == max_iter:
          break

    #Now we have two lists of tensor batches.
    #We need to reshape them to make two tensors, one for data (i.e X) and one for labels (i.e. y):
  data = np.array(data)
  data = np.reshape(data, (data.shape[0]*data.shape[1],) + data.shape[2:])
  #print('data shape:', data.shape)

  y_true = np.array(y_true)
  #print(y_true.shape)
  y_true = np.reshape(y_true, (y_true.shape[0]*y_true.shape[1],) + y_true.shape[2:])
  print('Y_true shape:', y_true.shape)

  print(f)
  #Average Cosine Similarity at the End
  np.set_printoptions(suppress=True)
  cosines1 = []
  for k in range(y_pred.shape[0]):
          #cosine = dg.vector_cosine(y_pred[k,123,:], y_true[k,123,:]) # for specs
          cosine = dg.vector_cosine(y_pred[k,225,:], y_true[k,225,:])  # for cochs
          cosines1.append(cosine)
  cosines1 = np.array(cosines1)
  #cosines = np.reshape(cosines, (800, 800))
  #savetxt('cosines.csv', cosines, delimiter=',', fmt='%f')
  print('Average Cosine Similarity:', np.mean(cosines1))
  AvgCosSim.append(np.mean(cosines1))

  cosines2 = []
  for k in range(y_pred.shape[0]):
      for j in range(y_pred.shape[0]):
          #cosine = dg.vector_cosine(y_pred[k,123,:], y_true[j,123,:]) # for specs
          cosine = dg.vector_cosine(y_pred[k,225,:], y_true[j,225,:])  # for cochs
          cosines2.append(cosine)
  cosines2 = np.array(cosines2)
  #cosines2 = np.reshape(cosines2, (7900, 7900))
  cosines2 = np.reshape(cosines2, (800, 800))
  maxrow = np.argmax(cosines2, axis=1)
  i = 0
  for k in range(cosines2.shape[0]):
    #print(cosines[k,k])

      if cosines2[k][maxrow[k]] == cosines2[k,k]:
          i = i + 1
  print('Predicted Number of Words:', i)
  PredWords.append(i)
  del model
  del y_pred
  del y_true
  del cosine
  del cosines1
  del cosines2
  gc.collect()


#plotting
AvgCosSim = np.array(AvgCosSim)
print(AvgCosSim)
PredWords = np.array(PredWords)
print(PredWords)

# plotting
plt.title("Validation Data")
#plt.title("Training Data")
plt.xlabel("Epochs")
plt.ylabel("AvgCosSim")
#plt.xticks(AvgCosSim,EpochsNumS)
plt.plot(AvgCosSim, label='Average Cosine Similarity', marker="o")
plt.legend(loc='upper left')
plt.savefig('AvgCosSim.png')
plt.clf()
plt.title("Validation Data")
#plt.title("Training Data")
plt.xlabel("Epochs")
plt.ylabel("# of Correct Words")
#plt.xticks(PredWords,EpochsNumS)
plt.plot(PredWords, label='Number of Predicted Words', marker="o")
plt.legend(loc='upper left')
plt.savefig('PredWords.png')
#plt.show()

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# Load the libraries
import pandas as pd
import numpy as np
import seaborn as sns
import os
import cv2
import matplotlib.pyplot as plt
import random
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import keras
from keras import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils import shuffle
from IPython.display import display
import os
def load_images_from_folder(folder, label = ""):
  path = []
  for filename in os.listdir(folder):
    print(filename)
    img_path = os.path.join(folder,filename)
    if img_path is not None:
      path.append([label,img_path])
  return path
images = []
dirp = "D:\ML Dataset"
for f in os.listdir(dirp):
  print("Running1")
  images += load_images_from_folder(dirp+"/"+f,label = f)
Running1
Chickoo (1100).jpg
Chickoo (1101).jpg
Chickoo (1102).jpg
Chickoo (1103).jpg
Chickoo (1104).jpg
....

len(images)
5818
Creating a dataframe of fruits and their respective image paths

x = images[0][1]
y = cv2.imread(x)
plt.imshow(y)
<matplotlib.image.AxesImage at 0x20d45518f10>

df = pd.DataFrame(images, columns = ["fruit", "path"])
df
fruit	path
0	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1).jpg
1	Chickoo	D:\ML Dataset/Chickoo\Chickoo (10).jpg
2	Chickoo	D:\ML Dataset/Chickoo\Chickoo (100).jpg
3	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1000).jpg
4	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1001).jpg
...	...	...
5813	Watermelon	D:\ML Dataset/Watermelon\Watermelon (995).jpg
5814	Watermelon	D:\ML Dataset/Watermelon\Watermelon (996).jpg
5815	Watermelon	D:\ML Dataset/Watermelon\Watermelon (997).jpg
5816	Watermelon	D:\ML Dataset/Watermelon\Watermelon (998).jpg
5817	Watermelon	D:\ML Dataset/Watermelon\Watermelon (999).jpg
5818 rows × 2 columns

df['fruit'].value_counts().unique
<bound method Series.unique of Papaya        2006
Chickoo       2000
Watermelon    1812
Name: fruit, dtype: int64>
df = shuffle(df, random_state = 0)  #shuffle just shuffles the values
df = df.reset_index(drop=True)  #reset_index names the indices from 0 to 5817 in order
df
fruit	path
0	Papaya	D:\ML Dataset/Papaya\Papaya (996).jpg
1	Watermelon	D:\ML Dataset/Watermelon\Watermelon (1381).jpg
2	Papaya	D:\ML Dataset/Papaya\Papaya (1319).jpg
3	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1736).jpg
4	Papaya	D:\ML Dataset/Papaya\Papaya (350).jpg
...	...	...
5813	Papaya	D:\ML Dataset/Papaya\Papaya (1256).jpg
5814	Papaya	D:\ML Dataset/Papaya\Papaya (1411).jpg
5815	Watermelon	D:\ML Dataset/Watermelon\Watermelon (561).jpg
5816	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1578).jpg
5817	Papaya	D:\ML Dataset/Papaya\Papaya (231).jpg
5818 rows × 2 columns

fruit_names = sorted(df.fruit.unique())
future_num = dict(zip(fruit_names, [t for t in range(len(fruit_names))]))
df["label"] = df["fruit"].map(future_num)
print(future_num)
{'Chickoo': 0, 'Papaya': 1, 'Watermelon': 2}
df.head()
fruit	path	label
0	Papaya	D:\ML Dataset/Papaya\Papaya (996).jpg	1
1	Watermelon	D:\ML Dataset/Watermelon\Watermelon (1381).jpg	2
2	Papaya	D:\ML Dataset/Papaya\Papaya (1319).jpg	1
3	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1736).jpg	0
4	Papaya	D:\ML Dataset/Papaya\Papaya (350).jpg	1
Distribution of different fruits in the dataset

plt.figure(figsize=(12,8))
p = df["fruit"].value_counts().plot(kind='bar')

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 15))

for i, ax in enumerate(axes.flat):
    ax.imshow(plt.imread(df.path[i]))  #plt.imread reads in RGB Format while cv2.imread reads in BGR Format
    ax.set_title(df.fruit[i]) 
plt.tight_layout()
plt.show()

def load_img(df):
    img_paths = df["path"].values
    img_labels = df["label"].values
    X = []
    y = []
    for i,path in enumerate(img_paths):
        img = cv2.imread(path)
        img = cv2.resize(img, (224,224))
        label = img_labels[i]
        X.append(img)
        y.append(label)
    return np.array(X),np.array(y)
def from_categorical(lst):  
    lst = lst.tolist()
    lst2 = []
    for x in lst:
        lst2.append(x.index(max(x)))
    return lst2
    
    
MDELLING AND TRAINING
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3),input_shape=(224,224,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding = 'same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))


model.add(Flatten())

model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(len(future_num)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 224, 224, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 112, 112, 32)      0         
_________________________________________________________________
dropout (Dropout)            (None, 112, 112, 32)      0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 112, 112, 64)      18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 56, 56, 64)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 56, 56, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 56, 56, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 28, 28, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 28, 28, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 28, 28, 64)        73792     
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 14, 14, 32)        18464     
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 7, 7, 32)          0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 7, 7, 16)          4624      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 3, 3, 16)          0         
_________________________________________________________________
dropout_5 (Dropout)          (None, 3, 3, 16)          0         
_________________________________________________________________
flatten (Flatten)            (None, 144)               0         
_________________________________________________________________
dense (Dense)                (None, 256)               37120     
_________________________________________________________________
activation (Activation)      (None, 256)               0         
_________________________________________________________________
dropout_6 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 3)                 771       
_________________________________________________________________
activation_1 (Activation)    (None, 3)                 0         
=================================================================
Total params: 228,019
Trainable params: 228,019
Non-trainable params: 0
_________________________________________________________________
from sklearn.model_selection import train_test_split
train_set , test_set = train_test_split(df,test_size=0.2,random_state=17)
train_set
fruit	path	label
1783	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1160).jpg	0
2220	Papaya	D:\ML Dataset/Papaya\Papaya (1295).jpg	1
322	Watermelon	D:\ML Dataset/Watermelon\Watermelon (814).jpg	2
3831	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1206).jpg	0
1227	Watermelon	D:\ML Dataset/Watermelon\Watermelon (1609).jpg	2
...	...	...	...
1337	Papaya	D:\ML Dataset/Papaya\Papaya (1242).jpg	1
406	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1516).jpg	0
5510	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1855).jpg	0
2191	Papaya	D:\ML Dataset/Papaya\Papaya (1574).jpg	1
2671	Chickoo	D:\ML Dataset/Chickoo\Chickoo (936).jpg	0
4654 rows × 3 columns

test_set
fruit	path	label
2228	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1517).jpg	0
3527	Papaya	D:\ML Dataset/Papaya\Papaya (840).jpg	1
4503	Chickoo	D:\ML Dataset/Chickoo\Chickoo (446).jpg	0
1853	Watermelon	D:\ML Dataset/Watermelon\Watermelon (277).jpg	2
1226	Chickoo	D:\ML Dataset/Chickoo\Chickoo (213).jpg	0
...	...	...	...
5061	Watermelon	D:\ML Dataset/Watermelon\Watermelon (700).jpg	2
2858	Chickoo	D:\ML Dataset/Chickoo\Chickoo (438).jpg	0
1887	Papaya	D:\ML Dataset/Papaya\Papaya (1733).jpg	1
1516	Papaya	D:\ML Dataset/Papaya\Papaya (731).jpg	1
1429	Papaya	D:\ML Dataset/Papaya\Papaya (663).jpg	1
1164 rows × 3 columns

train_set['fruit'].value_counts().unique
<bound method Series.unique of Papaya        1611
Chickoo       1589
Watermelon    1454
Name: fruit, dtype: int64>
X_train, y_train = load_img(train_set)
y_train = to_categorical(y_train)
model.summary()
Model: "sequential_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_11 (Conv2D)           (None, 224, 224, 32)      896       
_________________________________________________________________
max_pooling2d_9 (MaxPooling2 (None, 112, 112, 32)      0         
_________________________________________________________________
dropout_9 (Dropout)          (None, 112, 112, 32)      0         
_________________________________________________________________
conv2d_12 (Conv2D)           (None, 112, 112, 64)      18496     
_________________________________________________________________
max_pooling2d_10 (MaxPooling (None, 56, 56, 64)        0         
_________________________________________________________________
dropout_10 (Dropout)         (None, 56, 56, 64)        0         
_________________________________________________________________
conv2d_13 (Conv2D)           (None, 56, 56, 128)       73856     
_________________________________________________________________
max_pooling2d_11 (MaxPooling (None, 28, 28, 128)       0         
_________________________________________________________________
dropout_11 (Dropout)         (None, 28, 28, 128)       0         
_________________________________________________________________
conv2d_14 (Conv2D)           (None, 28, 28, 64)        73792     
_________________________________________________________________
max_pooling2d_12 (MaxPooling (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 14, 14, 64)        0         
_________________________________________________________________
conv2d_15 (Conv2D)           (None, 14, 14, 32)        18464     
_________________________________________________________________
max_pooling2d_13 (MaxPooling (None, 7, 7, 32)          0         
_________________________________________________________________
dropout_13 (Dropout)         (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_16 (Conv2D)           (None, 7, 7, 16)          4624      
_________________________________________________________________
max_pooling2d_14 (MaxPooling (None, 3, 3, 16)          0         
_________________________________________________________________
dropout_14 (Dropout)         (None, 3, 3, 16)          0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 144)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 256)               37120     
_________________________________________________________________
activation_2 (Activation)    (None, 256)               0         
_________________________________________________________________
dropout_15 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_6 (Dense)              (None, 3)                 771       
_________________________________________________________________
activation_3 (Activation)    (None, 3)                 0         
=================================================================
Total params: 228,019
Trainable params: 228,019
Non-trainable params: 0
_________________________________________________________________
history = model.fit(X_train,y_train,batch_size = 128,epochs = 30,validation_split = 0.2,verbose = 2)
Epoch 1/30
30/30 - 196s - loss: 6.7064 - accuracy: 0.3328 - val_loss: 1.1024 - val_accuracy: 0.3008
Epoch 2/30
30/30 - 198s - loss: 1.0891 - accuracy: 0.3640 - val_loss: 1.0729 - val_accuracy: 0.3287
Epoch 3/30
30/30 - 200s - loss: 1.0533 - accuracy: 0.4177 - val_loss: 1.0627 - val_accuracy: 0.5392
Epoch 4/30
30/30 - 199s - loss: 0.9068 - accuracy: 0.5670 - val_loss: 0.8570 - val_accuracy: 0.7594
Epoch 5/30
30/30 - 199s - loss: 0.7093 - accuracy: 0.7002 - val_loss: 0.8139 - val_accuracy: 0.6627
Epoch 6/30
30/30 - 199s - loss: 0.6365 - accuracy: 0.7486 - val_loss: 0.6903 - val_accuracy: 0.8228
Epoch 7/30
30/30 - 200s - loss: 0.4829 - accuracy: 0.8184 - val_loss: 0.5038 - val_accuracy: 0.8518
Epoch 8/30
30/30 - 200s - loss: 0.4335 - accuracy: 0.8442 - val_loss: 0.4985 - val_accuracy: 0.8722
Epoch 9/30
30/30 - 199s - loss: 0.3833 - accuracy: 0.8625 - val_loss: 0.4362 - val_accuracy: 0.9012
Epoch 10/30
30/30 - 199s - loss: 0.3599 - accuracy: 0.8662 - val_loss: 0.3491 - val_accuracy: 0.9151
Epoch 11/30
30/30 - 203s - loss: 0.3511 - accuracy: 0.8764 - val_loss: 0.4476 - val_accuracy: 0.8840
Epoch 12/30
30/30 - 200s - loss: 0.2777 - accuracy: 0.8993 - val_loss: 0.3282 - val_accuracy: 0.9108
Epoch 13/30
30/30 - 201s - loss: 0.2443 - accuracy: 0.9140 - val_loss: 0.2873 - val_accuracy: 0.9356
Epoch 14/30
30/30 - 198s - loss: 0.2250 - accuracy: 0.9173 - val_loss: 0.3145 - val_accuracy: 0.8872
Epoch 15/30
30/30 - 203s - loss: 0.2116 - accuracy: 0.9277 - val_loss: 0.3339 - val_accuracy: 0.9280
Epoch 16/30
30/30 - 200s - loss: 0.2497 - accuracy: 0.9122 - val_loss: 0.3086 - val_accuracy: 0.9409
Epoch 17/30
30/30 - 200s - loss: 0.1781 - accuracy: 0.9358 - val_loss: 0.3196 - val_accuracy: 0.9409
Epoch 18/30
30/30 - 197s - loss: 0.1944 - accuracy: 0.9337 - val_loss: 0.3089 - val_accuracy: 0.9194
Epoch 19/30
30/30 - 200s - loss: 0.1758 - accuracy: 0.9396 - val_loss: 0.1645 - val_accuracy: 0.9592
Epoch 20/30
30/30 - 198s - loss: 0.1362 - accuracy: 0.9559 - val_loss: 0.2292 - val_accuracy: 0.9463
Epoch 21/30
30/30 - 197s - loss: 0.1270 - accuracy: 0.9586 - val_loss: 0.2001 - val_accuracy: 0.9656
Epoch 22/30
30/30 - 200s - loss: 0.1180 - accuracy: 0.9635 - val_loss: 0.2342 - val_accuracy: 0.9248
Epoch 23/30
30/30 - 197s - loss: 0.1879 - accuracy: 0.9345 - val_loss: 0.2142 - val_accuracy: 0.9656
Epoch 24/30
30/30 - 196s - loss: 0.1387 - accuracy: 0.9554 - val_loss: 0.2613 - val_accuracy: 0.9538
Epoch 25/30
30/30 - 200s - loss: 0.1363 - accuracy: 0.9565 - val_loss: 0.2086 - val_accuracy: 0.9538
Epoch 26/30
30/30 - 203s - loss: 0.1134 - accuracy: 0.9616 - val_loss: 0.2067 - val_accuracy: 0.9441
Epoch 27/30
30/30 - 201s - loss: 0.0926 - accuracy: 0.9672 - val_loss: 0.1180 - val_accuracy: 0.9753
Epoch 28/30
30/30 - 200s - loss: 0.1187 - accuracy: 0.9602 - val_loss: 0.2125 - val_accuracy: 0.9194
Epoch 29/30
30/30 - 199s - loss: 0.1035 - accuracy: 0.9662 - val_loss: 0.1593 - val_accuracy: 0.9667
Epoch 30/30
30/30 - 196s - loss: 0.0874 - accuracy: 0.9753 - val_loss: 0.1177 - val_accuracy: 0.9753
plt.figure(0)

plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure(1)
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.title('Loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
<matplotlib.legend.Legend at 0x1ce915dc3a0>


def display_stats(y_test_class, pred):
# Display prediction statistics
    print("Classification Report:\n")
    print(classification_report(y_test_class, pred))
    print("\nConfusion Matrix:\n\n")
    print(confusion_matrix(y_test_class, pred))
    print("\nAccuracy:", round(accuracy_score(y_test_class, pred),5))
    return y_test_class
Predictions

test_set
fruit	path	label
2228	Chickoo	D:\ML Dataset/Chickoo\Chickoo (1517).jpg	0
3527	Papaya	D:\ML Dataset/Papaya\Papaya (840).jpg	1
4503	Chickoo	D:\ML Dataset/Chickoo\Chickoo (446).jpg	0
1853	Watermelon	D:\ML Dataset/Watermelon\Watermelon (277).jpg	2
1226	Chickoo	D:\ML Dataset/Chickoo\Chickoo (213).jpg	0
...	...	...	...
5061	Watermelon	D:\ML Dataset/Watermelon\Watermelon (700).jpg	2
2858	Chickoo	D:\ML Dataset/Chickoo\Chickoo (438).jpg	0
1887	Papaya	D:\ML Dataset/Papaya\Papaya (1733).jpg	1
1516	Papaya	D:\ML Dataset/Papaya\Papaya (731).jpg	1
1429	Papaya	D:\ML Dataset/Papaya\Papaya (663).jpg	1
1164 rows × 3 columns

x_test,y_test = load_img(test_set)
pred = model.predict_classes(x_test)



              precision    recall  f1-score   support

           0       1.00      1.00      1.00       411
           1       0.98      0.99      0.99       395
           2       0.99      0.98      0.98       358

    accuracy                           0.99      1164
   macro avg       0.99      0.99      0.99      1164
weighted avg       0.99      0.99      0.99      1164


Confusion Matrix:


[[409   0   2]
 [  0 393   2]
 [  1   7 350]]

Accuracy: 0.98969
array([0, 1, 0, ..., 1, 1, 1], dtype=int64)
print(np.count_nonzero(pred == 1))
400
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
s = {
    '0' : 'Chickoo',
    '1' : 'Papaya',
    '2' : 'Watermelon'
}

for i, ax in enumerate(ax.flat):
    ax.imshow(x_test[i])
    ax.set_title("Actual label: " + s[str(y_test[i])] + ", Predicted label: "+ s[str(pred[i])])
plt.tight_layout()
plt.show()

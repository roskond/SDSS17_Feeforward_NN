# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% id="rm8zDr92rT0U"
# importing packages
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve, accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam, RMSprop

# %%
# !pip install tensorflow

# %% id="ZdkdG_F239pz"
import pandas as pd

# %% id="kqueacjs7FQf"
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# %% id="dMeoGCynZM3f"
import os


# %% colab={"base_uri": "https://localhost:8080/"} executionInfo={"elapsed": 5867, "status": "ok", "timestamp": 1674226026784, "user": {"displayName": "Roshan V. Kondapalli", "userId": "11708963788614939151"}, "user_tz": 0} id="cJNq9a-Rs6t3" outputId="382f7278-a699-4c5f-d87f-f6ae657deccd"
filepath = 'SDSS17.csv'

# %% colab={"base_uri": "https://localhost:8080/", "height": 206} executionInfo={"elapsed": 2233, "status": "ok", "timestamp": 1674226096535, "user": {"displayName": "Roshan V. Kondapalli", "userId": "11708963788614939151"}, "user_tz": 0} id="zvmPefpx4evC" outputId="e443d196-b017-467f-d890-3910aa81bbc0"
df = pd.read_csv(filepath)
df.head(5)

# %% id="Pkw0RU7PsnyL"
from sklearn.preprocessing import LabelEncoder

# %% id="R0U9HpBZ4scf"
dfs = df.drop(['obj_ID','alpha','delta','run_ID','rerun_ID','cam_col','field_ID','fiber_ID','plate','spec_obj_ID'], axis = 1)

# %% id="79bX4Dbp5Iyx"
dfs = dfs.rename(columns={'u':'Ultraviolet Filter', 'r':'Red Filter', 'g': 'Green Filter',
                          'i': 'Near Infrared Filter', 'z':'Infrared Filter',
                          'redshift':'Redshift','class':'Class'})

# %% colab={"base_uri": "https://localhost:8080/", "height": 314} executionInfo={"elapsed": 17, "status": "error", "timestamp": 1674226107802, "user": {"displayName": "Roshan V. Kondapalli", "userId": "11708963788614939151"}, "user_tz": 0} id="6UQGt1-LjbMd" outputId="9a02ccdf-fb78-44aa-8cff-23bd700609f2"
#sns.heatmap(dfs.corr, cmap=sns.cubehelix_palette(as_cmap=True))

# %% colab={"background_save": true} id="h_c8WGzvk2E_" outputId="f42bd35c-d0af-47fe-e92b-23d427875956"
sns.pairplot(dfs, hue="Class")

# %%
plt.savefig("pairplotsloan.png")

# %%
f,ax = plt.subplots(figsize=(12,8))
sns.heatmap(dfs.corr(), cmap=sns.color_palette("YlOrBr", as_cmap=True), annot=True, linewidths=0.5, fmt= '.2f',ax=ax)
plt.show() 

# %% colab={"background_save": true} id="n7A8fpyZ5YXc"
dfs.head()

# %% colab={"background_save": true} id="VJDEUJSMTWKy"
from imblearn.over_sampling import SMOTE
from collections import Counter

# %% colab={"background_save": true} id="WRhuEGfs7B9r"
stellar = dfs.copy()

# %% id="_vQPRh8H7sYa"
X = stellar.drop(['Class'], axis = 1)
Y = stellar.loc[:, 'Class'].values

# %% id="fXeaRV9E8cFr"
X.head()

# %% id="hyElVM1QgZWY"
rb = RobustScaler()
X = rb.fit_transform(X)

# %% id="d6_Yjyot8fhG"
#lb = LabelEncoder()
#Y = lb.fit_transform(Y)
#Y = tf.keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)


# %% id="MUPCuWDrw6tU"
lb = LabelBinarizer()
Y = lb.fit_transform(Y)


# %% id="4rHybeDuUH1n"
sm = SMOTE(random_state=42)
X, Y = sm.fit_resample(X,Y)
#print('Resampled dataset %s' % Counter(Y))

# %% id="-7jWSJFY888b"
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state = None) 

# %% id="s7Pvus-8ewuc"
from sklearn.model_selection import GridSearchCV

# %% id="HIuTeCIqaMd8"
#Neural Network #from [18]

# %% [markdown] id="6Kg3Tesp0dUV"
# Neural Networks
#

# %% id="Xa-4etS-Slsv"
from tensorflow.keras.layers import Dropout

# %% id="vsFQsuCz9e1R"
#Network architecture

model = Sequential()
#model.add(Dropout(.28))
model.add(Dense(50, input_shape = (7,), activation = 'sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.4))
model.add(Dense(3, activation = 'softmax'))

# %% id="YbPkcFg0TMmt"
model.summary()

# %% id="NEMRO4LLhV2v"
#SGD Optimizer

model.compile(SGD(.003), "categorical_crossentropy", metrics=["accuracy"])
H_SGD = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 100)

# %% id="DZdSrBK3pnw0"
print("Classification Report: ")
predictions = model.predict(X_test, batch_size=128)
print(classification_report(Y_test.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in lb.classes_]))

# %%
from sklearn.metrics import confusion_matrix

# %%
rfc_matrix = confusion_matrix(H_SGD,X_test,Y_test,labels=['GALAXY','QSO','STAR'])
plt.show()
plt.savefig("rfc_matrix_pca.png")

# %% id="ulkMD-Ljaz2m"
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H_SGD.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 100), H_SGD.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, 100), H_SGD.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, 100), H_SGD.history["val_accuracy"], label="Validation Accuracty")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
#plt.savefig('first_plot.png')

# %% id="oXlGTTGKItXe"
#Network architecture

model_1 = Sequential()
#model.add(Dropout(.28))
model_1.add(Dense(50, input_shape = (7,), activation = 'sigmoid'))
model_1.add(Dropout(0.4))
model_1.add(Dense(50, activation='sigmoid'))
model_1.add(Dropout(0.4))
model_1.add(Dense(3, activation = 'softmax'))

# %% id="WBz9ynLVJiql"

# %% id="6ews9UBWJSRv"
model_1.summary()

# %% id="t3XwC4wEJmYK"
#RMSProp

model_1.compile(RMSprop(.003), "categorical_crossentropy", metrics=["accuracy"])
H_RMS = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 100)

# %% id="S4aXKvA5JueP"

# %% id="NCsFshM0bRz1"
print("[INFO] evaluating network...")
predictions = model_1.predict(X_test, batch_size=128)
print(classification_report(Y_test.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in lb.classes_]))

# %% id="b4eVeTX7bT2C"
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H_RMS.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 100), H_RMS.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, 100), H_RMS.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, 100), H_RMS.history["val_accuracy"], label="Validation Accuracty")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('first1_plot.png')

# %% id="7BbZloc8JzSx"

# %% id="1EE-pj2qJ0Wh"
model_2 = Sequential()
#model.add(Dropout(.28))
model_2.add(Dense(5, input_shape = (7,), activation = 'sigmoid'))
model_2.add(Dropout(0.4))
model_2.add(Dense(50, activation='sigmoid'))
model_2.add(Dropout(0.4))
model_2.add(Dense(3, activation = 'softmax'))

# %% id="K677KoemJ5Dy"
#Adam Optimizer

model_2.compile(Adam(.003), "categorical_crossentropy", metrics=["accuracy"])
H_AD = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs = 100)

# %% id="26dgZoNBb68b"
print("[INFO] evaluating network...")
predictions = model_2.predict(X_test, batch_size=128)
print(classification_report(Y_test.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in lb.classes_]))

# %% id="pORzzGqZb7GV"
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 100), H_AD.history["loss"], label="Training Loss")
plt.plot(np.arange(0, 100), H_AD.history["val_loss"], label="Validation Loss")
plt.plot(np.arange(0, 100), H_AD.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, 100), H_AD.history["val_accuracy"], label="Validation Accuracty")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.savefig('first2_plot.png')

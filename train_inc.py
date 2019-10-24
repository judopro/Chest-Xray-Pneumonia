import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import math

from keras.applications import InceptionV3
from keras.models import load_model

from keras.applications.inception_v3 import preprocess_input as incep_preprocess_input

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, BatchNormalization, Dropout

from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 128
target_size = (299, 299)

print("Using Inception v3")
base_model = InceptionV3(weights='imagenet', input_shape=(299, 299, 3), include_top=False)

x = base_model.output
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.33)(x)
x = BatchNormalization()(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
model.summary()

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function=incep_preprocess_input,
                                       shear_range=0.2, zoom_range=0.2,
                                       horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('./data/train/',
                                                    target_size=target_size, color_mode='rgb',
                                                    batch_size=batch_size, class_mode='binary',
                                                    shuffle=True, seed=42)

val_datagen = ImageDataGenerator(preprocessing_function=incep_preprocess_input)

val_generator = val_datagen.flow_from_directory('./data/test/',
                                                target_size=target_size, color_mode="rgb",
                                                batch_size=batch_size, shuffle=False, class_mode="binary")

step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = val_generator.n // val_generator.batch_size

chkpt1 = ModelCheckpoint(filepath="best_model_acc.hd5", monitor='accuracy', verbose=1, save_best_only=True, save_weights_only=False)
chkpt2 = ModelCheckpoint(filepath="best_model_val_acc.hd5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False)
chkpt3 = ModelCheckpoint(filepath="best_model_val_loss.hd5", monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=False)

df = pd.DataFrame({'data':train_generator.classes})
no_pne = int(df[df.data==train_generator.class_indices['NORMAL']].count())
yes_pne = int(df[df.data==train_generator.class_indices['PNEUMONIA']].count())

imb_rat = round(yes_pne / no_pne, 2)
sq_imb_rat = round(math.sqrt(imb_rat), 2)

no_weight = sq_imb_rat
yes_weight = 1.0

cweights = {
    train_generator.class_indices['NORMAL']:no_weight,
    train_generator.class_indices['PNEUMONIA']:yes_weight
}


text = "No PNE:{:.0f}\nYes PNE:{:.0f}\nImbalance Ratio: {:.2f}".format(no_pne, yes_pne, imb_rat)
print(text)
print("Using weight multipliers for classes as follows:")
print(cweights)

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    validation_data=val_generator,
                    validation_steps=step_size_valid,
                    callbacks=[chkpt1,chkpt2,chkpt3],
                    class_weight=cweights,
                    epochs=100, verbose=1)


model = load_model("best_model_val_acc.hd5")

probabilities = model.predict_generator(val_generator)
orig = val_generator.classes
preds = probabilities > 0.5

cm = confusion_matrix(orig, preds)
print(cm)

tn, fp, fn, tp = cm.ravel()

precision = tp/(tp+fp)
recall = tp/(tp+fn)

title = "Recall:{:.2f}%\nPrecision:{:.0f}%".format(recall * 100, precision * 100)
print(title)

plot_confusion_matrix(cm,figsize=(10,5), hide_ticks=True, cmap=plt.cm.Blues)
plt.xticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.yticks(range(2), ['Normal', 'Pneumonia'], fontsize=16)
plt.title(title)
plt.show()

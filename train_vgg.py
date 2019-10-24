import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Model
from keras.models import load_model
from keras.optimizers import Adam, RMSprop
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.layers import Flatten, Dense, BatchNormalization, Dropout
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix

import tensorflow as tf

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

batch_size = 128
target_size = (150, 150)

print("Using VGG-16")
base_model = VGG16(weights='imagenet', input_shape=(150, 150, 3), include_top=False)

x = base_model.output
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.33)(x)
x = BatchNormalization()(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

for layer in base_model.layers:
    if layer.name != 'block5_conv3':
        layer.trainable = False
    else:
        layer.trainable = True
        print("Setting 'block5_conv3' trainable")

for layer in model.layers:
    print("{} {}".format(layer.name, layer.trainable))

model.summary()

#optAdam = Adam(learning_rate=0.0001)
optRMS = RMSprop(learning_rate=0.0001)

model.compile(loss='binary_crossentropy', optimizer=optRMS, metrics=['accuracy'])

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range=0.2, zoom_range=0.2,
                                   horizontal_flip=True, fill_mode='nearest')

train_generator = train_datagen.flow_from_directory('./data/train/',
                                                    target_size=target_size, color_mode='rgb',
                                                    batch_size=batch_size, class_mode='binary',
                                                    shuffle=True, seed=42)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

val_generator = val_datagen.flow_from_directory('./data/test/',
                                                target_size=target_size, color_mode="rgb",
                                                batch_size=batch_size, shuffle=False, class_mode="binary")

step_size_train = train_generator.n // train_generator.batch_size
step_size_valid = val_generator.n // val_generator.batch_size

filename = "v_best_model_val_acc_{epoch:02d}_{val_accuracy:.2f}.hd5"
chkpt = ModelCheckpoint(filepath=filename, monitor='val_accuracy', verbose=1, save_best_only=False, save_weights_only=False)

df = pd.DataFrame({'data':train_generator.classes})
#no_pne = int(df[df.data==train_generator.class_indices['NORMAL']].count())
#yes_pne = int(df[df.data==train_generator.class_indices['PNEUMONIA']].count())

#imb_rat = round(yes_pne / no_pne, 2)
#sq_imb_rat = round(math.sqrt(imb_rat), 2)

#no_weight = sq_imb_rat
#yes_weight = 1.0

#cweights = {
#    train_generator.class_indices['NORMAL']:no_weight,
#    train_generator.class_indices['PNEUMONIA']:yes_weight
#}


#text = "No PNE:{:.0f}\nYes PNE:{:.0f}\nImbalance Ratio: {:.2f}".format(no_pne, yes_pne, imb_rat)
#print(text)
#print("Using weight multipliers for classes as follows:")
#print(cweights)

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=step_size_train,
                    validation_data=val_generator,
                    validation_steps=step_size_valid,
                    callbacks=[chkpt],
                    #class_weight=cweights,
                    epochs=100, verbose=1)


model = load_model("v_best_model_val_acc.hd5")
val_generator.reset()
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

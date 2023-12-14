from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Conv2DTranspose, UpSampling2D, add
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import re
import os
import cv2

folder_path = 'x4'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def Resolution(path,size):
    names = sorted(os.listdir(path))
    allHighimages = []
    for name in names:
        fpath = path + name
        img = cv2.imread(fpath, cv2.IMREAD_COLOR)
        image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #change bgr to rgb
        
        highimage = cv2.resize(image,(size,size),cv2.INTER_CUBIC) 
        highimage=highimage[:, :, :].astype(float) / 255 
        
        allHighimages.append(highimage) 
    allHighimages = np.array(allHighimages)
    return allHighimages

input_img = Input(shape=(None, None, 3))
l1 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu', 
            activity_regularizer = regularizers.l1(10e-10))(input_img)

l2 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu', 
            activity_regularizer = regularizers.l1(10e-10))(l1)

l3 = MaxPooling2D(padding = 'same')(l2)   
l3 = Dropout(0.3)(l3)

l4 = Conv2D(128, (3, 3),  padding = 'same', activation = 'relu', 
            activity_regularizer = regularizers.l1(10e-10))(l3)

l5 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu', 
            activity_regularizer = regularizers.l1(10e-10))(l4)

l6 = MaxPooling2D(padding = 'same')(l5) #2

l7 = Conv2D(256, (3, 3), padding = 'same', activation = 'relu', 
            activity_regularizer = regularizers.l1(10e-10))(l6)
l8 = Conv2DTranspose(256, (2,2), strides=(2,2))(l7)

l9 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu',
            activity_regularizer = regularizers.l1(10e-10))(l8)

l10 = Conv2D(128, (3, 3), padding = 'same', activation = 'relu',
             activity_regularizer = regularizers.l1(10e-10))(l9)     # 2  /   2 

l11 = add([l5, l10])
l12 = Conv2DTranspose(128, (2,2), strides=(2,2))(l11)

l13 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu',
             activity_regularizer = regularizers.l1(10e-10))(l12)

l14 = Conv2D(64, (3, 3), padding = 'same', activation = 'relu',
             activity_regularizer = regularizers.l1(10e-10))(l13)

l15 = add([l14, l2])
postUpsampling= Conv2DTranspose(64, (4,4), strides=(4,4))(l15)

decoded = Conv2D(3, (3, 3), padding = 'same', 
                 activation = 'relu', activity_regularizer = regularizers.l1(10e-10))(postUpsampling)


autoencoder = Model(input_img, decoded)

autoencoder.summary()

# Create an Adam optimizer with a specific learning rate
adam_optimizer = Adam(learning_rate=0.0001)

autoencoder.compile(optimizer=adam_optimizer, loss='mean_squared_error', metrics=['accuracy'])

x_train_low=Resolution("../DIV2K/DIV2K_train_LR_bicubic/X4/",128)

x_train_high=Resolution("../DIV2K/DIV2K_train_HR/",512)

x_valid_low=Resolution("../DIV2K/DIV2K_valid_LR_bicubic/X4/",128)

x_valid_high=Resolution("../DIV2K/DIV2K_valid_HR/",512)

print(x_train_low.shape)
print(x_train_high.shape)
print(x_valid_low.shape)
print(x_valid_high.shape)

dpi = 300
figsize = (1920/dpi, 1080/dpi)  # Setting figure size

plt.figure(figsize=figsize)

# For the low resolution image
ax1 = plt.subplot(1, 2, 1)  # Change to a 1x2 layout for side by side
plt.imshow(x_train_low[5])
plt.title("Low Resolution(Bicubic_x4)")
ax1.axis('off')  # Turn off the axis

# For the high resolution image
ax2 = plt.subplot(1, 2, 2)  # Change to a 1x2 layout for side by side
plt.imshow(x_train_high[5])
plt.title("High Resolution")
ax2.axis('off')  # Turn off the axis

plt.savefig(f'{folder_path}/lowvhigh.png', dpi=dpi)
plt.close()  # Close the current figure

callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)

history = autoencoder.fit(
    x_train_low, x_train_high,
    epochs=200,
    batch_size=16,
    shuffle=True,
    callbacks=[callback],
    validation_data=(x_valid_low, x_valid_high)
)


loss=history.history["loss"]
val_loss=history.history["val_loss"]
epochs=range(1,len(loss)+1)
plt.figure(figsize=figsize)
plt.plot(epochs,loss,'y',label="training loss")
plt.plot(epochs,val_loss,'r',label="validation loss")
plt.title("training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig(f'{folder_path}/loss.png', dpi=dpi)
plt.close()  # Close the figure after saving

acc=history.history["accuracy"]
val_acc=history.history["val_accuracy"]
epochs=range(1,len(loss)+1)
plt.figure(figsize=figsize) 
plt.plot(epochs,acc,'y',label="training acc")
plt.plot(epochs,val_acc,'r',label="validation acc")
plt.title("training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.savefig(f'{folder_path}/accuracy.png', dpi=dpi)
plt.close()  # Close the figure after saving

# Save the trained model
autoencoder.save('autoencoder_x4.h5')
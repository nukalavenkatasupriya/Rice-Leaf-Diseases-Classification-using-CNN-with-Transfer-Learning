from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization,GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input,decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
import matplotlib.pylab as plt
import numpy as np

img_height,img_width=(224,224)
batch_size=12
train_data_dir="Data/train"
test_data_dir="Data/test"
print("====================================================")

train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   validation_split=0.4)

train_generator = train_datagen.flow_from_directory(train_data_dir,
                                                target_size=(img_height,img_width),
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                subset='validation')



test_generator = train_datagen.flow_from_directory(test_data_dir,
                                                target_size=(img_height,img_width),
                                                batch_size=1,
                                                class_mode='categorical',
                                                subset='validation')
x,y=test_generator.next()
x.shape

base_model=ResNet50(include_top=False,weights='imagenet')
x=base_model.output
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x)
predictions=Dense(train_generator.num_classes,activation='softmax')(x)
model=Model(inputs=base_model.input,outputs=predictions)

for layer in base_model.layers:
    layer.trainable=False

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
history=model.fit(train_generator,
          epochs=15,
        validation_data=test_generator)

model.save(r"D:\RICE LEAF DISEASE DETECTION\ResNet50-5.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='validation accuracy',color='green')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(r"D:\RICE LEAF DISEASE DETECTION\resNet-3.png")
plt.show()




#test_loss,test_acc=model.evaluate(test_generator,verbose=5)
print("**************************")

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
#from tensorflow.keras import Model
from tensorflow.keras.applications.vgg16 import VGG16

train_dir="Data/train"
test_dir="Data/test"
train_augmentation = ImageDataGenerator(
                                    rescale=1./255,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True,)

train_gen = train_augmentation.flow_from_directory(train_dir,
                                                target_size=(128,128),
                                                batch_size=12,
                                                class_mode='categorical')

validation_augmentation=ImageDataGenerator(
                            rescale=1./255
                            )
validation_generator = validation_augmentation.flow_from_directory(test_dir,
                                                target_size=(128,128),
                                                batch_size=12,
                                                class_mode='categorical',
                                               )

conv_base=VGG16(input_shape=(128,128,3),include_top=False,weights='imagenet')

conv_base.summary()

for layer in conv_base.layers:
    layer.trainable=False

model=Sequential()
model.add(conv_base)
model.add(layers.Flatten())

model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(4,activation='softmax'))

model.summary()

conv_base.trainable=True
set_trainable=False
for layer in conv_base.layers:
    if layer.name=='blocks_conv1':
        set_trainable=True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

history=model.fit(train_gen,
                  steps_per_epoch=8,
                  epochs=10,
                  verbose=1,
                  validation_data=validation_generator)

model.save("D:\RICE LEAF DISEASE DETECTION\VGG16Model2-5.h5")

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history['accuracy'],'r',label='validation accuracy',color='green')
plt.xlabel('# epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig(r"D:\RICE LEAF DISEASE DETECTION\vgg-3.png")
plt.show()



vgg_acc=history.history['val_accuracy'][-1]
print(vgg_acc)





import matplotlib.pyplot as plt
import time
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.utils import np_utils

# This will do preprocessing and realtime data augmentation:
datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    zca_epsilon=1e-06,  # epsilon for ZCA whitening
    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
    # randomly shift images horizontally (fraction of total width)
    width_shift_range=0.1,
    # randomly shift images vertically (fraction of total height)
    height_shift_range=0.1,
    shear_range=0.,  # set range for random shear
    zoom_range=0.,  # set range for random zoom
    channel_shift_range=0.,  # set range for random channel shifts
    # set mode for filling points outside the input boundaries
    fill_mode='nearest',
    cval=0.,  # value used for fill_mode = "constant"
    horizontal_flip=True,  # randomly flip images
    vertical_flip=False,  # randomly flip images
    # set rescaling factor (applied before any other transformation)
    rescale=None,
    # set function that will be applied on each input
    preprocessing_function=None,
    # image data format, either "channels_first" or "channels_last"
    data_format=None,
    # fraction of images reserved for validation (strictly between 0 and 1)
    validation_split=0.0
)

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
# y_train = np_utils.to_categorical(y_train, 10)
# y_test = np_utils.to_categorical(y_test, 10)
x_train = x_train.astype('float64')
x_test = x_test.astype('float64')

for index, image in enumerate(x_train):
    total = 0
    # plt.imshow(image.astype(np.uint8))
    # plt.show()
    # construct the actual Python generator
    print("[INFO] generating images..." + str(index))
    imageGen = datagen.flow(image.reshape((1, 32, 32, 3)), batch_size=1, save_to_dir="../cifar10_augmented/class_" + str(y_train[index][0]),
                          save_prefix="aug_" + str(index), save_format="png")

    # loop over examples from our image data augmentation generator
    for image in imageGen:
        # increment our counter
        total += 1
        # plt.imshow(image.reshape((32, 32, 3)).astype(np.uint8))
        # plt.show()
        # if we have reached the specified number of examples, break
        # from the loop
        if total == 20:
            break


import os

from keras import models,layers
from keras.applications import VGG16
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator

from optimizer.KerasOptimizer import KerasOptimizer
from optimizer.Strategy import Strategy



def custom_model():
    vgg = VGG16(weights='imagenet', classes=3, include_top=False, input_shape=(224, 224, 3))

    # Create the model
    model = models.Sequential()

    # Add the vgg convolutional base model
    model.add(vgg)

    model.add(layers.Flatten())
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(3, activation='softmax'))

    #model.summary()

    return model

def train_model( hyperparams):
    train_data = "C:/Users/bkmksta/Desktop/PlantDiseaseDetection/datasetsingle/Train"
    validation_data = "C:/Users/bkmksta/Desktop/PlantDiseaseDetection/datasetsingle/Validation"

    batch_size = hyperparams["batch_size"]
    EPOCHS = hyperparams["epochs"]
    INIT_LR = hyperparams["learning_rate"]
    width = 224
    height = 224

    # this is the augmentation configuration we will use for training
    train_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a generator that will read pictures found in
    # subfolers of 'data/train', and indefinitely generate
    # batches of augmented image data
    train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(width, height),
        color_mode='rgb',
        batch_size=batch_size,
        shuffle=True,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

    # this is the augmentation configurationw e will use for testing:
    # only rescaling
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    # this is a similar generator, for test data
    validation_generator = validation_datagen.flow_from_directory(
        validation_data,
        target_size=(width, height),
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=False,
        class_mode='categorical')

    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)

    # initialize the model
    print("[INFO] compiling model...")
    model = custom_model()
    opt = SGD(lr=INIT_LR, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=EPOCHS,
        shuffle=True,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)

    return history.history['accuracy']


def run():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = current_dir + '/../dataset/mnist.npz'
    optimizer = KerasOptimizer(dataset)

    optimizer.select_optimizer_strategy(Strategy.MAXIMIZE)
    optimizer.add_hyperparameter('batch_size', [16, 32, 64])
    optimizer.add_hyperparameter('epochs', [1, 2, 3])
    optimizer.add_hyperparameter('learning_rate', [0.001, 0.01, 0.1])
    optimizer.show_graph_on_end()
    optimizer.run(custom_model, train_model)

if __name__ == '__main__':
    run()
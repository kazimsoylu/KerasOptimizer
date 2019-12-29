import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from optimizer.KerasOptimizer import KerasOptimizer
from optimizer.Strategy import Strategy
import os


def custom_model():
    num_classes = 10
    input_shape = (28, 28, 1)

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # model.summary()
    opt = SGD(lr=0.01, decay=1e-6, momentum=0.9)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=opt,
                  metrics=['accuracy'])

    return model


current_dir = os.path.dirname(os.path.abspath(__file__))
dataset = current_dir + '/../dataset/mnist.npz'
optimizer = KerasOptimizer(dataset)

optimizer.select_optimizer_strategy(Strategy.MAXIMIZE)
optimizer.add_hyperparameter('batch_size', [16, 32, 64])
optimizer.add_hyperparameter('epochs', [1, 2, 3])
optimizer.add_hyperparameter('learning_rate', [0.001, 0.01, 0.1])
optimizer.show_graph_on_end()
optimizer.run(custom_model)

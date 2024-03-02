import visualkeras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.utils import plot_model

num_classes = 4
# Create a Sequential model
model = Sequential()
model.add(Conv2D(32, (8, 8), activation="relu", input_shape=(512, 512, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (8, 8), activation="relu"))
model.add(Conv2D(64, (4, 4), activation="relu"))
model.add(Conv2D(64, (6, 6), activation="relu"))
model.add(Conv2D(64, (2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(84, (4, 4), activation="relu", strides=(2, 2)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(56, (2, 2), activation="relu"))
model.add(Conv2D(34, (5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Dropout(0.3))
model.add(Conv2D(96, (2, 2), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dense(94, activation="relu"))
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer="adam")
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)



visualkeras.layered_view(model).show() # display using your system viewer
visualkeras.layered_view(model,to_file='output.png',legend=True) # write to disk# write and show

visualkeras.layered_view(model,legend=True)
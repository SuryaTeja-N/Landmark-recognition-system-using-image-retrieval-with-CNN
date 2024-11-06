import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load preprocessed data
images = np.load('images.npy')
labels = np.load('labels.npy')

print(f'Loaded {images.shape[0]} images with shape {images.shape[1:]}')
print(f'Loaded {labels.shape[0]} labels with shape {labels.shape[1]}')

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Load ResNet50 with pre-trained weights from ImageNet
# We don't include the top layer, as we want to add our custom classifier layer for landmark recognition
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Adding custom layers on top of ResNet50
# First, we flatten the output from ResNet50, then add a Dense layer with 128 units
# Finally, we add the output layer with units equal to the number of classes and 'softmax' activation
x = Flatten()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(labels.shape[1], activation='softmax')(x)

# Creating the new model that consists of the pre-trained ResNet50 and the custom layers
model = Model(inputs=base_model.input, outputs=output)

# Unfreezing some of the deeper layers to fine-tune the model for our specific dataset
# We'll leave the first few layers frozen and only train the latter half
# This will allow the model to adapt better to the specific features of the landmark dataset
for layer in base_model.layers[:100]:  # Freeze the first 100 layers
    layer.trainable = False
for layer in base_model.layers[100:]:  # Unfreeze the remaining layers
    layer.trainable = True

# Compiling the model
# Using Adam optimizer with a lower learning rate of 0.00001 to ensure that the changes made during fine-tuning are gradual
model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

# Training the model
# We train for 15 epochs now since we are fine-tuning some layers, which will require more iterations to adjust properly
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=15, batch_size=32)

# Save the new model
model.save('landmark_recognition_resnet50_finetuned.h5')

# Evaluating the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Accuracy: {val_accuracy:.2f}')

# Plotting the training history to visualize how the training and validation accuracy evolve
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Comments:
# 1. We're unfreezing some deeper layers in the ResNet50 model to let them train on our specific dataset.
# 2. This should help improve accuracy since the model will now adapt to landmark-specific features more effectively.
# 3. We're using a lower learning rate to make gradual, subtle updates to the pre-trained weights.
# 4. This approach balances between leveraging pre-trained features and adapting them to our dataset.
import numpy as np
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt

# Load training and validation datasets
X_data = np.load('images.npy')
y_data = np.load('labels.npy')

# Preprocess images
X_data = preprocess_input(X_data)

# Inspect labels
print("y_data.shape before any processing:", y_data.shape)

# Adjust labels
if len(y_data.shape) > 1 and y_data.shape[1] == 1:
    y_data = y_data.reshape(-1)
elif len(y_data.shape) > 1:
    y_data = y_data[:, 0]  # Take the first label per sample

# Ensure labels are integer encoded
y_data = y_data.astype(int)

# Check the number of classes
num_classes = np.max(y_data) + 1  # Assuming labels start from 0
print("Number of classes:", num_classes)

# One-hot encode the labels
y_data = to_categorical(y_data, num_classes=num_classes)
print("y_data.shape after to_categorical:", y_data.shape)

# Use 95% of the data for training and 5% for validation
split_index = int(0.95 * len(X_data))
X_train = X_data[:split_index]
y_train = y_data[:split_index]
X_val = X_data[split_index:]
y_val = y_data[split_index:]

# Data augmentation (less aggressive)
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_datagen.fit(X_train)

# Load EfficientNetB0 base model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers
base_model.trainable = False

# Add custom layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
output = Dense(num_classes, activation='softmax')(x)

# Define the complete model
model = Model(inputs=base_model.input, outputs=output)

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callback to reduce learning rate if the validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=1, min_lr=0.00001)

# Train the model (initial training with frozen base)
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), 
                    validation_data=(X_val, y_val), epochs=5, callbacks=[reduce_lr])

# Unfreeze some layers of the base model for fine-tuning
for layer in base_model.layers[-20:]:
    layer.trainable = True

# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training (fine-tuning)
history_fine = model.fit(train_datagen.flow(X_train, y_train, batch_size=32), 
                         validation_data=(X_val, y_val), epochs=5, callbacks=[reduce_lr])

# Combine history
for key in history.history.keys():
    history.history[key] += history_fine.history[key]

# Plotting training and validation accuracy and loss
plt.figure(figsize=(12, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')

plt.tight_layout()
plt.show()

# Save the final model
model.save('landmark_recognition_efficientnet_improved.h5')

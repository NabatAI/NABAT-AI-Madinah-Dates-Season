import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Paths
original_dir = '/Users/abdulazizashy/Documents/images'  # Directory with all images
train_dir = '/Users/abdulazizashy/Documents/tomato_train'
test_dir = '/Users/abdulazizashy/Documents/tomato_test'

# Create directories if they don't exist
os.makedirs(os.path.join(train_dir, 'tomato'), exist_ok=True)
os.makedirs(os.path.join(test_dir, 'tomato'), exist_ok=True)

# List all images
all_images = [f for f in os.listdir(original_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

if not all_images:
    print("No images found in the original directory.")
    exit()

# Split the data
train_files, test_files = train_test_split(all_images, test_size=0.2, random_state=42)

# Function to move images to the appropriate directory
def move_images(file_list, target_dir):
    for file_name in file_list:
        src = os.path.join(original_dir, file_name)
        dst = os.path.join(target_dir, file_name)
        shutil.move(src, dst)

# Move images to train and test directories
move_images(train_files, os.path.join(train_dir, 'tomato'))
move_images(test_files, os.path.join(test_dir, 'tomato'))

# Create ImageDataGenerators
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary'
)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Adjust steps_per_epoch and validation_steps
steps_per_epoch = train_generator.samples // 32
validation_steps = test_generator.samples // 32

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=test_generator,
    validation_steps=validation_steps
)

# Print a summary of the training
print("Training complete!")
print("Model summary:")
model.summary()

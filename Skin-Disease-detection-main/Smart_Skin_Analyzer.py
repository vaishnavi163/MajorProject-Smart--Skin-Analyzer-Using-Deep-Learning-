#!/usr/bin/env python
# coding: utf-8

# In[15]:


import os
import glob
import numpy as np
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical


# In[16]:


file_path = r"C:\Users\vaish\Desktop\Major Project\Project\Dataset"


# In[17]:


name_class = os.listdir(file_path)
print("Classes:", name_class)


# In[21]:


filepaths = list(glob.glob(file_path + '/**/*.*'))
print("Total images:", len(filepaths))


# In[23]:


labels = [os.path.basename(os.path.dirname(x)) for x in filepaths]


# In[25]:


data = pd.DataFrame({"Filepath": filepaths, "Label": labels})
data = data.sample(frac=1).reset_index(drop=True)


# In[27]:


plt.figure(figsize=(10, 5))
sns.barplot(x=data["Label"].value_counts().index, y=data["Label"].value_counts())
plt.xticks(rotation=90)
plt.title("Class Distribution")
plt.show()


# In[33]:


# Train-Test Split
train, test = train_test_split(data, test_size=0.25, random_state=42, stratify=data["Label"])


# In[35]:


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,  # Augmentation
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)


# In[37]:


# Data Generators
train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    x_col='Filepath',
    y_col='Label',
    target_size=(100, 100),
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42
)

valid_gen = test_datagen.flow_from_dataframe(
    dataframe=test,
    x_col='Filepath',
    y_col='Label',
    target_size=(100, 100),
    class_mode='categorical',
    batch_size=32,
    shuffle=False,
    seed=42
)


# In[39]:


# Load Pretrained Model (ResNet50)
pretrained_model = ResNet50(input_shape=(100, 100, 3), include_top=False, weights='imagenet', pooling='avg')
pretrained_model.trainable = False  # Freeze layers


# In[41]:


# Custom Model on Top of ResNet50
x = Dense(128, activation='relu')(pretrained_model.output)
x = Dense(128, activation='relu')(x)
outputs = Dense(len(name_class), activation='softmax')(x)  # Adjust for 21 classes
model = Model(inputs=pretrained_model.input, outputs=outputs)


# In[43]:


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


# In[45]:


my_callbacks = [EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True)]


# In[47]:


# Train Model
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=15,
    callbacks=my_callbacks
)


# In[49]:


# Save Model
model.save("model_resnet50.h5")


# In[51]:


# Plot Accuracy & Loss
pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot(title="Accuracy")
plt.show()

pd.DataFrame(history.history)[['loss', 'val_loss']].plot(title="Loss")
plt.show()


# In[53]:


# Evaluate Model
results = model.evaluate(valid_gen, verbose=0)
print("Test Loss: {:.5f}".format(results[0]))
print("Test Accuracy: {:.2f}%".format(results[1] * 100))


# In[55]:


steps = valid_gen.samples // valid_gen.batch_size
pred_probs = model.predict(valid_gen, steps=steps)
print(f"Raw Predictions Shape: {pred_probs.shape}")
pred_labels = np.argmax(pred_probs, axis=1)


# In[59]:


# y_test_labels = test["Label"].values[:len(pred_labels)]
# print(f"Test labels count: {len(y_test_labels)}")
# print(f"Predicted labels count: {len(pred_labels)}")
y_test_labels = test["Label"].values[:len(pred_labels)]
print(f"Test labels count: {len(y_test_labels)}")
print(f"Predicted labels count: {len(pred_labels)}")


# In[61]:


# # Get the label-to-index mapping from the training generator
# label_to_index = train_gen.class_indices

# # Create a reverse mapping from index to label
# index_to_label = {v: k for k, v in label_to_index.items()}

# # Convert y_test_labels from string labels to numeric labels
# y_test_numeric = [label_to_index[label] for label in y_test_labels]

# # Print to verify the mapping
# print(f"First 10 true labels (numeric): {y_test_numeric[:10]}")
# print(f"First 10 predicted labels (numeric): {pred_labels[:10]}")

# # Generate classification report
# print(classification_report(y_test_numeric, pred_labels))


# Get the label-to-index mapping from the training generator
label_to_index = train_gen.class_indices

# Convert y_test_labels from string labels to numeric labels
y_test_numeric = [label_to_index[label] for label in y_test_labels]

# Print to verify the mapping
print(f"First 10 true labels (numeric): {y_test_numeric[:10]}")
print(f"First 10 predicted labels (numeric): {pred_labels[:10]}")

# Generate classification report
print(classification_report(y_test_numeric, pred_labels))


# In[83]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

def predict_image(image_path):
    # Load Image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Fix color format
    img = cv2.resize(img, (100, 100))  # Resize image to match input size
    
    # Preprocess Image
    img_array = np.expand_dims(img, axis=0)
    img_array = preprocess_input(img_array)

    # Load Model & Predict
    loaded_model = load_model('model_resnet50.h5')
    prediction = loaded_model.predict(img_array)[0] * 100  # Get prediction probabilities

    # Get Prediction Label
    max_index = np.argmax(prediction)
    predicted_label = name_class[max_index]  # name_class should be a list or dictionary

    # Display Image & Result
    plt.imshow(img)
    plt.title(f"Prediction: {predicted_label} ({prediction[max_index]:.2f}%)")
    plt.axis("off")
    plt.show()

    return predicted_label


# Example Prediction
image_path = r"C:\Users\vaish\Desktop\Major Project\Project\Acne (1).png" # Use raw string for file path
predicted_class = predict_image(image_path)
print(f"Predicted Class: {predicted_class}")


# In[127]:


'''
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input

# Load the model once, outside the function to avoid loading repeatedly
loaded_model = load_model('model_resnet50.h5')

def predict_image(image_path, true_label):
    # Load Image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Fix color format
    img_resized = cv2.resize(img_rgb, (100, 100))  # Resize image to match input size
    
    # Preprocess Image
    img_array = np.expand_dims(img_resized, axis=0)
    img_array = preprocess_input(img_array)

    # Predict the image class
    prediction = loaded_model.predict(img_array)[0] * 100  # Get prediction probabilities

    # Get Prediction Label
    max_index = np.argmax(prediction)
    predicted_label = name_class[max_index]  # name_class should be a list or dictionary

    # Display True Image & Predicted Result
    plt.figure(figsize=(10, 5))  # Set figure size

    # True Image on Left
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    plt.title(f"True Label: {true_label}")  # Actual true label
    plt.axis("off")

    # Predicted Image on Right
    plt.subplot(1, 2, 2)
    plt.imshow(img_resized)
    plt.title(f"Predicted Label: {predicted_label} ({prediction[max_index]:.2f}%)")
    plt.axis("off")

    plt.show()

    return predicted_label


# Example: If you have a dictionary or a list for true labels:
true_labels_dict = {
    "disease1.jpg": "Disease A",  # Replace with actual true labels
    "disease2.jpg": "Disease B",
    # Add other images and their true labels
}

# Example Prediction for a specific image
image_path = r"C:\Users\vaish\Desktop\Major Project\Project\Hairloss (27).jpg"  # Use raw string for file path
image_name = image_path.split("\\")[-1]  # Extract image name from path
true_label = true_labels_dict.get(image_name, "Unknown")  # Get the true label for the image
predicted_class = predict_image(image_path, true_label)
print(f"Predicted Class: {predicted_class}")
'''



# In[ ]:





import os
import cv2
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input, MaxPooling2D, Dropout, BatchNormalization, Reshape
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

warnings.filterwarnings("ignore")

"""## Setting Random Seed for Reproducibility

When working with machine learning models, especially neural networks, there is a degree of randomness involved:
- Initialization of weights
- Data shuffling
- Random transformations during data augmentation
- GPU parallelism, etc.

This randomness can lead to **slightly different results each time you train a model**, even if the data and architecture are the same. To ensure consistent and reproducible results, we **set a fixed random seed**.

"""

# Impose seed for the reproducibility
seed = 42
keras.utils.set_random_seed(seed)
np.random.seed(seed)
Target_size = 32

path = "E:\\Traffic Sign Image Classification"
train_path = path + "/Train"
test_path = path + "/Test"
test_csv_path = path + "/Test.csv"

"""## 📁 Loading Image Paths and Labels for Training

In this section, we are preparing our training dataset by:
1. Reading the folder structure of the training dataset
2. Building a list of image file paths
3. Assigning corresponding class labels based on folder names

This is a **common approach in image classification tasks** where each class is stored in its own subfolder.


"""

# List of images path to load them
img_list = []
label_train_list = []
folders = os.listdir(train_path)  # directory of the folder containing all the train images

for folder in folders:  # access singularly the 43 folders contained in the train folder
    for img in os.listdir(train_path + "/"+folder): # access singlularly every image contained in the folder
        img_list.append  (train_path +  "/"+ folder+"/"+img) # creates an array with all the images in order of access
        label_train_list.append(folder)                      # creates at the same time an array with the number of the folder accessed (corresponds to the class)

# Sorted numpy array label of the corresponding image
label_train_list = np.array(label_train_list, dtype=int)

# Dictionary used for the print of images with the number of the class and the corresponding name
label_name={
0: "Speed Limit 20",
1: "Speed Limit 30",
2: "Speed Limit 50",
3: "Speed Limit 60",
4: "Speed Limit 70",
5: "Speed Limit 80",
6: "End of a Speed Limit 80",
7: "Speed Limit 100",
8: "Speed Limit 120",
9: "No overtaking",
10: "No overtaking by trucks",
11: "Crossroads",
12: "Priority Road",
13: "Give way",
14: "Stop",
15: "All vehicles prohibited in both directions",
16: "No trucks",
17: "No Entry",
18: "Other Hazards",
19: "Curve to left",
20: "Curve to right",
21: "Double curve, first to the left",
22: "Uneven Road",
23: "Slippery Road",
24: "Road Narrows Near Side",
25: "Roadworks",
26: "Traffic lights",
27: "No pedestrians",
28: "Children",
29: "Cycle Route",
30: "Be careful in winter",
31: "Wild animals",
32: "No parking",
33: "Turn right ahead",
34: "Turn left ahead",
35: "Ahead Only",
36: "Proceed straight or turn right",
37: "Proceed straight or turn left",
38: "Pass onto right",
39: "Pass onto left",
40: "Roundabout",
41: "No overtaking",
42: "End of Truck Overtaking Prohibition",
}

"""## Dataset Visualization – Histogram of Class Distribution

Before training a model, it's crucial to understand the **distribution of your classes**. If a dataset is imbalanced, it may bias the model during training. This histogram provides a visual summary of how many images exist per class.

"""

# Dataset visualization -> print of an histogram

df=pd.DataFrame({"img":img_list,"label":label_train_list})     # define a type of data formed by couples
df["label"]=df["label"].astype(int)                            # changes the type of the label column
df["actual_label"]=df["label"].map(label_name)                 # creates a new column with the name of the classes mapped exaclty in the row of the corresponding clas

plt.figure(figsize=(23,8))                                            # create the images

ax = sns.countplot( x = df["actual_label"],                      # on the x axis the names of the classes
                   palette = "viridis",                               # how to color the histogram
                     order = df['actual_label'].value_counts().index) # counts the number of elements with the same name of class

for p in ax.containers:
    ax.bar_label(p, fontsize=12, color='black', padding=5)            # for each column of the histogram add a label of the corresponding value

plt.xticks(rotation=90);                                              # rotates the graphs of 90°

"""### Histogram Summary

- The histogram visualizes the number of images available for each traffic sign class.
- **Class imbalance** is evident:
  - Example: `Speed Limit: 50` has **2250 images**.
  - In contrast, `Speed Limit: 20` has only **210 images**.
- **Implications**:
  - The model may become biased toward overrepresented classes.
- **Possible Solutions**:
  - Apply **data augmentation** to increase samples for underrepresented classes.
  - Use **class weights** in model training to handle imbalance.
  - Evaluate the model using **per-class accuracy or F1-score** for fairness.

"""

# Function for loading and resizing the images
def process_image(img_path, label):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)  # decode image as RGB
    img = tf.image.resize(img, [Target_size, Target_size], method=tf.image.ResizeMethod.BILINEAR)
    img = img / 255.0  # Normalization
    return img, label

# Create a dataset with images and labels
dataset = tf.data.Dataset.from_tensor_slices((img_list, label_train_list)) #It takes each pair (image, label) and transforms it into a dataset of individual elements.

# Pre-processing of the dataset
dataset = dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

"""### 🖼️ Image Preprocessing with TensorFlow

#### 🔧 `process_image` Function
- **Purpose**: To load, decode, resize, and normalize an image.
- **Steps**:
  1. `tf.io.read_file(img_path)` – Reads the image from the given file path.
  2. `tf.image.decode_jpeg(..., channels=3)` – Decodes it as an RGB image.
  3. `tf.image.resize(..., [Target_size, Target_size])` – Resizes image to 32x32 pixels.
  4. `img / 255.0` – Normalizes pixel values to the range [0, 1].

#### 📦 Dataset Creation and Processing
- `tf.data.Dataset.from_tensor_slices((img_list, label_train_list))`
  - Creates a dataset of image paths and their labels.
- `.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)`
  - Applies `process_image` function to each image-label pair.
  - `AUTOTUNE` lets TensorFlow decide optimal parallelism for performance.

#### 💡 Why this matters?
- Efficient preprocessing pipeline helps in faster and scalable training.
- Normalization and resizing ensure uniformity across the dataset.

"""

# Model definition

l2_lambda = 0.0001  # value for the regularizer

model=Sequential()
model.add(Input(shape=(Target_size,Target_size,3)))

model.add(Conv2D(64,kernel_size=(3,3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(l2_lambda)))
model.add(Conv2D(128,kernel_size=(3,3),activation='relu',padding='same', kernel_regularizer=regularizers.l2(l2_lambda)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(256,kernel_size=(3,3),activation='relu',padding='same', kernel_regularizer=regularizers.l2(l2_lambda)))
model.add(Conv2D(512,kernel_size=(3,3),activation='relu',padding='same', kernel_regularizer=regularizers.l2(l2_lambda)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512,activation='relu'))
model.add(Dense(128,activation='relu'))

model.add(Dropout(0.2))
model.add(Dense(43,activation='softmax'))

"""### 🧠 CNN Model Architecture for Image Classification

This model is built using **Keras Sequential API** to classify traffic signs into 43 categories.

---

#### 📌 Regularization
- `l2_lambda = 0.0001`  
  Adds L2 regularization to prevent overfitting by penalizing large weights.

---

#### 🧱 Model Layers (Step-by-Step)

1. **Input Layer**
   - `Input(shape=(32, 32, 3))` – Accepts 32x32 RGB images.

2. **1st Convolution Block**
   - `Conv2D(64)` – 64 filters, 3x3 kernel, ReLU activation.
   - `Conv2D(128)` – 128 filters, 3x3 kernel, ReLU, with L2 regularization.
   - `MaxPooling2D(2x2)` – Downsamples spatial size.
   - `BatchNormalization()` – Stabilizes learning by normalizing activations.
   - `Dropout(0.25)` – Prevents overfitting by randomly disabling neurons.

3. **2nd Convolution Block**
   - `Conv2D(256)` – 256 filters.
   - `Conv2D(512)` – 512 filters.
   - `MaxPooling2D(2x2)`
   - `BatchNormalization()`
   - `Dropout(0.25)`

4. **Flatten Layer**
   - Converts 2D feature maps into a 1D vector for dense layers.

5. **Fully Connected Layers**
   - `Dense(512)` – Fully connected layer with ReLU.
   - `Dense(128)` – Another dense layer.
   - `Dropout(0.2)` – Additional dropout for regularization.

6. **Output Layer**
   - `Dense(43, activation='softmax')` – Outputs probabilities for 43 classes.

---

#### ✅ Summary
This is a **deep CNN model** with regularization techniques (L2, Dropout, BatchNorm) to improve generalization and performance on a multi-class classification task.

"""

model.summary()

"""### ⚙️ Model Compilation

This section prepares the model for training by defining the **optimizer**, **loss function**, and **evaluation metric**.

---

#### 🔧 Optimizer
- **Adam Optimizer** with:
  - `learning_rate = 1e-3` – Controls how fast the model updates weights.
  - `ema_momentum = 0.95` – Maintains an exponential moving average of model weights (optional performance boost during inference).

"""

# Model compilation
opt      = keras.optimizers.Adam(learning_rate = 1e-3, ema_momentum=0.95)     # Adam used as pìoptimizez
loss_fcn = keras.losses.SparseCategoricalCrossentropy()                       # Sparse Categorical Crossentropy used as loss function

model.compile(optimizer = opt,                                                # compiling the model
              loss      = loss_fcn,
              metrics   = ['accuracy'])

# Split of the dataset into train e validation set with a rate of 20%

split = 0.2

dataset_size = len(list(dataset))
train_size = int((1 - split) * dataset_size)

dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed) # shuffle the data

train_dataset = dataset.take(train_size)  # keep the first 80% of the dataset
val_dataset = dataset.skip(train_size)    # Skip the data kept for the training,set

"""### Data Augmentation Setup

To improve the model's generalization and make it more robust to variations in the input images, we apply **data augmentation** during training.

---

#### What is Data Augmentation?
Data augmentation artificially increases the size and diversity of the dataset by applying random transformations to the training images.


"""

# Data augmentation:

data_augmentation = Sequential([
                      keras.layers.RandomRotation(0.05),       # Randomly rotates the image ±5%
                      keras.layers.RandomTranslation(0.1,0.1), # Randomly shifts the image ±10% along width and height
                      keras.layers.RandomZoom(0.1)             # Randomly zooms in/out by ±10%
])

"""### 🚀 Data Augmentation & Optimized Data Loading

This section enhances the model training pipeline by applying **data augmentation** and optimizing **batching** and **data loading**.

---

#### 🔄 Data Augmentation on Training Set
- Data augmentation is applied **only on the training dataset** to introduce variability in the input images and prevent overfitting.

"""

batch_size = 128 # Use the same batch size for the mapping of the images and for the fitting of the model to avoid bottleneck and memory loss

# Apply of data augmentation only to the training images (there is no risk of misalignment).
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

# Data's batching and loading optimization
train_dataset = train_dataset.shuffle(buffer_size=int(train_dataset.cardinality()), seed=seed).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE) # no shuffle on validation set, we want to compare results
# method prefetch: while the GPU trains the current batch, the CPU preloads the next batch.

"""### 📉 Callbacks for Model Optimization

Two essential callbacks are set up to improve training performance: **Reduce Learning Rate on Plateau** and **Early Stopping**.

---
### 🔄 Reduce Learning Rate on Plateau

- **monitor** (`'val_loss'`):  
  The metric to watch; when it stops improving, the callback triggers.  
- **factor** (`0.5`):  
  Multiplier applied to the current learning rate when triggered. New LR = old LR × factor.  
- **patience** (`2`):  
  Number of epochs with no improvement after which the learning rate is reduced.  
- **min_delta** (`0.005`):  
  Minimum change in the monitored metric to qualify as an improvement. Changes smaller than this count as no improvement.  
- **min_lr** (`1e-7`):  
  Lower bound on the learning rate; ensures it never goes below this threshold.

---

### ⏸️ Early Stopping

- **monitor** (`'val_loss'`):  
  The metric to watch; when it stops improving, training may stop.  
- **patience** (`5`):  
  Number of epochs without improvement after which training is halted.  
- **min_delta** (`0.001`):  
  Minimum change in the monitored metric to qualify as an improvement; smaller changes are ignored.  
- **restore_best_weights** (`True`):  
  When training stops, model weights revert to the state they were in at the epoch with the best monitored metric.

---

### ⚙️ How They Work Together

- **ReduceLROnPlateau** gently lowers the learning rate when progress stalls, allowing finer adjustments to weights.  
- **EarlyStopping** halts training once no meaningful improvement is seen for a longer window, preventing overfitting and saving compute.  

By combining both, you get dynamic learning‐rate control plus an automated stop when further training yields diminishing returns.  

"""

# Two callbacks: reduce learning rate and early stopping(so number of epochs are raised)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, min_lr=1e-7, min_delta=0.005)
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, min_delta=0.001)

# Patience indicates the number of epochs without improvements before stopping the training.
# The patience of early stopping must be set greater than the patience of reduce_lr.
# Factor represents the reduction factor of the learning rate when the monitored metric (val_loss in your case) stops improving.
# The learning rate controls the step size that the optimization algorithm takes in the direction of minimizing the loss function.
# Val_loss is monitored to prevent overfitting.
# min_delta is set to indicate when to apply the callback.

# Example for the reduce learning rate: if there isn't a decrement on the validation loss of 0.005(=min_delta) with respect to the previous 2(=patience) epochs
# the learning rate will be halved

"""## Model Training (`model.fit`)

This step actually trains the neural network using your prepared datasets and callbacks.

---

### 🔑 Key Arguments

- **`train_dataset`**  
  The `tf.data.Dataset` providing training examples (images + labels), already shuffled, augmented, batched, and prefetched.

- **`validation_data=val_dataset`**  
  A separate `tf.data.Dataset` used to evaluate the model after each epoch. No augmentation or shuffling—only batched and prefetched.

- **`epochs=n_epochs`**  
  Total number of passes over the entire training set (here set to 10).  
  > Even though this is “higher” than you might need, **EarlyStopping** will halt training once validation loss stops improving.

- **`batch_size=batch_size`**  
  Number of samples processed before the model’s weights are updated. Matches the batch size used when building the datasets (128).

- **`callbacks=[reduce_lr, early_stopping]`**  
  - **`ReduceLROnPlateau`**: Lowers the learning rate if the validation loss plateaus.  
  - **`EarlyStopping`**: Stops training early and restores the best weights once validation loss ceases to improve.

- **`verbose=1`**  
  Controls logging during training:  
  - `0`: silent  
  - `1`: progress bar per epoch  
  - `2`: one line per epoch  

---

### 📊 Returned Object: `history`

- The `history` object records the values of loss and metrics (e.g., accuracy) on both the training and validation sets **for each epoch**.  
- You can access it via `history.history` to plot learning curves or find the epoch with the best validation accuracy.

"""

# Model Fit

n_epochs   = 10 # very high number of epochs but early stopping prevents useless epochs saving time
history = model.fit(train_dataset,
                    validation_data = val_dataset,
                    epochs          = n_epochs,
                    batch_size      = batch_size,
                    callbacks       = [reduce_lr, early_stopping],
                    verbose         = 1
                   )

model.save('my_model.h5')

model = load_model('my_model.h5')

"""## 📈 Visualizing & Summarizing Training History

After calling `model.fit()`, you get a `History` object that contains per‑epoch metrics. This snippet converts it to a plain dictionary, plots learning curves, prints final values, and identifies the best epoch.

"""

print(type(history))
if type(history) != dict:
    history = history.history

plt.figure(figsize=(20, 3))
for i, metric in enumerate(["accuracy", "loss"]):
    plt.subplot(1, 2, i + 1)
    plt.plot(history[metric])
    plt.plot(history["val_" + metric])
    plt.title("Model {}".format(metric))
    plt.xlabel("epochs")
    plt.ylabel(metric)
    plt.legend(["train", "val"])

# print fo the training values
print("Accuracy: ", history["accuracy"][-1])
print("Validation Accuracy: ", history["val_accuracy"][-1])

print("Loss: ", history["loss"][-1])
print("Validation Loss: ", history["val_loss"][-1])

# prints optimal capacity -> at which epoch the model had the higher validation accuracy
print("Optimal Capacity: ", np.argmax(history["val_accuracy"]) + 1)

"""## Loading Test Labels and Image Paths

Before evaluating the model on unseen data, we need to prepare the test set by reading a CSV file that contains the image file names and their corresponding class labels.

"""

# Obtaining test labels and correspective images' paths

test = pd.read_csv(test_csv_path)   # reads the csv file containing the labels in a data structor that has as columns ClassId and Path
test_labels=[]
test_path=[]
for i in range(len(test)):                          # for every row are saved (in order) in two arrays, the class id and the path of the image
    test_labels.append(test["ClassId"][i])
    test_path.append(path +'/' + test["Path"][i])

test_labels = np.array(test_labels,dtype=int)       # trasforms the labels in integers

"""## Preparing the Test Set

After loading the test image paths and labels, we need to preprocess and prepare the dataset for evaluation. This involves creating a `tf.data.Dataset` object from the test data, applying pre-processing transformations, batching, and optimizing for faster evaluation.

"""

# Create test set
test_dataset = tf.data.Dataset.from_tensor_slices((test_path, test_labels))

# Pre-processing of test set
test_dataset = test_dataset.map(process_image, num_parallel_calls=tf.data.AUTOTUNE)

# Batch to speed-up the test
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

"""## Model Prediction & Evaluation

This section shows how to use your trained model to make predictions on the test set and compute overall performance metrics.

"""

# Generate prediction of the model
predictions = model.predict(test_dataset)

# Get the class with the highest probability for each prediction
predictions = np.argmax(predictions, axis=-1)

# Compute the accuracy
loss, accuracy = model.evaluate(test_dataset)
#accuracy = accuracy_score(test_labels, predictions)

print(f"Test Loss: {loss:.4f},Test Accuracy: {accuracy:.4f}")

"""## Extracting Images and Labels from Test Dataset

In this step, we unbatch the `test_dataset`, which consists of batched images and their corresponding labels, to extract the individual image‐label pairs for evaluation or further processing.

"""

# Extract the images and the labels
test_dataset_unbatched = test_dataset.unbatch()
x_test, y_test = zip(*test_dataset_unbatched)
x_test = np.array(x_test)
y_test = np.array(y_test)

"""## Plotting Random Sample of 10 Images with True and Predicted Labels

In this section, we randomly select 10 images from the test dataset and display them along with their true labels and model's predicted labels.

"""

# Plot of 10 random images with corresponding prediction and true labels

plt.figure(figsize=(25, 25))

indices = random.sample(range(0, len(x_test)), 10)

j=0
for i in indices:
    j+=1
    plt.subplot(5, 5, j)
    plt.imshow(x_test[i])
    plt.title("True: {} {}\n Predicted: {} {}".format(y_test[i], label_name[y_test[i]],predictions[i], label_name[predictions[i]]) )
    plt.axis("off")


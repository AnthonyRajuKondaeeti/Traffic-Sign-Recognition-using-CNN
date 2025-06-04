# Traffic Sign Image Classification Project

## Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify traffic sign images into 43 distinct categories. The dataset consists of images organized by class in separate folders, and the model is trained to recognize various traffic signs with high accuracy. The project includes data preprocessing, augmentation, model training, evaluation, and visualization of results.

## Dataset
The dataset is sourced from a traffic sign image classification dataset stored at `"E:\\Traffic Sign Image Classification"`. It is divided into:
- **Training Set**: Images stored in subfolders under `Train/`, with each folder representing a class (0–42).
- **Test Set**: Image paths and labels provided in `Test.csv`.

### Class Distribution
- The dataset is imbalanced, with some classes (e.g., Speed Limit 50) having significantly more images than others (e.g., Speed Limit 20).
- A histogram of class distribution is generated to visualize this imbalance.

### Class Labels
The dataset includes 43 classes, such as:
- Speed Limit signs (20, 30, 50, etc.)
- No Overtaking, Stop, Give Way, etc.
- A dictionary (`label_name`) maps class IDs to descriptive names for visualization.

## Preprocessing
- **Image Loading**: Images are loaded using TensorFlow's `tf.data` API, decoded as RGB, resized to 32x32 pixels, and normalized to [0, 1].
- **Dataset Splitting**: The training dataset is split into 80% training and 20% validation sets.
- **Data Augmentation**: Applied to the training set only, including random rotations (±5%), translations (±10%), and zooms (±10%).

## Model Architecture
The CNN model is built using the Keras `Sequential` API with the following structure:
- **Input Layer**: Accepts 32x32x3 RGB images.
- **Convolutional Blocks**:
  - Block 1: Conv2D (64 filters) → Conv2D (128 filters) → MaxPooling → BatchNormalization → Dropout (0.25).
  - Block 2: Conv2D (256 filters) → Conv2D (512 filters) → MaxPooling → BatchNormalization → Dropout (0.25).
- **Fully Connected Layers**:
  - Flatten → Dense (512, ReLU) → Dense (128, ReLU) → Dropout (0.2) → Dense (43, softmax).
- **Regularization**: L2 regularization (`lambda=0.0001`) applied to convolutional layers to prevent overfitting.

## Training
- **Optimizer**: Adam with a learning rate of 0.001 and EMA momentum of 0.95.
- **Loss Function**: Sparse Categorical Crossentropy.
- **Callbacks**:
  - **ReduceLROnPlateau**: Reduces learning rate by a factor of 0.5 if validation loss doesn't improve by 0.005 for 2 epochs (minimum LR: 1e-7).
  - **EarlyStopping**: Stops training if validation loss doesn't improve by 0.001 for 5 epochs, restoring the best weights.
- **Epochs**: Set to 10, but early stopping typically halts training earlier.
- **Batch Size**: 128 for efficient training and evaluation.

## Evaluation
- The model is evaluated on the test set (`Test.csv`) with preprocessed images.
- **Metrics**: Test loss and accuracy are computed using `model.evaluate()`.
- **Visualization**: 10 random test images are displayed with their true and predicted labels.

## Results
- Training and validation accuracy/loss are plotted to analyze model performance across epochs.
- The "optimal capacity" (epoch with the highest validation accuracy) is reported.
- Example output (hypothetical):
  - Test Loss: 0.1234
  - Test Accuracy: 0.9567

## Dependencies
- Python 3.x
- TensorFlow/Keras
- NumPy
- Pandas
- Matplotlib
- Seaborn
- OpenCV (`cv2`)

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   ```
2. Install dependencies:
   ```bash
   pip install tensorflow numpy pandas matplotlib seaborn opencv-python
   ```
3. Update the `path` variable in the script to point to your dataset directory.

## Usage
1. Place the dataset in the specified directory (`E:\\Traffic Sign Image Classification`).
2. Run the script:
   ```bash
   python traffic_sign_classification.py
   ```
3. The script will:
   - Load and preprocess the dataset.
   - Train the CNN model.
   - Save the trained model as `my_model.h5`.
   - Evaluate the model on the test set.
   - Generate visualizations (class distribution histogram, training curves, and sample predictions).

## Project Structure
```
Traffic_Sign_Classification/
├── Train/                    # Training images in class subfolders
├── Test/                     # Test images
├── Test.csv                  # Test image paths and labels
├── traffic_sign_classification.py  # Main script
├── my_model.h5               # Trained model (generated after training)
└── README.md                 # This file
```

## Notes
- **Class Imbalance**: Consider using class weights or additional augmentation for underrepresented classes to improve performance.
- **Model Improvements**: Experiment with deeper architectures, different optimizers, or hyperparameter tuning for better accuracy.
- **Hardware**: A GPU is recommended for faster training, especially with large datasets.

## License
This project is licensed under the MIT License.

# Traffic Sign Image Classification

## Overview
This project implements a **Convolutional Neural Network (CNN)** using TensorFlow/Keras to classify traffic sign images into 43 distinct categories. The dataset is sourced from a traffic sign classification dataset, and the project includes data preprocessing, augmentation, model training, evaluation, and visualization of results.

## Dataset
The dataset is located at `E:\\Traffic Sign Image Classification` and consists of:
- **Training Set**: Images organized in subfolders under `Train/`, each representing one of 43 classes.
- **Test Set**: Image paths and labels provided in `Test.csv`.
- **Class Distribution**: The dataset is imbalanced, e.g., "Speed Limit 50" has 2250 images, while "Speed Limit 20" has 210 images. A histogram visualizes this distribution.

### Class Labels
The 43 classes include various traffic signs, such as:
- Speed Limit signs (20, 30, 50, etc.)
- No Overtaking, Stop, Give Way, Roundabout, etc.
- A dictionary (`label_name`) maps class IDs (0–42) to descriptive names.

## Preprocessing
- **Image Processing**:
  - Images are loaded using TensorFlow's `tf.data` API, decoded as RGB, resized to 32x32 pixels, and normalized to [0, 1].
- **Dataset Splitting**: The training dataset is split into 80% training and 20% validation sets.
- **Data Augmentation**: Applied to the training set with random rotations (±5%), translations (±10%), and zooms (±10%).

## Model Architecture
The CNN is built using the Keras `Sequential` API:
- **Input**: 32x32x3 RGB images.
- **Convolutional Blocks**:
  - Block 1: Conv2D (64 filters) → Conv2D (128 filters) → MaxPooling → BatchNormalization → Dropout (0.25).
  - Block 2: Conv2D (256 filters) → Conv2D (512 filters) → MaxPooling → BatchNormalization → Dropout (0.25).
- **Fully Connected Layers**: Flatten → Dense (512, ReLU) → Dense (128, ReLU) → Dropout (0.2) → Dense (43, softmax).
- **Regularization**: L2 regularization (`lambda=0.0001`) on convolutional layers.

## Training
- **Optimizer**: Adam (learning rate = 0.001, EMA momentum = 0.95).
- **Loss Function**: Sparse Categorical Crossentropy.
- **Callbacks**:
  - **ReduceLROnPlateau**: Reduces learning rate by 0.5 if validation loss doesn't improve by 0.005 for 2 epochs (minimum LR: 1e-7).
  - **EarlyStopping**: Stops training if validation loss doesn't improve by 0.001 for 5 epochs, restoring best weights.
- **Epochs**: Set to 10, with early stopping to prevent unnecessary training.
- **Batch Size**: 128.

## Evaluation
- The model is evaluated on the test set with preprocessed images.
- **Metrics**:
  - Test Loss: 0.2171
  - Test Accuracy: 0.9696
- **Visualization**: Training/validation accuracy and loss curves are plotted, and 10 random test images are displayed with true and predicted labels.

## Results
- **Training Performance**:
  - Final Training Accuracy: ~0.9918
  - Final Validation Accuracy: ~0.9986
  - Final Training Loss: ~0.0978
  - Final Validation Loss: ~0.0717
- **Optimal Capacity**: Epoch with highest validation accuracy (e.g., epoch 8).
- **Test Performance**: Achieved 96.96% accuracy on the test set.

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
1. Place the dataset in `E:\\Traffic Sign Image Classification` or update the `path` variable.
2. Run the script:
   ```bash
   python traffic_sign_classification.py
   ```
3. The script will:
   - Load and preprocess the dataset.
   - Train the CNN model and save it as `my_model.h5`.
   - Evaluate the model on the test set.
   - Generate visualizations (class distribution histogram, training curves, and sample predictions).

## Project Structure
```
Traffic_Sign_Classification/
├── Train/                    # Training images in class subfolders
├── Test/                     # Test images
├── Test.csv                  # Test image paths and labels
├── TICK (1).ipynb           # Jupyter notebook with the code
├── my_model.h5              # Trained model (generated after training)
└── README.md                # This file
```

## Notes
- **Class Imbalance**: Consider class weights or further augmentation for underrepresented classes.
- **Model Improvements**: Experiment with deeper architectures or hyperparameter tuning for better performance.
- **Hardware**: A GPU is recommended for faster training.
- **Model Saving Warning**: The script uses the legacy HDF5 format (`my_model.h5`). Consider using the native Keras format (`my_model.keras`) for future compatibility.

## License
This project is licensed under the MIT License.

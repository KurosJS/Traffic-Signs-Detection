# Traffic Signs Detection

This repository implements a traffic sign detection and classification system using **YOLOv11** for detection and a deep learning model for classification. The project integrates datasets from multiple sources, performs extensive augmentation, and processes bounding box data for training and evaluation.

---

## Features

- **Traffic Sign Detection**: Utilizes YOLOv11 for detecting traffic signs in real-time scenarios.
- **Deep Learning Classification**: A convolutional neural network (CNN) model is trained to classify detected signs into specific categories.
- **Custom Dataset Integration**:
  - German, Chinese, and additional custom traffic signs from multiple countries.
  - Manual annotation and bounding box data integration.
- **Data Augmentation**:
  - Augmentation applied to both training and testing datasets.
  - Bounding box updates for augmented images.
- **Meta Analysis**: Traffic signs are grouped by shape and color in `Meta.csv` for improved analysis.
- **Bounding Box Format Conversion**:
  - Bounding box data conversion between YOLO format and Pascal VOC format.

---

## Dataset Details

### Primary Dataset
The initial dataset used is the **GTSRB - German Traffic Sign Recognition Benchmark**, which includes **43 classes** of traffic signs.

### Custom Dataset Integration
1. **Chinese Road Signs**:
   - 19 classes selected from an existing dataset.
   - YOLOv11 was trained to detect "just a traffic sign" by merging the original 43 GTSRB classes into a single class.
   - Bounding box data was generated for unlabeled images, and the data was converted to match the GTSRB dataset's box data format.

2. **Additional Traffic Signs**:
   - **8 New Classes** were added:
     - **6 German signs**
     - **1 UK sign** (Falling Rocks)
     - **1 USA sign** (Hospital)
   - Images were collected via Google Images and Google Maps.
   - Annotation was performed manually using **CVAT**, exported in Pascal VOC format, and processed for bounding box updates.

---

## Deep Learning Model for Classification

The traffic sign classification system uses a convolutional neural network (CNN) implemented with TensorFlow/Keras.

### Model Architecture
```python
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=64, kernel_size=(5,5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.15))
model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.20))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(rate=0.25))
model.add(Dense(70, activation='softmax'))
```

### Model Summary
- **Input**: Processed traffic sign images.
- **Output**: Class probabilities for **70 categories**.
- **Loss Function**: `categorical_crossentropy`.
- **Optimizer**: `adam`.

### Training Process
The model is trained on augmented traffic sign images to classify them into 70 categories, including the newly added signs.

---

## Installation

### Clone the Repository
```bash
git clone https://github.com/KurosJS/Traffic-Signs-Detection.git
cd Traffic-Signs-Detection
```

### Install Dependencies
Ensure Python 3.8 or later is installed, then run:
```bash
pip install -r requirements.txt
```

---

## Workflow

1. **Dataset Preparation**:
   - Process raw images and bounding box data using provided scripts.
   - Apply data augmentation to enhance training and testing datasets.

2. **Model Training**:
   - Train the YOLOv11 detection model:
     ```bash
     yolo task=detect mode=train model=yolov11.pt data=dataset.yaml epochs=100 imgsz=640
     ```
   - Train the CNN classifier:
     ```python
     model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)
     ```

3. **Evaluation**:
   - Evaluate the detection model:
     ```bash
     yolo task=detect mode=val model=best.pt data=dataset.yaml
     ```
   - Evaluate the CNN classification model:
     ```python
     model.evaluate(X_test, y_test)
     ```

4. **Real-Time Detection**:
   - Use YOLOv11 for real-time detection and pass detected regions to the classifier for category prediction.

---

## Augmentation

The dataset is augmented using Albumentations:
- **Random Brightness and Contrast**
- **Noise Addition**
- **Scaling, Shifting, and Rotating**
- **Gaussian Blur**
- **Perspective Transformations**

Bounding box data is automatically updated for augmented images.

---

## Results

- **Traffic Sign Detection**: Employs YOLOv11 for real-time detection of traffic signs.
- **Deep Learning Classification**: A CNN classifies detected signs into 70 categories.
- **Custom Dataset Integration**:
  - Incorporates **Chinese Road Signs** dataset and extends it with additional custom traffic signs.
- **Bounding Box Conversion**:
  - Converts bounding box formats between YOLO and Pascal VOC.
- **Augmentation**:
  - Extensively augments training and testing datasets with updated bounding box information.
- **Meta Analysis**:
  - Traffic signs are grouped by shape and color in `Meta.csv`.


---

## Acknowledgments

- **GTSRB Dataset** for the foundational traffic sign dataset.
- **CVAT** for manual annotation of additional classes.
- **Albumentations** for augmentation support.
- **YOLOv11** and **TensorFlow/Keras** for detection and classification.
- **Chinese Road Signs Dataset**: Sourced from [this Kaggle dataset](https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification).
- **Google Images** and **Google Maps** for additional traffic sign images.
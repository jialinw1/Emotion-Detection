# Emotion Detection Project

![Emotion Detection.png](Emotion%20Detection.png)

## Prerequisites

The following packages need to be installed:

```bash
pip install tensorflow
pip install opencv-python
pip install numpy
pip install Pillow
pip install matplotlib
pip install scikit-learn
```

## Deeper CNN Model 

---

#### Layer 1: Convolutional Block

- **Conv2D Layer**: 64 filters, 3x3 kernel -> ReLU
- **Conv2D Layer**: 64 filters, 3x3 kernel -> ReLU
- **MaxPooling2D**: 2x2 pool size

---

#### Layer 2: Convolutional Block

- **Conv2D Layer**: 128 filters, 3x3 kernel -> ReLU
- **Conv2D Layer**: 128 filters, 3x3 kernel -> ReLU
- **MaxPooling2D**: 2x2 pool size

---

#### Layer 3: Convolutional Block

- **Conv2D Layer**: 256 filters, 3x3 kernel -> ReLU
- **Conv2D Layer**: 256 filters, 3x3 kernel -> ReLU
- **MaxPooling2D**: 2x2 pool size

---

#### Fully Connected Layer

- **Flatten**: Converts 2D feature maps into a 1D vector

---

#### Dense Layers

- **Dense Layer**: 512 units -> ReLU
- **Dropout**: 0.5 (50% of neurons randomly dropped)
- **Dense Layer**: (Number of classes) -> Softmax


## Image Datasets

Two datasets are used in this project:

1. **FER2013** for emotion detection.
2. **UTKFace** for age detection.

## Training Data

1. **Splitting the Dataset**:  
   - Run `AutoSepUTK.py` to split the UTKFace dataset into training and test sets.

2. **Training the Models**:  
   - Run `TrainModel.py` to train two separate models: one for emotion detection and one for age detection.

---

#### Note

This project is used for practice and achieves an accuracy of approximately **50%** due to the choice of image datasets.

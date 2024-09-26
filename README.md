# Emotion-Detection
The necessary packages need to be installed:<br>

    1. pip install tensorflow
    2. pip install opencv-python
    3. pip install numpy
    4. pip install Pillow
    5. pip install matplotlib
    6. pip install scikit-learn

<br>
** Deeper CNN Model **
Input Layer
|
|-- Conv2D Layer (64 filters, 3x3 kernel) -> ReLU
|-- Conv2D Layer (64 filters, 3x3 kernel) -> ReLU
|-- MaxPooling2D (2x2 pool size)
|
|-- Conv2D Layer (128 filters, 3x3 kernel) -> ReLU
|-- Conv2D Layer (128 filters, 3x3 kernel) -> ReLU
|-- MaxPooling2D (2x2 pool size)
|
|-- Conv2D Layer (256 filters, 3x3 kernel) -> ReLU
|-- Conv2D Layer (256 filters, 3x3 kernel) -> ReLU
|-- MaxPooling2D (2x2 pool size)
|
|-- Flatten (convert 2D feature maps into 1D vector)
|
|-- Dense Layer (512 units) -> ReLU
|-- Dropout (0.5)
|-- Dense Layer (number of classes) -> Softmax



<br>
Using *'fer2013'* as emotion and *'utkface-new'* as age image set train two different model which these image set can be found in kaggle.
<br>
Run AutoSepUTK.py split the train file and test file.
<br>
Run TrainModel.py train as two separate models.
<br>
This project is used for practice and has an accuary of 50% due to the selection of picture data sets.

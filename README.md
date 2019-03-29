# FaceAligner
A function that can help you align faces in the images for recognition and other types of uses.

### Prerequistics
```
pip install mtcnn
pip install python-opencv
pip install numpy
```

### Usage

``` 
aligner = FaceAligner() 
output = aligner.align(image) # RGB image
```

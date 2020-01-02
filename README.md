# Count-Faces-in-Image
A self learning project hosted by analytics vidhya where a machine learning model was used to detect and count mutiple faces in an image.

## Datasets
2 datasets were provided by Analytics Vidhya. 1st had the image name and the corresponding head count. Another had the image name and the corresponding bboxes. The second dataset had multiple rows of the same sample image because each image had mutiple faces and thereby multiple bboxes.

## I/O
It was a multi output problem where not only the headcount but also the bboxes around each face had to be detected therefore keras funcional model was used instead of sequential model. 

## Pre-Processing
Since each image can have varying number of faces and varying bboxes for the same and CNN model can only process fixed size I/O problems therefore default dict was used with keys as the imagename and values as list of bboxes in that image. The total labels was the unique values in the dictionary.This was done by adding the vaues to a set. After succesful train test validation split of the dataset loop was run for each item in the values and the same key was appended to a lsit. In this way a fixed size input and fixed size output was generated. one hot labels for each label was created.

## Model Training
A shallow CNN model was used with a convolutional base of 3 layers and max pooling was also employed. It was multi output problem. one output was the bbox coordinates and another was the headcount in the images. Both regression and classification was done parallely by the same model. For the bbox predcition, linear activation function was used and for head count prediction softmax function was used at the output layer. Loss function used was distance_loss and categorical cross entropy. IOU and f-measure were used as metric correspondigly for the two tasks. Model was trained for 50 epochs and model checkpointing was also employed to reduce the overfiting.

## Model Evaluation
A f1-score of 62% was registered for headcount prediction and IOU score of 48% was registered for face detection.

 

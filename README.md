# Facial-Expression-Recognizer

Artificial Inteligence assignment. 

## How does it works?
XCeption (Extreme Inception) Convolutional Neural Network (CNN) module.
Depthwise Separable Convolution with two steps:
- Depthwise convolution: Applies a separate convolution to each channel of the input image.
- Pointwise convolution: Uses a convolution 1×1 to combine the channels resulting from the previous step.

XCeption's Structure:
1. Entry Flow:
    - Extracts initial features from the image.
    - Uses convolutions and depth-wise separable modules.
2. Middle Flow:
    - Repeat a block 8 times.
    - Exclusively uses depth-first separable convolutions with residual connections (as in ResNet).
3. Exit Flow:
    - Summarizes the extracted features.
    - Ends with dense (fully connected) layers for classification.
  
## Results:
Confusion Matrix:<br>
<img align=center src=https://github.com/VYR4L/Facial-Expression-Recognizer/blob/main/old/confusion_matrix_xception.png>

- Happy (class 3): Excellent performance — 1446 correct answers, with few errors. This is in line with the f1-score of 0.79 in the classification report.
- Surprise (class 5): 470 hits, but also many errors with happy and sad — suggests that these expressions may be visually similar.
- Fear (class 1): Poor performance — many errors and low recall (0.32), indicating that the model has difficulty correctly identifying fear.
- Angry, sad, neutral: Confusions with each other, which is common because they are expressions with less visual contrast.

Classification report:

              precision    recall  f1-score   support

       angry       0.51      0.40      0.45       960
        fear       0.42      0.32      0.36      1018
       happy       0.74      0.84      0.79      1825
     neutral       0.50      0.59      0.54      1216
         sad       0.45      0.47      0.46      1139
    surprise       0.71      0.60      0.65       797
    --------------------------------------------------
    accuracy                           0.57      7066
    macro avg       0.53     0.55      0.54      7066
    weighted avg    0.57     0.57      0.57      7066

- Happy and surprise are the highest classified emotions.
- Fear and angry are the worst — low precision and recall, which indicates confusion with other classes.
- avg macro vs. Weighted avg: The macro (simple average) is lower, showing that smaller classes (like fear) are performing poorly.

## Tutorial:
### For the training:
1. Download the following Kaggle DataSet: **<https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset/data>**;
2. Extract it and make sure the path is right. You can test it using<br><code>print(DATA_SET||TRAIN_IMAGES||VALIDATION_IMAGES)</code>;
3. Install PyTorch from its official website: **<https://pytorch.org/>**
4. Install the dependecies through: <br><code>pip install -r requirements.txt</code>
5. Run <code>python training_xception.py</code> and wait 'til complete. Make sure you're in the right folder.

### For the aplication:
1. Make sure all dependencies are installed. If you haven't already done so, you can do so by running <code>pip install -r requirements.txt</code>
2. Run <code>python main.py</code>

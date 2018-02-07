
# Self-Driving Car Engineer Nanodegree

## Deep Learning

## Project: Build a Traffic Sign Recognition Classifier

In this notebook, a template is provided for you to implement your functionality in stages, which is required to successfully complete this project. If additional code is required that cannot be included in the notebook, be sure that the Python code is successfully imported and included in your submission if necessary. 

> **Note**: Once you have completed all of the code implementations, you need to finalize your work by exporting the iPython Notebook as an HTML document. Before exporting the notebook to html, all of the code cells need to have been run so that reviewers can see the final implementation and output. You can then export the notebook by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission. 

In addition to implementing code, there is a writeup to complete. The writeup should be completed in a separate file, which can be either a markdown file or a pdf document. There is a [write up template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) that can be used to guide the writing process. Completing the code template and writeup template will cover all of the [rubric points](https://review.udacity.com/#!/rubrics/481/view) for this project.

The [rubric](https://review.udacity.com/#!/rubrics/481/view) contains "Stand Out Suggestions" for enhancing the project beyond the minimum requirements. The stand out suggestions are optional. If you decide to pursue the "stand out suggestions", you can include the code in this Ipython notebook and also discuss the results in the writeup file.


>**Note:** Code and Markdown cells can be executed using the **Shift + Enter** keyboard shortcut. In addition, Markdown cells can be edited by typically double-clicking the cell to enter edit mode.

---
## Step 0: Load The Data


```python
# Load pickled data
import pickle

# TODO: Fill this in based on where you saved the training and testing data

training_file = './traffic-signs-data/train.p'
validation_file='./traffic-signs-data/valid.p'
testing_file = './traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_valid: ", X_valid.shape)
print("y_valid: ", y_valid.shape)
print("X_test: ", X_test.shape)
print("y_test: ", y_test.shape)

```

    X_train:  (34799, 32, 32, 3)
    y_train:  (34799,)
    X_valid:  (4410, 32, 32, 3)
    y_valid:  (4410,)
    X_test:  (12630, 32, 32, 3)
    y_test:  (12630,)


---

## Step 1: Dataset Summary & Exploration

The pickled data is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. **THESE COORDINATES ASSUME THE ORIGINAL IMAGE. THE PICKLED DATA CONTAINS RESIZED VERSIONS (32 by 32) OF THESE IMAGES**

Complete the basic data summary below. Use python, numpy and/or pandas methods to calculate the data summary rather than hard coding the results. For example, the [pandas shape method](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.shape.html) might be useful for calculating some of the summary results. 

### Provide a Basic Summary of the Data Set Using Python, Numpy and/or Pandas


```python
### Replace each question mark with the appropriate value. 
### Use python, pandas or numpy methods rather than hard coding the results
import numpy as np

# TODO: Number of training examples
n_train = X_train.shape[0]

# TODO: Number of validation examples
n_validation = X_valid.shape[0]

# TODO: Number of testing examples.
n_test = X_test.shape[0]

# TODO: What's the shape of an traffic sign image?
image_shape = X_train.shape[1:]

# TODO: How many unique classes/labels there are in the dataset.
n_classes = len(np.unique(y_train))

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)
```

    Number of training examples = 34799
    Number of testing examples = 12630
    Image data shape = (32, 32, 3)
    Number of classes = 43


### Include an exploratory visualization of the dataset

Visualize the German Traffic Signs Dataset using the pickled file(s). This is open ended, suggestions include: plotting traffic sign images, plotting the count of each sign, etc. 

The [Matplotlib](http://matplotlib.org/) [examples](http://matplotlib.org/examples/index.html) and [gallery](http://matplotlib.org/gallery.html) pages are a great resource for doing visualizations in Python.

**NOTE:** It's recommended you start with something simple first. If you wish to do more, come back to it after you've completed the rest of the sections. It can be interesting to look at the distribution of classes in the training, validation and test set. Is the distribution the same? Are there more examples of some classes than others?


```python
### Data exploration visualization code goes here.
### Feel free to use as many code cells as needed.
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as plt
import pandas as pd
# Visualizations will be shown in the notebook.
%matplotlib inline


def imagePlot(image_data,label, squeeze = False, cmap = None):
    fig, axs = plt.subplots(5,10, figsize=(15, 7))
    fig.subplots_adjust(hspace = .7, wspace=.001)
    axs = axs.ravel()
    for i in range(50):
        index = random.randint(0, image_data.shape[0])
        if squeeze:
            image = image_data[index].squeeze()
        else:
            image = image_data[index]
        axs[i].axis('off')
        if cmap:
            axs[i].imshow(image,cmap=cmap)
        else:
            axs[i].imshow(image)
        axs[i].set_title(label[index])

imagePlot(X_train,y_train)
```


![png](output_8_0.png)



```python
# histogram of label frequency
plt.figure(figsize=(12,7))

hist1, bins1 = np.histogram(y_train, bins=n_classes)
width1 = 0.6 * (bins1[1] - bins1[0])
center = (bins1[:-1] + bins1[1:]) / 2
plt.bar(center, hist1, width=width1)

hist2, bins2 = np.histogram(y_test, bins=n_classes)
width2 = 0.6 * (bins2[1] - bins2[0])
plt.bar(center, hist2, width=width2)

hist3, bins3 = np.histogram(y_valid, bins=n_classes)
width3 = 0.6 * (bins3[1] - bins3[0])
plt.bar(center, hist3, width=width3)

plt.xlabel("class")
plt.ylabel("Number of images")
plt.legend(['y_train','y_test','y_valid'], loc = 'upper right')
plt.show()
```


![png](output_9_0.png)


----

## Step 2: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The LeNet-5 implementation shown in the [classroom](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) at the end of the CNN lesson is a solid starting point. You'll have to change the number of classes and possibly the preprocessing, but aside from that it's plug and play! 

With the LeNet-5 solution from the lecture, you should expect a validation set accuracy of about 0.89. To meet specifications, the validation set accuracy will need to be at least 0.93. It is possible to get an even higher accuracy, but 0.93 is the minimum for a successful project submission. 

There are various aspects to consider when thinking about this problem:

- Neural network architecture (is the network over or underfitting?)
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It's not required to be familiar with the approach used in the paper but, it's good practice to try to read papers like these.

### Pre-process the Data Set (normalization, grayscale, etc.)

Minimally, the image data should be normalized so that the data has mean zero and equal variance. For image data, `(pixel - 128)/ 128` is a quick way to approximately normalize the data and can be used in this project. 

Other pre-processing steps are optional. You can try different techniques to see if it improves performance. 

Use the code cell (or multiple code cells, if necessary) to implement the first step of your project.


```python
### Preprocess the data here. It is required to normalize the data. Other preprocessing steps could include 
### converting to grayscale, etc.
### Feel free to use as many code cells as needed.

def rgb2grayscale(image_data):
    return np.sum(image_data/3,axis = 3, keepdims = True)
    
def normalize_grayscale(image_data):
    """
    Normalize the image data with Min-Max scaling to a range of [-0.5, 0.5]
    :param image_data: The image data to be normalized
    :return: Normalized image data
    """
    a = -0.5
    b = 0.5
    grayscale_min = 0
    grayscale_max = 255
    return a + ( ( (image_data - grayscale_min)*(b - a) )/( grayscale_max - grayscale_min ) )

def preprocessing(image_data):
    return normalize_grayscale(rgb2grayscale(image_data))


X_train = preprocessing(X_train)
X_valid = preprocessing(X_valid)
X_test = preprocessing(X_test)

print("X_train_rect: ", X_train.shape)
print("y_train_rect: ", y_train.shape)
print("X_valid_rect: ", X_valid.shape)
print("y_valid_rect: ", y_valid.shape)
print("X_test_rect: ", X_test.shape)
print("y_test_rect: ", y_test.shape)
    
```

    X_train_rect:  (34799, 32, 32, 1)
    y_train_rect:  (34799,)
    X_valid_rect:  (4410, 32, 32, 1)
    y_valid_rect:  (4410,)
    X_test_rect:  (12630, 32, 32, 1)
    y_test_rect:  (12630,)


## Question 1
Describe how you preprocessed the data. Why did you choose that technique?
### Answer
I use two preprocessing methods:
    1. RGBtoGrayscale - It helps reducing the training time, and it is also mentioned in LeCun's traffic sign classification article that it works well.
    2. Normalizing the data to range (-0.5,0.5). This will roughly reduce the mean from around 82 to around 0. It can normalize the data dimensions so that they are of approximately the same scale.


```python
imagePlot(X_train,y_train, squeeze = True, cmap = 'gray')
#imagePlot(X_valid, y_valid, squeeze = True, cmap = 'gray')
#imagePlot(X_test, y_test, squeeze = True, cmap = 'gray')
```


![png](output_15_0.png)



```python
from sklearn.utils import shuffle

X_train, y_train = shuffle(X_train, y_train)
```


```python
import tensorflow as tf

EPOCHS = 15
BATCH_SIZE = 128
```

### Model Architecture


```python
### Define your architecture here.
### Feel free to use as many code cells as needed.


from tensorflow.contrib.layers import flatten

def LeNet(x):    
    # Arguments used for tf.truncated_normal, randomly defines variables for the weights and biases for each layer
    mu = 0
    sigma = 0.1
    
    # Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x30.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 30), mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(30))
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], padding='VALID') + conv1_b

    # Activation.
    conv1 = tf.nn.relu(conv1)

    # Pooling. Input = 28x28x30. Output = 14x14x30.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Layer 2: Convolutional. Output = 10x10x56.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 30, 56), mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(56))
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], padding='VALID') + conv2_b
    
    # Activation.
    conv2 = tf.nn.relu(conv2)

    # Pooling. Input = 10x10x56. Output = 5x5x56.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    # Flatten. Input = 5x5x56. Output = 1400.
    fc0   = flatten(conv2)
    fc0 = tf.nn.dropout(fc0,keep_prob)
    
    # Layer 3: Fully Connected. Input = 1400. Output = 120.
    fc1_W = tf.Variable(tf.truncated_normal(shape=(1400, 120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1   = tf.matmul(fc0, fc1_W) + fc1_b
    
    # Activation.
    fc1    = tf.nn.relu(fc1)
    fc1 = tf.nn.dropout(fc1,keep_prob)

    # Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(120, 84), mean = mu, stddev = sigma))
    fc2_b  = tf.Variable(tf.zeros(84))
    fc2    = tf.matmul(fc1, fc2_W) + fc2_b
    
    # Activation.
    fc2    = tf.nn.relu(fc2)

    # Layer 5: Fully Connected. Input = 84. Output = 43.
    fc3_W  = tf.Variable(tf.truncated_normal(shape=(84, 43), mean = mu, stddev = sigma))
    fc3_b  = tf.Variable(tf.zeros(43))
    logits = tf.matmul(fc2, fc3_W) + fc3_b
    
    return logits
```

## Question 2
Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.
### Answer

| Layer         		|     Description	        					    | 
|:---------------------:|:---------------------------------------------:    | 
| Layer1           		| Convolutional. Input = 32x32x1. Output = 28x28x30 | 
| RELU                  |                                                   |
| Max pooling          	| Input = 28x28x30. Output = 14x14x30.	            |      
| Layer2				| Convolutional. Output = 10x10x56.                 |
| RELU		            |                                                   |
| Max pooling	      	| Input = 10x10x56. Output = 5x5x56. 				|
| Flatten               | Input = 5x5x56. Output = 1400.                    |    
| Dropout               | keep_prob = 0.7                                   |
| Layer3        		| Fully Connected. Input = 1400. Output = 120.      |
| RELU		            |                                                   |
| Dropout               | keep_prob = 0.7                                   |
| Layer4				| Fully Connected. Input = 120. Output = 84.        |
| RELU		            |                                                   |
| Layer5                | Fully Connected. Input = 84. Output = 43.         |

### Features and Labels


```python
x = tf.placeholder(tf.float32, (None, 32, 32, 1))
y = tf.placeholder(tf.int32, (None))
keep_prob = tf.placeholder(tf.float32)
one_hot_y = tf.one_hot(y, 43)
```

### Train, Validate and Test the Model

A validation set can be used to assess how well the model is performing. A low accuracy on the training and validation
sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.


```python
### Train your model here.
### Calculate and report the accuracy on the training and validation set.
### Once a final model architecture is selected, 
### the accuracy on the test set should be calculated and reported as well.
### Feel free to use as many code cells as needed.

rate = 0.001
actual_keep_prob = 0.7

logits = LeNet(x)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_operation = optimizer.minimize(loss_operation)



correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: 1.0})
        total_accuracy += (accuracy * len(batch_x))
    return total_accuracy / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training...")
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, keep_prob: actual_keep_prob})
            
        validation_accuracy = evaluate(X_valid, y_valid)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()
        
    saver.save(sess, './lenet')
    print("Model saved")
```

    Training...
    
    EPOCH 1 ...
    Validation Accuracy = 0.828
    
    EPOCH 2 ...
    Validation Accuracy = 0.893
    
    EPOCH 3 ...
    Validation Accuracy = 0.913
    
    EPOCH 4 ...
    Validation Accuracy = 0.938
    
    EPOCH 5 ...
    Validation Accuracy = 0.955
    
    EPOCH 6 ...
    Validation Accuracy = 0.964
    
    EPOCH 7 ...
    Validation Accuracy = 0.959
    
    EPOCH 8 ...
    Validation Accuracy = 0.965
    
    EPOCH 9 ...
    Validation Accuracy = 0.963
    
    EPOCH 10 ...
    Validation Accuracy = 0.970
    
    EPOCH 11 ...
    Validation Accuracy = 0.962
    
    EPOCH 12 ...
    Validation Accuracy = 0.966
    
    EPOCH 13 ...
    Validation Accuracy = 0.967
    
    EPOCH 14 ...
    Validation Accuracy = 0.963
    
    EPOCH 15 ...
    Validation Accuracy = 0.964
    
    Model saved


## Question 3
Describe how you trained your model. 
### Answer
I improved the LeNet to train the model. I increase the depth and add the dropout and max pooling method.

## Question 4
Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.
### Answer
To improve the accuracy, first I did the preprocessing of the image (RGB2GrayScale and Normalization). By using the improved LeNet and by adjusting the parameters, finally I set the EPOCHS = 15 and BATCH_SIZE = 128, leaning rate = 0.001 and keep_prob = 0.7 as the final value to get the accuracy around 96%.


```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    train_accuracy = evaluate(X_train, y_train)
    validation_accuracy = evaluate(X_valid, y_valid)
    test_accuracy = evaluate(X_test, y_test)
    print("Train Accuracy = {:.3f}".format(train_accuracy))
    print("Validation Accuracy = {:.3f}".format(validation_accuracy))
    print("Test Accuracy = {:.3f}".format(test_accuracy))
```

    Train Accuracy = 0.999
    Validation Accuracy = 0.964
    Test Accuracy = 0.954


---

## Step 3: Test a Model on New Images

To give yourself more insight into how your model is working, download at least five pictures of German traffic signs from the web and use your model to predict the traffic sign type.

You may find `signnames.csv` useful as it contains mappings from the class id (integer) to the actual sign name.


```python
#reading in an image
import cv2
import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

new_labels = [11, 1, 12, 14, 38, 34, 18, 25]

fig, axs = plt.subplots(2,4, figsize=(15, 5))
fig.subplots_adjust(hspace = .2, wspace=.001)
axs = axs.ravel()

test_images = []

for i, img in enumerate(glob.glob('./test_images/*.png')):
    image = cv2.imread(img)
    axs[i].axis('off')
    axs[i].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    test_images.append(image) 
    
test_images = np.reshape(test_images, (8,32,32,3))
images_normalized=preprocessing(test_images)

```


![png](output_29_0.png)


## Question 5
Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook.
### Answer
The candidate images are shown above. It would be hard to classify if the image is significantly different from the training data. Or the image quality is quite low. I can get a good prediction result using my CNN.

* Image 1: Right-of-way at the next intersection
* image 2: Speed limit (30km/h)
* image 3: Priority road
* image 4: Stop
* image 5: Keep right
* image 6: Turn left ahead
* image 7: General caution
* image 8: Road work

### Predict the Sign Type for Each Image


```python
### Run the predictions here and use the model to output the prediction for each image.
### Make sure to pre-process the images with the same pre-processing pipeline used earlier.
### Feel free to use as many code cells as needed.
import csv

print("")



predic = tf.argmax(logits, 1)
i = 1
with open( './signnames.csv', 'rt') as f:
    reader = csv.reader(f)
    label_name = list(reader)

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    prediction_label = sess.run(predic, feed_dict = {x: images_normalized, keep_prob: 1.0})


for e in prediction_label:
    print("Sign Type Prediction of image" ,i,"-----", label_name[e+1])
    i = i + 1

    
```

    Sign Type Prediction of image 1 ----- ['11', 'Right-of-way at the next intersection']
    Sign Type Prediction of image 2 ----- ['1', 'Speed limit (30km/h)']
    Sign Type Prediction of image 3 ----- ['12', 'Priority road']
    Sign Type Prediction of image 4 ----- ['14', 'Stop']
    Sign Type Prediction of image 5 ----- ['38', 'Keep right']
    Sign Type Prediction of image 6 ----- ['34', 'Turn left ahead']
    Sign Type Prediction of image 7 ----- ['18', 'General caution']
    Sign Type Prediction of image 8 ----- ['25', 'Road work']


## Question 6
Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.
### Answer
From the results above, we can see that the prediction accuracy is 100% because it predicts all the 8 images correctly. The accuracy is higher comparing with the predicting result on previous test set.

### Analyze Performance


```python
### Calculate the accuracy for these 5 new images. 
### For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate on these new images.

with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    my_accuracy = evaluate(images_normalized, new_labels)
    print("Test Set Accuracy = {:0.1%}".format(my_accuracy))

```

    Test Set Accuracy = 100.0%


### Output Top 5 Softmax Probabilities For Each Image Found on the Web

For each of the new images, print out the model's softmax probabilities to show the **certainty** of the model's predictions (limit the output to the top 5 probabilities for each image). [`tf.nn.top_k`](https://www.tensorflow.org/versions/r0.12/api_docs/python/nn.html#top_k) could prove helpful here. 

The example below demonstrates how tf.nn.top_k can be used to find the top k predictions for each image.

`tf.nn.top_k` will return the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of a possible 43) and the correspoding class ids.

Take this numpy array as an example. The values in the array represent predictions. The array contains softmax probabilities for five candidate images with six possible classes. `tf.nn.top_k` is used to choose the three classes with the highest probability:

```
# (5, 6) array
a = np.array([[ 0.24879643,  0.07032244,  0.12641572,  0.34763842,  0.07893497,
         0.12789202],
       [ 0.28086119,  0.27569815,  0.08594638,  0.0178669 ,  0.18063401,
         0.15899337],
       [ 0.26076848,  0.23664738,  0.08020603,  0.07001922,  0.1134371 ,
         0.23892179],
       [ 0.11943333,  0.29198961,  0.02605103,  0.26234032,  0.1351348 ,
         0.16505091],
       [ 0.09561176,  0.34396535,  0.0643941 ,  0.16240774,  0.24206137,
         0.09155967]])
```

Running it through `sess.run(tf.nn.top_k(tf.constant(a), k=3))` produces:

```
TopKV2(values=array([[ 0.34763842,  0.24879643,  0.12789202],
       [ 0.28086119,  0.27569815,  0.18063401],
       [ 0.26076848,  0.23892179,  0.23664738],
       [ 0.29198961,  0.26234032,  0.16505091],
       [ 0.34396535,  0.24206137,  0.16240774]]), indices=array([[3, 0, 5],
       [0, 1, 4],
       [0, 5, 1],
       [1, 3, 5],
       [1, 4, 3]], dtype=int32))
```

Looking just at the first row we get `[ 0.34763842,  0.24879643,  0.12789202]`, you can confirm these are the 3 largest probabilities in `a`. You'll also notice `[3, 0, 5]` are the corresponding indices.


```python
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))
    new_classes = sess.run(logits, feed_dict={x: images_normalized, keep_prob : 1.0})
    predicts = sess.run(tf.nn.top_k(new_classes, k=5, sorted=True))

for i in range(len(predicts[0])):
    print('Image', i+1, 'probabilities:', predicts[0][i])
    print('        predicted classes:', predicts[1][i])
```

    Image 1 probabilities: [54.22739   27.564034  17.530163  15.110448   6.2145658]
            predicted classes: [11 30 21 27 28]
    Image 2 probabilities: [29.845423   17.717634    7.2236967   3.1854234   0.61586994]
            predicted classes: [ 1  2  5  0 31]
    Image 3 probabilities: [52.803734  21.015413   9.218461   5.4529085 -0.7736264]
            predicted classes: [12 40 15  9  7]
    Image 4 probabilities: [9.452698  5.200999  2.8529932 1.1711658 0.6735901]
            predicted classes: [14 33 39 25  1]
    Image 5 probabilities: [93.731125   20.357084    5.5643163  -0.22029892 -0.2923887 ]
            predicted classes: [38 34  2 40 23]
    Image 6 probabilities: [28.721418  12.432667   3.388598   2.8568501  1.3255996]
            predicted classes: [34 38  9 13 36]
    Image 7 probabilities: [53.97844   20.121658  16.243341   5.232601  -0.5972513]
            predicted classes: [18 26 27 28 24]
    Image 8 probabilities: [10.509097   4.788968   3.3909569  2.584158   2.098513 ]
            predicted classes: [25 26 24 19 37]


## Question 7
Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. 
### Answer
The top five softmax probabilities for each image along with the sign type of each probability are shown above.

### Project Writeup

Once you have completed the code implementation, document your results in a project writeup using this [template](https://github.com/udacity/CarND-Traffic-Sign-Classifier-Project/blob/master/writeup_template.md) as a guide. The writeup can be in a markdown or pdf file. 

> **Note**: Once you have completed all of the code implementations and successfully answered each question above, you may finalize your work by exporting the iPython Notebook as an HTML document. You can do this by using the menu above and navigating to  \n",
    "**File -> Download as -> HTML (.html)**. Include the finished document along with this notebook as your submission.

---

## Step 4 (Optional): Visualize the Neural Network's State with Test Images

 This Section is not required to complete but acts as an additional excersise for understaning the output of a neural network's weights. While neural networks can be a great learning device they are often referred to as a black box. We can understand what the weights of a neural network look like better by plotting their feature maps. After successfully training your neural network you can see what it's feature maps look like by plotting the output of the network's weight layers in response to a test stimuli image. From these plotted feature maps, it's possible to see what characteristics of an image the network finds interesting. For a sign, maybe the inner network feature maps react with high activation to the sign's boundary outline or to the contrast in the sign's painted symbol.

 Provided for you below is the function code that allows you to get the visualization output of any tensorflow weight layer you want. The inputs to the function should be a stimuli image, one used during training or a new one you provided, and then the tensorflow variable name that represents the layer's state during the training process, for instance if you wanted to see what the [LeNet lab's](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/6df7ae49-c61c-4bb2-a23e-6527e69209ec/lessons/601ae704-1035-4287-8b11-e2c2716217ad/concepts/d4aca031-508f-4e0b-b493-e7b706120f81) feature maps looked like for it's second convolutional layer you could enter conv2 as the tf_activation variable.

For an example of what feature map outputs look like, check out NVIDIA's results in their paper [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) in the section Visualization of internal CNN State. NVIDIA was able to show that their network's inner weights had high activations to road boundary lines by comparing feature maps from an image with a clear path to one without. Try experimenting with a similar test to show that your trained network's weights are looking for interesting features, whether it's looking at differences in feature maps from images with or without a sign, or even what feature maps look like in a trained network vs a completely untrained one on the same sign image.

<figure>
 <img src="visualize_cnn.png" width="380" alt="Combined Image" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Your output should look something like this (above)</p> 
 </figcaption>
</figure>
 <p></p> 



```python
### Visualize your network's feature maps here.
### Feel free to use as many code cells as needed.

# image_input: the test image being fed into the network to produce the feature maps
# tf_activation: should be a tf variable name used during your training procedure that represents the calculated state of a specific weight layer
# activation_min/max: can be used to view the activation contrast in more detail, by default matplot sets min and max to the actual min and max values of the output
# plt_num: used to plot out multiple different weight feature map sets on the same block, just extend the plt number for each new feature map entry

def outputFeatureMap(image_input, tf_activation, activation_min=-1, activation_max=-1 ,plt_num=1):
    # Here make sure to preprocess your image_input in a way your network expects
    # with size, normalization, ect if needed
    # image_input =
    # Note: x should be the same name as your network's tensorflow data placeholder variable
    # If you get an error tf_activation is not defined it may be having trouble accessing the variable from inside a function
    activation = tf_activation.eval(session=sess,feed_dict={x : image_input})
    featuremaps = activation.shape[3]
    plt.figure(plt_num, figsize=(15,15))
    for featuremap in range(featuremaps):
        plt.subplot(6,8, featuremap+1) # sets the number of feature maps to show on each row and column
        plt.title('FeatureMap ' + str(featuremap)) # displays the feature map number
        if activation_min != -1 & activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin =activation_min, vmax=activation_max, cmap="gray")
        elif activation_max != -1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmax=activation_max, cmap="gray")
        elif activation_min !=-1:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", vmin=activation_min, cmap="gray")
        else:
            plt.imshow(activation[0,:,:, featuremap], interpolation="nearest", cmap="gray")
```

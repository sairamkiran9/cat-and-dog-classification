# cat-and-dog-classification

**Computer Vision** is the field of computer science that focuses on replicating parts of the complexity of the human visual system and enabling computers to identify and process objects in images and videos in the same way that humans do. With computer vision, our system can extract, analyze and understand useful information from an individual image or a sequence of images. Computer vision is a domain of artificial intelligence that works on enabling computers to see, identify, and process images in the same way that human vision does, and then provide the appropriate results. Image classification is one of the application of computer vision.

### Image Classification
It is a process of predicting a specific class, or label, for something that is defined by a set of data. Image classification is a subset of the classification problem, where an entire image is assigned a label and categorized based on the image features.

Further the image calssification is classified into two type:
- Supervised classification which uses the spectral signatures obtained from training samples to classify an image.
- Unsupervised classification which finds spectral classes (or clusters) in a multiband image without the analyst’s intervention.

In this project, I will perform supervised image classification by training the images and their respective labels over the neural network model to predict an image class while evaluating.

### Applications
Web services are often protected with a challenge that's supposed to be easy for people to solve, but difficult for computers. Such a challenge is often called a CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof). HIPs are used for many purposes, such as to reduce email and blog spam and prevent brute-force attacks on web site passwords. Asirra(Animal Species Image Recognition for Restricting Access) is a HIP that works by asking users to identify photographs of cats and dogs.

### Dataset Used
The [dataset](https://https://www.kaggle.com/c/dogs-vs-cats/data) contains 25,000 colour images of cats and dogs 12500 each, with its name and number of the image as a label in jpg format(Ex:"cat.1.jpg") for training and 12500 images of both with a number of the image as label(Ex:"100.jpg"). The dataset is prepared by MICROSOFT CORPORATION from the data provided by Asirra, the world's largest site devoted to finding homes for homeless pet for building CAPTCHA of dogs and cats.

### Method Used
Modern deep learning involves tens or even hundreds of successive layers of representation, and they are all learned automatically from exposure to training data(ANN). Meanwhile, other approaches to machine learning focus on learning only one or two layers of representation of the data( which is the major limitation of machine learning).<br>

A convolutional neural network (CNN) is a specific type of artificial neural network that uses the concept of perceptions, a machine learning unit algorithm, for supervised learning, to extract features from the data. Many algorithms are developed to achieve the best accuracy possible. Some models are Lenet5, AlexNet, VGG16, VGG19, GoogLeNet, and many more.
These networks are blocks of sequential convolutional layers and other layers that help in extracting the important necessary features required as the input is passed through them.<br>

The deep learning architecture for image classification generally includes convolutional layers, making it a convolutional neural network (CNN). A typical use case for CNNs is where you feed the network images and the network classifies the data. CNN's start with an input “kernel” which isn’t intended to parse all the training data at once. For example, to input an image of 100 x 100 pixels, you wouldn’t want a layer with 10,000 nodes. Rather, you create a scanning input layer of say 10 x 10 which you feed the first 10 x 10 pixels of the image. Once you passed that input, you feed it the next 10 x 10 pixels by moving the kernel one pixel to the right. And the technique is known as sliding windows.

### Loading the dataset
First of all, there is no need to load the data into our working instance for a small dataset. But if the dataset size is large, I recommend it to load the data into the working environment to reduce the initial training time of the model. While working in colab we can directly upload data from a local system or import data from a drive. If the data is available on the Kaggle site, it is best to load the data from the website itself using Kaggle's API provided by Kaggle. You can refer to the procedure in this [article](https://www.kaggle.com/general/74235).

### Data Augmentation
Data augmentation is a strategy that enables practitioners to significantly increase the diversity of data available for training models, without actually collecting new data. Data augmentation techniques such as cropping, padding, and horizontal flipping are commonly used to train large neural networks. By performing this we can increase the model accuracy and mainly avoid overfitting of the model(a major disadvantage of neural networks).

### Traning
The model is iterated for 50 epochs and validated with a dataset of size 0.2 from training set. The sequential model is iterated several times to attain best accuracy possible by iterating several times our model backpropagates for every iterations by changing its inner weights and learning rate factor, and the model is saved to reuse whenever need i.e it reduces the training time.

### Results
The model as attained 95.75% of total accuracy towards the validation data. The respective validation, training accuracy and loss curves along with the confusion matrix is being visualised using matplotlib module and a special visualisation API of tensorflow - Tensorboard.

Some results of the predicted outputs of the model.<br>

![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/dog.png)
![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/cat.png) <br>

#### Visualisation plots using matplotlib module.<br>

![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/acc.png)
![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/loss.png) <br>
##### Confusion matrix<br>
![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/matrix.png)<br>

#### Visualisation plots using Tensorboard.<br>

Using tensorboard we can get the model architecture with the respective dimensions of input and outputs at each and every node.<br>

![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/tensor.jpg)<br>

The below are the some parameters distributions that are stored in the event log files during the training time of 50 epochs. 

![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/pic1.jpg) <br>
![alt text](https://github.com/sairamkiran9/cat-and-dog-classification/blob/master/imgs/pic2.jpg)<br>

Tensorboard is a wonderful API for visualisation and prepare plots as needed, check out the tensorboard API in tensorflow.org.

### Conclusion
This is the model for predicting the classification of images. The same architecture can be used for predicting the Covid-19 patients by training the model with lungs X-ray datasets. This project is a two class classifier model, it can be extend to predict any number of classes by making the output dense layer to N classes and training the model with sufficient data of information.<br>
 1. Covid-19 patients classification (2-Class classifier)<br>
 2. Clothes classification           (N-Class classifier)<br>
 3. Dog breed classification         (N-Class classifier)<br>
 4. Handwriiten digitd classification(10-Class classifier)<br>
 5. Leaves species classification    (N-Class classifier)<br>
  where N depends on the number of types of data available through datasets.
These are some extension examples of this project and moreover the model can only predict a single cat or dog image. We can improvise the model using architectures like [YOLO](https://towardsdatascience.com/yolo-you-only-look-once-real-time-object-detection-explained-492dc9230006).

For improvising the model performance, we can make changes in batch size, learning rate and model parameters. For better understanding go through this [article](https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e).

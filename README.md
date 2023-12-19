# CIFAR-10
The agenda of the case study is to develop a multiclass classification model using the CFIR-10 dataset. CFIR-10 is a dataset comprising images across ten distinct classes .The goal is to create a model capable of accurately classifying these images into their respective  classes.

# Environment details:

The libraries used were:<br>
-python==3.10.12<br>
-matplotlib==3.7.1<br>
-tensorflow==2.15.0<br>
-numpy==1.23.5<br>
-gradio==4.10.0<br>
-seaborn==0.12.2<br>
-sklearn==1.2.2<br>
-PIL==9.4.0<br>






# CIFAR-10 Dataset Information(https://www.cs.toronto.edu/~kriz/cifar.html) 

The CFIR-10 dataset is a collection of images categorized into ten classes. 

<ins>Classes</ins>:  airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck<br>


<ins>Size</ins>:  The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.<br>


<ins>Image size</ins>:  32x32


<ins>Class Distribution</ins>:  The dataset has an imbalanced class distribution

# Preprocessing:


The images were normalized before set for training

# Model Architecture:
The model architecture is designed as a sequential stack of layers utilizing Convolutional Neural Network (CNN) components followed by densely connected layers for the final classification. Here is a breakdown of the architecture:<br>

1.<ins>Convolutional Layers</ins>:<br>

Three sets of convolutional layers (Conv2D) with 64 filters of size (3, 3) and ReLU activation.<br>
Each convolutional layer is followed by Batch Normalization and MaxPooling (MaxPooling2D) layers with a (2, 2) pool size to downsample the feature maps.<br>

2.<ins>Flattening</ins>:<br>

The output from the convolutional layers is flattened using Flatten() to prepare it for the fully connected layers.<br>

3.<ins>Densely Connected Layers</ins>:<br>

Two dense layers (Dense) with 64 and 32 units respectively, using ReLU activation
functions. A dropout of 25% is applied after each dense layer to mitigate overfitting.<br>

4.<ins>Output Layer</ins>:<br>

The final layer is a dense layer with 10 units, employing a softmax activation function to output probabilities for each of the ten classes in the CFIR-10 dataset.

# Pretrained Model:
Using the Keras library, I loaded the <ins>VGG16 model</ins> with pre-trained weights, excluding the fully connected layers at the top, to prepare it for retraining on my specific dataset. I have Fine-tuned a select number of convolutional layers along with the top layers to adapt the model to my dataset.


# Model Compilation:

<ins>Optimizer</ins>: Adam optimizer with a learning rate of 0.001 is used to optimize the model's weights during training.<br>

<ins>Loss Function</ins>: The loss function chosen for optimization is the sparse_categorical_crossentropy. This loss function is suitable for multiclass classification tasks<br>




# How to run code:
1.<u>Clone</u> repositary :<br>
`git clone https://github.com/Nirmal-Sudheer/CIFAR-10.git`  <br>
`cd CIFAR-10.git`<br>`

2.Install Dependencies for training:<br>
`pip install -r requirements.txt`<br>

  or to use transfer learning<br>

`pip install -r requirements_VGG16.txt`<br>


3.Upload the dataset into environment and then run the `CIFAR_10_Training.ipynb` file.<br>

  Or to use transfer learning<br>
  
Upload the dataset into environment and then run the `CIFAR-10_VGG16.ipynb` file


4.Install Dependencies for deployment:<br>
`pip install -r requirements_Deployment.txt`

5.Upload saved model into environment and then run CIFAR_10_Deployment.ipynb file <br>

6.Upload images to gradio interface to run inference.

# Model Results:
<ins>Without Transfer learning:</ins>
<ins>Training Accuracy: 82%</ins><br>
The model demonstrated an accuracy of 82% on the training dataset. This indicates that during the training phase, the model correctly classified 82% of the training samples.<br>
<ins>Test Accuracy: 72%</ins><br>
The model's performance on unseen data, represented by the test dataset, resulted in an accuracy of 72%.<br>

<ins>With Transfer learning:</ins><br>
<ins>Training Accuracy: 79%</ins><br>
The model demonstrated an accuracy of 79% on the training dataset .<br>
<ins>Test Accuracy: 70%</ins><br>
The model's performance on unseen data, represented by the test dataset, resulted in an accuracy of 70%.<br>




# Conclusion:
The model shows promise in its ability to classify CFIR-10 images with 72% accuracy on unseen data. Further enhancements and fine-tuning could potentially improve its performance and generalization ability.<br>

Fine-tuning the VGG16 model on the dataset yielded a test accuracy of 70%. With additional resources and dedicated hyperparameter tuning, achieving a higher accuracy appears feasible, indicating the potential for enhanced performance with optimized model configurations and increased computational capabilities.





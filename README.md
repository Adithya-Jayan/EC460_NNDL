# EC460_NNDL
Repository for Assignments and codes related to EC460 - Neural Networks and Deep learning

**(Note that colab links will not be accesible to non-team members)**
##### Assignment1 colab link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11PMXnygY4Hjz4EmXYStrvDwVzU4MYPHk?usp=sharing)
##### Assignment2 colab link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Dg9AGCpsdrytOQXL8GSubB59l2K7HQZd?usp=sharing)
##### Assignment3 colab link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1NvQH-ElZp7WMZesYloWm6jAYseW3r06Z)
##### Assignment4 colab link: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1QvWDKWYFFZ2JguEkcdUoPerZ2UBhSRtD?usp=sharing)

## Questions
### Assignment 1

##### Q.1. Write python code for
- (a). Plotting the following activation functions for the input x in the range of -20 to 20. 
	>- Sigmoid 
	>- TanH 
	>- Softsign 
	>- ReLU 
	>- Softmax

- (b). The derivative of the following activation functions and their plots for the input x in the range of -20 to 20.
	>- Sigmoid 
	>- TanH 
	>- Softsign 
	>- ReLU 

##### Q.2. Assume we have a 2-input neuron and has the following parameters: w = [1, 1] ; b=2 .Write a python code for calculating the feed-forward output of neural networks by using following activation functions for input x=[4, 5]. 
>- Sigmoid 
>- TanH 
>- Softsign 
>- ReLU 
>- Network output comparison with different activation 
functions using Bar chart plot 
>- verify each result by analytical method

##### Q.3. Write python code to compute the following Regression loss value for the given true output and predicted output by network:

`# y_true = [11, 20,19,17,10,24,23] ## Target or actual Value `

`# y_pred =[12,18,19.5,18,9,23,24] ## Predicted value by ANN model `
> - MSE 
> - MAE 
> - MBE 
> - Huber Loss 
> - Epsilon Hinge Loss 
> - Square Epsilon Hinge Loss 
> - verify each loss value by analytical method 

##### Q.4. Write python code to compute the values of following Binary classification functions for the given true output and predicted output by network:

`y_pred = [0.99, 0.11, 0.11, 0.99, 0.11, 0.11, 0.99, 0.99, 0.99, 0.11, 0.99, 0.99, 0.11, 0.99, 0.99] ## Predicted value by ANN model `

`y_true = [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 1] ## Target or actual Value `

 
> - Binary cross entropy loss 
> - Jaccard Loss 
> - Dice loss 
> - verify each loss value by analytical method

##### Q.5. Let us consider the given data points (X, Y): (1, 1), (2,1), (3,2), (4,2), (5,4) and the equation of the line passing through origin Y=0.7*X - 0.1. 
> 1. Plot the graph between X and Y using Python coding 
> 2. Calculate the predicted output using above mentioned line equation using Python coding and verify the results with analytical method. 
> 3. Calculate the MSE value using Python coding and verify the value with the analytical method. 
> 4. Plot the regression line with Python coding 

##### Q.6. The neural network shown in Fig.1 has the following hyper parameters and input: Choose random weights of the neuron and bias=0, learning rate =0.01 and inputs to the neuron and target values are as follows.

|X1| X2 |Y(target) |
|---|---|---|
|4 |1 |2 |
|2 |8 |-14 |
|1 |0 |1 |
|3 |2 |-1 |
|1 |4 |-7 |
|6 |7 |-8 |

> Write a python code for calculating the output of neural network using Gradient Descent algorithm

---
### Assignment 2

##### Q.1. Write python code from scratch for simple Linear Regression problem, the following training data are given. 
`
X = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6] 
Y = [5.1, 6.1, 6.9, 7.8, 9.2, 9.9, 11.5, 12, 12.8] `

The model Y as a linear function of X 
1. Use batch gradient descent learning algorithm to learn model parameters for α = 0.01 choose random values of weights and bias and epochs=1000. Use MSE as loss function with an appropriate convergence criterion. 
2. Plot cost function (J) for the learning duration 
3. plot the regression line 
4. repeat (2) to (3) for stochastic gradient descent and Adam optimization algorithm
5. Plot comparative loss curve 

##### Q.2. The neural network shown in Fig.1 has the following hyper parameters and input: Choose random weights and bias of the neuron and learning rate =0.01 and inputs to the neuron and target 
values are as follows.

|X1 |X2 |Y(target) |
|---|---|---|
|4 |1 |2 |
|2 |8 |-14| 
|1 |0 |1 |
|3 |2 |-1 |
|1 |4 |-7 |
|6 |7 |-8 |

1. Write a python code for predicted the output of neural network for given set of inputs using Stochastic Gradient Descent algorithm for the loss functions: 
    1. Mean Square Error 
    2. Squared Epsilon Hinge Loss
2. Plot comparative loss curve 
3. repeat (1) and Adam optimization algorithm

##### Q.3. A group of 20 students studied 0 to 6 hours for the exam. Some passed and others failed. Results are given below 

| Student | Hours studied - x | Result (0 – fail, 1 – pass) - y | 
| --- | --- | --- | 
| 1 | 0.5 | 0 | 
| 2 | 0.75 | 0 | 
| 3 | 1.00 | 0 | 
| 4 | 1.25 | 0 | 
| 5 | 1.50 | 0 | 
| 6 | 1.75 | 0 | 
| 7 | 1.75 | 1 | 
| 8 | 2.00 | 0 | 
| 9 | 2.25 | 1 | 
| 10 | 2.50 | 0 | 
| 11 | 2.75 | 1 | 
| 12 | 3.00 | 0 | 
| 13 | 3.25 | 1 | 
| 14 | 3.50 | 0 | 
| 15 | 4.00 | 1 | 
| 16 | 4.25 | 1 | 
| 17 | 4.50 | 1 | 
| 18 | 4.75 | 1 | 
| 19 | 5.00 | 1 | 
| 20 | 5.50 | 1 |

- (a). Write python code for scratch to build neural network model to determine the optimal linear 
hypothesis using linear regression to predict if a student passes or not based on the number hours 
studied with the use for stochastic gradient descent and Adam optimization algorithm with model 
parameters for α = 0.01 choose random values of weights and bias and epochs=10000. Use 
appropriate regression loss function. 
- (b). 
    - (i).Write python code from scratch to determine the optimal logistic hypothesis using logistic regression to predict if a student passes or not based on the number hours studied with the use for stochastic gradient descent with model parameters for α = 0.01 choose random values of weights and bias and epochs=40000; Loss function: Binary Cross Entropy (BCE), Threshold value=0.5 
        - (a) plot the cost function vs epoch 
        - (b) Predict pass or failed result of your designed model on random study hours enter by you. 
    - (ii) Repeat part (i) analysis with Dice Loss function. 
    - (iii)Repeat part (i) analysis with Adam optimization algorithm. 

##### Q.4. Build a model to recognize different handwritten digits from MNIST dataset by using multinomial logistic regression.Use of Adam optimization algorithm to learn model with parameters for α = 0.01, epoch = 40000 and random parameters of the model and Loss function: Softmax loss function.
- (a) Plot the cost function vs epoch 
- (b) Predict the digit of your designed model on random test data enter by you 
- (c) print confusion matrix 
- (d) calculate classification metrics such as precision,recall, f1-score and accuracy

##### Q.5. Build a model to discriminate the red, green and blue points in 2-dimensional space shown below: 

The input data and target are as follows: 

`
X=np.array([[-0.1, 1.4], 
[-0.5,0.2], 
 [1.3,0.9], 
 [-0.6,0.4], 
 [-1.6,0.2], 
 [0.2,0.2], 
 [-0.3,-0.4], 
 [0.7,-0.8], 
 [1.1,-1.5], 
 [-1.0,0.9], 
 [-0.5,1.5], 
 [-1.3,-0.4], 
 [-1.4,-1.2], 
 [-0.9,-0.7], 
 [0.4,-1.3], 
 [-0.4,0.6], 
 [0.3,-0.5], 
 [-1.6,-0.7], 
 [-0.5,-1.4], 
 [-1.0,-1.4]]) 
y=np.array ([0,0,1,0,2,1,1,1,1,0,0,2,2,2,1,0,1,2,2,2]); `

Here, 
0=red, 1=green and 2= blue dots 
In other words, given a point in 2-dimensions, x=(x1,x2), predict output either red, green or blue by 
using multinomial logistic regression. 
- (a) 
    - (i) Compare predicted results with ground truth using bar chat plot 
    - (ii) plot loss curve 
    - (iii) print confusion matrix 
    - (iv) calculate classification metrics such as precision, recall, f1-score and accuracy 
    - (v) Visualize classified data by Scatter plot. Use of gradient descent learning algorithm to learn model with parameters for α = 0.01, Softmax loss function and random parameters of the model.
- (b)repeat part (a) Use Stochastic gradient descent algorithm to learn model 
- (c) repeat part (a) with use of Adam Optimization algorithm to learn model

---
### Assignment 3

##### Q.1. Build a ANN model from scratch for predicting best housing selling prices in Boston using three features (i.e. “RM: average number of rooms per dwelling; LSTAT: percentage of population considered lower status: PTRATIO: pupil-teacher ratio by town”) of Boston dataset (Use Sklearn Dataset) by using Stochastic Gradient Descent algorithm for the loss functions: 
- (a) Mean Square Error 
- (b) Huber Loss 
- (c) Squared Epsilon Hinge Loss 
- (i) Plot comparative loss curve for at least 500 epochs.
- (ii) Print comparison of Boston housing selling prices among above mentioned loss functions using bar chart plot and which loss function is providing better housing selling prices among others.
- (iii) Implement above ANN model with Keras Library and verify the above results.

##### Q.2. Build a ANN model from to recognize breast cancer from Breast Dataset (Use Sklearn Dataset). Use Stochastic gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the parameters of the ANN model for loss functions 
- (a) Binary cross entropy 
- (b) Dice Loss 
- (i) Plot comparative loss curve for at least 200 epochs.
- (ii) Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy and ROUC curve for each loss function.
- (iii) Repeat part (ii) to (iii) using Adam gradient descent algorithm
- (iv) Implement above ANN model with Keras Library and verify the above results. 

##### Q.3. Build a ANN model from scratch to recognize diabetes-from pima-indians-diabetes-database (i.e. https://github.com/duonghuuphuc/keras/tree/master/dataset ). Use Stochastic gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the ANN model for loss functions 
- (a) Binary cross entropy 
- (b) Dice Loss 
- (i) Visualize input dataset and Plot comparative loss curve for at least 200 epochs.
- (ii) Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy and ROUC curve for each loss function.
- (iii) Repeat part (i) to (ii) using Adam gradient descent algorithm
- (iv) Implement above ANN model with Keras Library and verify the above results. 

##### Q.4. Build a ANN model from scratch to recognize Iris-setosa, Iris -virginica and Iris-versicolor from the Iris Dataset ((Use Sklearn Dataset) which contains four features (length and width of sepals and petals) of 50 samples of three species of Iris (Iris setosa, Iris virginica and Iris versicolor. For implementation, use Stochastic gradient descent algorithm to learn model with parameters for α = 0.01and random parameters of the ANN model for the Softmax loss function 
- (i)Visualize data by boxplot of Sepal Length & Sepal width and Petal Length and width for three IRIS
species. 
- (ii) Plot comparative loss curve for at least 200 epochs.
- (iii) Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy and ROUC curve
- (iv) Visualize classified data by Scatter plot
- (v) Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy and ROUC curve for each loss function.
- (vi) Repeat part (ii) to (v) using Adam gradient descent algorithm
- (vii) Implement above ANN model with Keras Library and verify the above results. 

##### Q.5. Build a ANN model from scratch to recognize human emotion using Facial emotion recognition dataset (FER2013) (https://github.com/gitshanks/fer2013). For implementation, use Stochastic gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the ANN model for the Softmax loss function 
- (i)Visualize Facial emotion recognition dataset (FER2013. 
- (ii) Plot comparative loss curve for at least 200 epochs.
- (iii)Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy and ROUC curve
- (iv) Repeat part (ii) to (iii) using Adam gradient descent algorithm
- (v) Implement above ANN model with Keras Library and verify the above result

---
### Assignment 4

##### Q.1.
- (i) Write python code for plotting the following activation functions and their derivative for the input x in the range of -20 to 20. 
    - (a) ReLU 
    - (b) LekayReLU 
    - (c) Parametric ReLU 
    - (d) Exponential ReLU(ELU) 
    - (e) Scaled Exponential Linear Units (SELU) 
    - (f) SoftPlus (Smooth ReLU) 
- (ii)
    - (a).Write python from scratch for 2D Linear convolution between input=np.array([[1,2,3],[4,5,6],[7,8,9]])and filter=np.array([[1,2,1],[0,0,0],[-1,-2,-1]]) 
    - (b).Write python from scratch for 2D Linear convolution by Toeplitz matrix method between input image(lena.jpg) and kernel = np.array([[1, 2, 1],[2, 4, 2],[1, 2, 1]]))/16 
    - (c) Compute number of multiplications and parameters required for 2D Linear Convolution in part (a) and part(b) 
    - (d) Apply Max pooling and Average pooling on convoled image in part (b) 
- (iii) 
    - (a)Write python from scratch for 2D Spatial Separbale convolution between input image(lena.jpg) and Gausian filter = np.array([1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1])/256 
    - (b) Compute number of multiplications and parameters required for2D Spatial Separbale convolution 
	
##### Q.2. Build a CNN(LeNet5) model from scratch to recognize handwritten digit from the optical handwritten digit dataset(Use Sklearn Dataset and split dataset into training dataset (80%) and testing dataset (20%) ). Use a Stochastic gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the CNN model for 
- (a) Softmax loss function 
- (b) Focal loss function.
    - (i) Plot a comparative loss curve for at least 200 epochs. 
    - (ii) Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy on test datset and ROC curve for each loss function. 
    - (iii) Repeat part (i) to (ii) using an Adam gradient descent algorithm 
    - (iv) Implement above CNN model with Keras/Tensorflow/Pytorch Library and verify the above results. 
	
##### Q.3. Build CNN(LeNet5) model from scratch to recognize diabetes-fromPima-Indians-diabetesdatabase (i.e. https://github.com/duonghuuphuc/keras/tree/master/dataset ). Use Adamgradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of theCNN model for Binary cross entropy loss function. 
- (i) Visualize input dataset and Plot comparative loss curve for at least 200 epochs. 
- (ii) Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy on test datsetand ROC curve for each loss function. 
- (iii) Implement above CNN model with Keras/Tensorflow/Pytorch Library and verify the above results.

##### Q.4. Build a CNNmodel from scratch to recognize human emotion using Facial emotionrecognition dataset (FER2013) (https://github.com/gitshanks/fer2013)(split dataset into training dataset (80%) and testing dataset (20%) ). For implementation, use a Adam gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the CNN model forthe Softmax loss function 
- (i)Visualize Facial emotion recognition dataset (FER2013). 
- (ii) Plot a comparative loss curve for at least 200 epochs. 
- (iii)Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy on test dataset and ROC curve 
- (iv) Implement above CNN model withKeras/Tensorflow/Pytorch Library and verify the above results.

---
### Assignment 5

##### Q.1. Build a  1D CNN model from scratch to recognize human activity using HAR dataset  (https://archive.ics.uci.edu/ml/datasets/human+activity+recognition+using+smartphones) (split dataset into training dataset (80%) and  testing dataset (20%) ). For implementation, use a Stochastic gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the CNN model for Focal loss function
- (i)Visualize HAR dataset
- (ii) Plot a comparative loss curve for at least 200 epochs.
- (iii)Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy on test dataset and ROC curve
- (iv)  Implement above 1D CNN model with Keras Library and verify the above results. 

##### Q.2. Python Implementation  of LSTM  from scratch  for   Binary to  Octal  conversion. 

##### Q.3. Implement A CNN-RNN (LSTM)Framework  using  (Keras/Tensorflow/Pytorch) for CropYield Prediction (reference  paper, and dataset attached and other related  dataset is available at https://github.com/saeedkhaki92/CNN-RNN-Yield-Prediction) . For implementation, use a Stochastic gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the CNN model for  RMSE as l loss function
- (i) Plot a comparative loss curve for at least 100 epochs.
- (ii) Compare  for Soybean  and Corn  yield prediction performance (RMSE and Correlation Coefﬁcient)  for years 2016, 2017,and 2018 of 1D CNN, RNN(LSTM), and CNN- RNN(LSTM)

##### Q.4. Python Implementation  of GRU  from scratch  for   Binary to Deceimal conversion. 

##### Q.5. Build a CNN(AleNet)model  (using Keras/Tensorflow/Pytorch)  to recognize breast cancer from Breast Dataset (Use Sklearn Dataset and  split dataset into training dataset (80%) and  testing dataset (20%) ). Use an Adam gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the parameters of the  CNNmodel for Focal loss function.
- (i) Plot a comparative loss curve for at least 200 epochs.
- (ii) Print confusion matrix, calculate classification metrics such as precision, recall, f1-score and accuracy on test datset and ROC curve for each loss function.

---
### Assignment 6

##### Q.1. Implement CNN(VGG16) model using Keras/Tensorflow library for Steel Defect Detection from dataset (https://www.kaggle.com/c/severstal-steel-defect-detection/data). For implementation, split dataset into training dataset (80%) and testing dataset (20%) ). Use a Adam gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the CNN model for Binary cross entropy loss function. For better training performance, you can use batch-normalization and dropout if necessary.
- (i)Plot a comparative loss curve for at least 50 epochs 
- (ii) Print confusion matrix, calculate classification metrics such as precision, recall, F1-score, IoU and accuracy on test dataset and ROC curve. 

##### Q.2. Implement CNN(VGG16) model for Predicting Invasive Ductal Carcinoma (IDC) in Breast Cancer Histology Images (https://www.kaggle.com/paultimothymooney/breast-histopathology-images). For implementation, split dataset into training dataset (80%) and testing dataset (20%) ). Use an Adam gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the parameters of the CNN model for Focal loss function. For better training performance, you can use batch-normalization and dropout if necessary. 
- (i) Plot a comparative loss curve for at least 50 epochs. 
- (ii)Print confusion matrix, calculate classification metrics such as precision, recall, F1-score, IoU and accuracy on test dataset and ROC curve. 

##### Q.3. Implement FCN8 model using Keras/Tensorflow library for Building footprint segmentation from aerial remote sensing images (https://github.com/menvuthy/building-footprint-dataset). For implementation, split dataset into training dataset (80%) and testing dataset (20%) ). Use a Adam gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the CNN model for Dice Loss function. For better training performance, you can use batch-normalization and dropout if necessary. 
- (i) Plot a comparative loss curve for at least 50 epochs. 
- (ii) Print confusion matrix, calculate overall as well as classwise classification metrics such as bulding and Background accuracy, Segmentation Accuracy, Dice Coefficient, IoU) on test dataset. 

##### Q.4.Implement U-Net model for the multi-class schematic segmentation of aerial images (Dataset: https://www.kaggle.com/humansintheloop/semantic-segmentation-of-aerial-imagery). Use an Adam gradient descent algorithm to learn model with parameters for α = 0.01 and random parameters of the model for Focal loss function. For better training performance, you can use batch-normalization and dropout if necessary. 
- (i)Plot a comparative loss curve for at least 50 epochs. 
- (ii)Print confusion matrix, calculate overall as well as class-wise classification metrics such as precision, recall, F1-score, IoU, Accuracy on test datset. 

##### Q.5.
- (a)Write python from scratch for 2D Depthwise Separbale convolution between input image(lena.jpg) and depthwise and pointwise filters are as follows and also compute number of multiplications and parameters required for 2D Depthwise Separbale convolutions depthwise filter array = np.array([[[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]], [[1, 2, 1], [2, 4, 2], [1, 2, 1]]])/16 pointwise filter array = np.array([[1], [1], [1]])/512 
- (b) Write python from scratch for 2D Atrous/Dilated cnvolution between input image(lena.jpg) and filter which is given below with rate 3: filter= np.array([[1, 2, 1],[2, 4, 2], [1, 2, 1]])/16

---


# MNIST_FASHION_CLASSIFICATION
Classifying images using different models and try the best for getting the perfect accuracy.
# Introduction
This project implements some ML algorithm towards learning the classification of Images in our case the images are taken from the MNIST Fashion data, the main problem we addressed in the project is to obtain a model that classify the visual images with a good accuracy, for this we draw a base line, this base line is the accuracy of the baseline model which is the KNN, the idea is to use other algorithms that will result in better accuracies than the drawn baseline model.

# KNN
KNN is the first implemented model into classifying these images, before running the model, some parameters should be determined such as the number of neighbors for classifying the unseen data, giving weight or not to some parameters and the most important parameter is the metric used for calculating the distance between the point and its neighbors as these parameters play a vital role into the accuracy of the model one needs to use them well not arbitrarily.

# Random Forrests
After drawing the baseline model, the search for more complex models is needed, not for the sake of their complexity but because they tend most of the time to be more accurate, there came the Random Forrest, because of its robust into the classification tasks it is chosen in this project, and because of the methodology of ensemble learning it makes this model better than the original model Decision Trees.

# SVM
This model is quite known in the classification of images, and hence the good accuracy it results in classification, and it is useful in data with high number of features, and in our case the number of features in the MNIST Fashion is quite high 784 features for each image, but the parameters of this model need to be chosen carefully.
Logistic Regression
Although it is mentioned in the report, but it is not used and removed, because after trying hard to use the multi-class Log. R. it didn’t pass the accuracy of the base model, hence no need to use it because of the low accuracy in comparing with the other models used.
# Evaluation Metric
The evaluation metric to specify how good is the model, is the accuracy in this report, because it calculates how good the model predicts the classification of the image, first of all evaluating the models using the validation sets 20k, then evaluate the accuracy on the unseen data 10k test from the MNIST Fashion images, also visualizing the images helps in making observations and results.
# GUI
An interface is implemented and developed for the ease of use of the models, it is developed using tkinter library in python. 
# Technical Details and Evaluation
The models used here are built in models from the sklearn library, and the MNIST Fashion data are downloaded from their GitHub repository, and the HOG is used from the skimage library.
First of all, making a random distribution for the data which is known as shuffling the data, with a specific random seed, to be consistent with evaluating all the models, then we made a visualization for some of the data distribution.

 
# Validation
The validation set is created by splitting the 60k training examples into 40k as a train set and 20k as the validation set, also the data is shuffled well using random function.
# KNN
As mentioned KNN here is the base model, and we have tried multiple set of parameters to get different tuning parameters, we examined for the metric parameters two metrics “Manhattan” and “Euclidean”, also the number of neighbors used are 1 and 3 neighbors, the number of neighbors specifies the number of the closest points that have the vote in predicting the value of the new point.

# Conclusions and Discussions

After deriving multiple machine learning classification algorithms, it appears that some will work better than others with the image classifications, also maybe with specifying the parameters more carefully and accurately the accuracy can be above 0.9 which is almost perfect for classification, or using more advanced models from the deep learning models.
Some limitations happened while classification, such as some classes have very low accuracy as in class 6 “shirt”, this is a big problem as it makes the overall accuracy lower, some corrections could be used to fix this such as giving more weights to specific pixels, or blurring the image before using it or using any possible feature that fixes the low accuracy of some classes and at the same time improving the accuracy for the other classes.

Affect of applying the new features
After applying new features which are the transform of the images using HOG transform, the accuracy for the baseline model and the RF changed, but for the Random forest there is just a slight fall in accuracy which can be neglected, and the base model achieved high accuracy 0.87 which may not consider as big or drastic change, but it is good, and can be using alongside with the HOG transform to classify images and obtain high accuracy.


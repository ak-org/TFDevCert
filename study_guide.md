
# Overview 
## Introduction
This study aid will help you prepare for Tensorflow Developer Certification test. You should expect to spend 8-16 hours per week depending on your skills and experience.  

## Certification Test Highlights

* 100% coding test. No single/multiple choice questions
* **The exam is open book**. You can access any resource on the internet necessary during the exam. Examples of resources : tensorflow.org, stackflow, blog posts, tutorials etc.
* You have 5 hours to complete 5 coding questions
* Solution model for each question can earn score betweeen 0 and 5 on scale of 5. 
  * Score of each question is **not** weighted equally
* First question is a warm question and subsequent questions will get harder
* You do not need GPU to train the models. A local machine with 4+ cores and 8+ GB memory is sufficient to take the test
* Do not forget to submit at the end of the exam

## Recommended Self Learning resources
1. The famous ML Book https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/ (Ch:4,10-16)
2. Machine Learning Foundations https://goo.gle/ml-foundations
3. Udacity Course https://www.udacity.com/course/intro-to-tensorflow-for-deep-learning--ud187
4. Coursera Course https://www.coursera.org/professional-certificates/tensorflow-in-practice 
5. ML Zero to Hero (First 4 videos) https://www.youtube.com/playlist?list=PLZKsYDC2S5rM6yKBs5ParXS6RWda6iAnK
6. NLP Zero to Hero https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S
7. Hello world in TF https://developers.google.com/codelabs/tensorflow-1-helloworld#0
8. https://www.LearnTensorFlow.io 
9. Tensorflow Tutorials https://www.tensorflow.org/tutorials 
10. Tensorflow Guide https://www.tensorflow.org/guide 

## Paid Courses
If you decide to take one of these paid courses, you should still finish the weekly assignments listed in the study plan.

1. https://www.coursera.org/professional-certificates/tensorflow-in-practice (Recommended by Google Tensorflow Team)
2. https://www.TFCertification.com 
3. https://www.udemy.com/course/tensorflow-developer-certificate-machine-learning-zero-to-mastery/ 



## Pre-requisites 
* 12+ months of Python coding experience
* Comfortable with Notebooks, IDE such as PyCharm or Visual Studio
### Nice to have
* Experience with TensorFlow
* Basic understanding of following concepts
    * Multiclass vs binary classification
    * Regression
    * Regularization
    * Overfitting, Underfitting
    * Weights, biases
    * Loss
    * Optimizer
    * Saving and loading a machine learning model

# Study Plan

## Study Plan for Experienced 
Following this study plan if you have good grasp of Python programing and you are familiar with concepts of Machine Learning.
### **Week 0 - Learn about certification and setup dev environment**
* Review the [certification handbook]
* Review the [Certificate Information] 
* [Setup Dev environment] on your local machine  
**Feel free to skip next two steps, if you already have a colab environment setup.**
* Google colab tutorial https://www.youtube.com/watch?v=inN8seMm7UI
* Access colab at https://colab.research.google.com/ (Use personal google account)

### **Week 1 - Linear Regression Model**
* Review the [regression] tutorial and run the code in your colab environment
* Review and run the code of the [Second regression example]
* Learn to save and load model by following this [save keras model] tutorial
    ### Assignment for Week 1
    Develop linear regression models using Tensorflow Keras Sequential APIs. Train and evaluate linear regression models on following datasets:
    * [Red wine quality] dataset details
        * [Red wine csv] data file
        * [White wine csv] data file
    * [Boston Housing] dataset details
        * [Boston Housing csv] data file


### **Week 2 - Deep Neural Network**
* Review the [basic classification] tutorial and run code in your colab environment
* Explore the [tensorflow datasets] documentation
* Review the [tfdata] tutorial and run code in your colab environment
    ### No Assignment for Week 2
### **Week 3 - Convolutional Neural Network for Computer Vision**
* Watch Episode #2-7 and #9 of [Machine Learning Foundations]
* Review the [Convolutional Neural Networks and Computer Vision with TensorFlow] article and code in your colab environment
    ### Assignment for Week 3
    * Develop a Image classifer for these datasets: 
        * [horse-or-human] - download and load data from disk
        * [beans] - Use tf.data APIs to load data
        * [cifar10] - use Keras API to load data


### **Week 4 - Tokenization and Sequencing for NLP**
* Watch Episode #8-10 of [Machine Learning Foundations]
* Watch [NLP Zero to Hero] playlist
* Review the [text classification tutorial] and run code in your colab environment
* Review the [Natural Language Processing with TensorFlow] article and code in your colab environment
    ### Assignment for Week 4
    * Develop a NLP classifer for these datasets: 
        * [imdb_reviews] - Use tf.data APIs to load data
        * [sentiment140] - Use tf.data APIs to load data
        * [clickbait] - download and load data from disk
### **Week 5 - Time Series data Modeling**
* Review the [time series forecasting tutorial] and run code in your colab environment
* Review the [Time series data Processing with TensorFlow] article and code in your colab environment
    ### Assignment for Week 5
    * Develop a machine model to predict on time series data: 
        * [apple] stock price data - download and load data from disk
        * [sunspot] data - download and load data from disk
        * [sensor] data - download and load from disk


**CHECKPOINT**
```
You should be comfortable with topics covered in the certification exam. 

After this week, you should be exclusively practising your coding skills in PyCharm IDE. 

Please note that you will take certification test within PyCharm IDE.
```
### **Week 6 - Practice exercises**
Ensure that your PyCharm environment is compliant with instructions provided in the [Setup Dev environment] guide. Complete following tasks in your PyCharm IDE.
* Run your solution of assignment from week 1 and 3. 
* Run code from [image_augmentation] tutorial
### **Week 7 - Practice exercises (contd.)**
Ensure that your PyCharm environment is compliant with instructions provided in the [Setup Dev environment] guide. Complete following tasks in your PyCharm IDE.
* Run your solution of assignment from week 4 and 5 in your PyCharm IDE. 
### **Week 8 - Take the certification test**
* Follow the [Exam] link to register for the exam. Follow instructions on the website to setup test environment and take the test


## Study Plan for Beginners 
Following this study plan if you have good grasp of Python programing but you are not familiar with machine learning concepts

If you are not familiar with colab, follow the links below and learn about colab and setup your environment

* Google colab tutorial https://www.youtube.com/watch?v=inN8seMm7UI
* Access colab at https://colab.research.google.com/ (Use personal google account)

### **Week 1 - Introduction to machine learning and TensorFlow**
In the first week, You will learn about fundamentals of machine learning and tensorflow

* Watch Episode #1 of [Machine Learning Foundations]
* Read about [fundamentals of neural networks]
* Watch part 1 of 2 in this [videos] playlist

### **Week 2 - Continue to Learn Tensorflow 2**
In the second week, you will continue to build on learnings from previous week. 
* Watch part 2 of 2 in this [videos] playlist
* Go through the [Getting started with TensorFlow: A guide to the fundamentals] tutorials
* Run the code clicking on "Open in Colab" button at the top of the article.
* Go through the [TensorFlow 2.x quick start] tutorial
* Run the code by clicking on Run in Google colab" button at the top of the article.

By the end of week 2, you should have basic understanding of Machine learning concepts. Once you have completed this section, follow the study plan in the **Study Plan for Experienced**  section









[sentiment140]: https://www.tensorflow.org/datasets/catalog/sentiment140 "sentiment140"
[imdb_reviews]: https://www.tensorflow.org/datasets/catalog/imdb_reviews "imdb_reviews"
[Certificate Information]: https://www.tensorflow.org/certificate "Certificate Information"
[Certification Handbook]: https://www.tensorflow.org/extras/cert/TF_Certificate_Candidate_Handbook.pdf "Certification Handbook"
[Setup Dev environment]: https://www.tensorflow.org/extras/cert/Setting_Up_TF_Developer_Certificate_Exam.pdf "Setup Dev environment"
[Regression]: https://www.tensorflow.org/tutorials/keras/regression "Regression"
[Second regression example]: https://dev.mrdbourke.com/tensorflow-deep-learning/01_neural_network_regression_in_tensorflow/ "Second regression example"
[colab tutorial]: https://www.youtube.com/watch?v=inN8seMm7UI
[red wine quality]: https://archive.ics.uci.edu/ml/datasets/wine+quality/ "red wine quality"
[Boston Housing]: https://www.cs.toronto.edu/~delve/data/boston/bostonDetail.html "Boston Housing"
[Machine Learning Foundations]: https://goo.gle/ml-foundations "Machine Learning Foundations"
[Tensorflow Fundamentals]: https://dev.mrdbourke.com/tensorflow-deep-learning/00_tensorflow_fundamentals/
[TensorFlow 2.x quick start]: https://www.tensorflow.org/tutorials/quickstart/beginner " TensorFlow 2.x quick start"
[fundamentals of neural networks]: https://ml-cheatsheet.readthedocs.io/en/latest/nn_concepts.html "fundamentals of neural networks"
[Getting started with TensorFlow: A guide to the fundamentals]: https://dev.mrdbourke.com/tensorflow-deep-learning/00_tensorflow_fundamentals/ "Getting started with TensorFlow: A guide to the fundamentals"
[Convolutional Neural Networks and Computer Vision with TensorFlow]: https://dev.mrdbourke.com/tensorflow-deep-learning/03_convolutional_neural_networks_in_tensorflow/ "Convolutional Neural Networks and Computer Vision with TensorFlow"




[Natural Language Processing with TensorFlow]: https://dev.mrdbourke.com/tensorflow-deep-learning/08_introduction_to_nlp_in_tensorflow/ "Natural Language Processing with TensorFlow"
[Time series data Processing with TensorFlow]: https://dev.mrdbourke.com/tensorflow-deep-learning/10_time_series_forecasting_in_tensorflow/ "Time series data  Processing with TensorFlow"
[videos]: https://www.youtube.com/playlist?list=PL6vjgQ2-qJFfU2vF6-lG9DlSa4tROkzt9 "videos"
[NLP Zero to Hero]: https://www.youtube.com/playlist?list=PLQY2H8rRoyvzDbLUZkbudP-MFQZwNmU4S "NLP Zero to Hero"
[text classification tutorial]: https://www.tensorflow.org/tutorials/keras/text_classification "text classification tutorial"
[time series forecasting tutorial]: https://www.tensorflow.org/tutorials/structured_data/time_series "time series forecasting tutorial"
[basic classification]: https://www.tensorflow.org/tutorials/keras/classification "basic classification"
[quick start for beginners]: https://www.tensorflow.org/tutorials/keras/classification "quick start for beginners"
[tensorflow datasets]: https://www.tensorflow.org/datasets "tensorflow datasets"
[tfdata]: https://www.tensorflow.org/guide/data "tfdata"
[save keras model]: https://www.tensorflow.org/guide/keras/save_and_serialize "save keras model"
[beans]: https://www.tensorflow.org/datasets/catalog/beans "beans"
[cifar10]: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/cifar10/load_data "cifar10"
[horse-or-human]: https://www.kaggle.com/sanikamal/horses-or-humans-dataset "horse-or-human"
[clickbait]: https://github.com/bhargaviparanjape/clickbait/tree/master/dataset "clickbait"
[exam]: https://app.trueability.com/google-certificates/tensorflow-developer "exam"
[apple]: https://github.com/matplotlib/sample_data/blob/master/aapl.csv "apple"
[sunspot]: https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv "sunspot"
[sensor]: https://www.kaggle.com/taranvee/smart-home-dataset-with-weather-information "sensor"
[image_augmentation]: https://www.tensorflow.org/tutorials/images/data_augmentation "image_augmentation"
[Red wine csv]: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv "red wine csv"
[White wine csv]: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv "white wine csv"
[Boston Housing csv]: http://course1.winona.edu/bdeppa/Stat%20425/Data/Boston_Housing.csv "Boston Housing csv"

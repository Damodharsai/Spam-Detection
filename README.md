# ML Spam Detection

![Kaggle](https://img.shields.io/badge/Dataset-Kaggle-blue.svg) ![Python 3.6](https://img.shields.io/badge/Python-3.6-brightgreen.svg) ![Scikit-Learn](https://img.shields.io/badge/Libraries-ScikitLearn-orange.svg)![NLTK](https://img.shields.io/badge/NLTK-coral.svg)

This repository consists of files required for end to end implementation and deployment of Machine Learning Spam Detection web application created with flask and deployed on the Heroku platform.

## Table of Contents
  * [App Link](#app-link)
  * [About the App](#about-the-app)
  * [Deployement on Heroku](#deployement-on-heroku)
  * [Technologies Used](#technologies-used)


## App Link
If you want to view the deployed model, click on the following link:<br />
[https://spamdetectordamodharsai.herokuapp.com/](https://spamdetectordamodharsai.herokuapp.com/)

A glimpse of the web app:

![image](https://user-images.githubusercontent.com/78199382/113012232-34185b00-9198-11eb-9c9e-128f365cc3e3.png)

## About the App
The ML Spam Detection is a Flask web application which predicts whether the message is a spam or not. SMS Spam Collection dataset from Kaggle was used to classify the messages into 2 classes- Ham(1) and Spam(0) using stemming, Bag of Words model and Naive Bayes Classifiers.

Note:The dataset is an unbalanced dataset and therefore, for this situation the role of Precision becomes quite important.Precision is more focused in the positive class than in the negative class, it actually measures the probability of correct detection of positive values,

Consider the following case scenario -'suppose if the message is not a spam and if it's been predicted by the model as a spam, the consumer is going to miss that message.' So, for this type of unbalanced dataset, precision defined as {TP/(TP+FP)} plays an important role along with accuracy_score. My objective was to reduce the FP(False Positive) value as much as possible for this case and in order to overcome this issue, Naive Bayes classifiers namely, MultinomiallNB and BernoulliNB were implemented to get best accuracy_score and precision_score from the dataset.

The code is written in Python 3.6.10. If you don't have Python installed, you can find it [here](https://www.python.org/downloads/). If you are using a lower version of Python you can upgrade using the pip package, ensuring you have the latest version of pip. To install the required packages and libraries, run this command in the project directory after [cloning](https://www.howtogeek.com/451360/how-to-clone-a-github-repository/) the repository:
```bash
pip install -r requirements.txt
```

## Deployement on Heroku
Login or signup in order to create virtual app. You can either connect your github profile or download ctl to manually to deploy this project.

[![](https://i.imgur.com/dKmlpqX.png)](https://heroku.com)

The next step would be to follow the instruction given in the [Heroku Documentation](https://devcenter.heroku.com/articles/getting-started-with-python) to deploy a web app.

## Technologies Used

![](https://forthebadge.com/images/badges/made-with-python.svg)

[<img target="_blank" src="https://flask.palletsprojects.com/en/1.1.x/_images/flask-logo.png" width=170>](https://flask.palletsprojects.com/en/1.1.x/) [<img target="_blank" src="https://number1.co.za/wp-content/uploads/2017/10/gunicorn_logo-300x85.png" width=280>](https://gunicorn.org) [<img target="_blank" src="https://scikit-learn.org/stable/_static/scikit-learn-logo-small.png" width=200>](https://scikit-learn.org/stable/) 



## Table of Contents

1. [Installation and Libraries](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation and Libraries <a name="installation"></a>

Python 3* and sklearn components were used in creating this project. I also Used nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger']) for the classifier model training.

Used libraries: pandas, numpy, nltk, json, sklearn, sqlite3, sqlalchemy, plotly, pickle

## Project Motivation<a name="motivation"></a>

In this project, I used disaster message data from Figure Eight to build a model for an API that classifies disaster messages in order to:

1. Create a machine learning pipeline to categorize events to ensure people can send the messages to an appropriate disaster relief agency.
2. Create a web app where an emergency worker can input a new message and get classification results in several categories. 

## File Descriptions <a name="files"></a>

There are five main files available here:

1. /data/Process_data.py - ETL data pipeline which is to Extract, Transform, and Load clean data for building classification ML model. It will read the dataset, clean the data, and then store it in a SQLite database. 
2. /models/train_classifier.py - ML pipeline which will split the data into a training set and a test set. Then, it will create a machine learning pipeline that uses NLTK, as well as scikit-learn's Pipeline and GridSearchCV to output a final model, finally, it will export the model to a pickle file.
3. /app/run.py - This will run the web app display the dashboard.
4. /app/templates/master.html - HTML file creating the main page for a Flask Web App.
5. /app/templates/go.html - HTML file creating the function in the web app for classifying new messages in categories.

I used both markdown cells and # in the code to help walk through the processes of individual steps.

## Results<a name="results"></a>
A flask web app dash board can be run on the local host, which can show basic visulization of the training data and disaster response message filter.

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

I used disaster data from Figure Eight to build a model for an API that classifies disaster messages

# Disaster Response Pipeline Project


### Project Overview

In the Project Workspace, we'll woek with real messages that were sent during disaster events. We will be creating a machine learning pipeline to categorize these events so that you can send the messages to an appropriate disaster relief agency.

### ETL 

ETL Pipeline , process_data.py, write a data cleaning pipeline that: Loads the messages and categories datasets Merges the two datasets Cleans the data Stores it in a SQLite database

### ML

ML Pipeline , train_classifier.py, write a machine learning pipeline that: Loads data from the SQLite database Splits the dataset into training and test sets Builds a text processing and machine learning pipeline Trains and tunes a model using GridSearchCV Outputs results on the test set Exports the final model as a pickle file




### Required packages:

- flask
- joblib
- pandas
- plot.ly
- numpy
- scikit-learn
- sqlalchemy

### Files:
```
- app
| - template
| |- master.html  # main page of web app
| |- go.html  # classification result page of web app
|- run.py  # Flask file that runs app

- data
|- disaster_categories.csv  # data 
|- disaster_messages.csv  # data
|- process_data.py
|- DisasterResponse.db   # database 

- models
|- train_classifier.py
|- classifier.pkl  # ML model

- notebooks
|- ETL Pipeline Preparation.ipynb # etl code
|- ML Pipeline Preparation.ipynb # ml code

```

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
		
2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Screenshots

![Screenshot #1](https://i.imgur.com/EU9FdSP.png)


![Screenshot #2](https://i.imgur.com/YsucjD9.png)



### Acknowledgement

Thanks Udacity for providing this platform and Figure-Eight for providing the dataset.





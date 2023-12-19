# Disaster Response Pipeline Project

### Introduction 

This project aim is to build a Natural Language Processing tool to classify pre-labeled disaster messages across 36 categories. This project will help disaster response agencies effectively to categorize incoming messages during critical situations,
This project is also includes a web application where disaster response worker can input messages received and get classification results


Project is divided into 3 key sections:

 1) Building an ETL pipeline to extract, clean and save the data in a SQLite Database from following 2 csv files
	 a.disaster_messages.csv: Messages data.
 	 b.disaster_categories.csv: Disaster categories of messages.
 2) Building a ML pipeline to train our model
 3) Run a Web App to show our model results
 
 #Repository Description
        Disaster_Response
       -/home/workspace
           - app
                - templates
                    - go.html
                    - master.html
                - run.py
          - data
                - disaster_message.csv
                - disaster_categories.csv
                - DisasterResponse.db
                - process_data.py
          - models
                - classifier.pkl
                - train_classifier.py
                
          |-- README


#Building an ETL pipeline
 
process_data.py : ETL Pipeline will merge and cleaning the data:
 Loads the messages and categories datasets
 Merges the two datasets
 Cleans the data
 Stores it in a SQLite database
 
## Building a ML pipeline
train_classifier.py : script write a machine learning pipeline that:

  Loads data from the SQLite database
  Splits the dataset into training and test sets
  Builds a text processing and machine learning pipeline
  Trains and tunes a model using GridSearchCV
  Outputs results on the test set
  Exports the final model as a pickle file
 
##Run web app
 run.py : 
 File to run Flask app that classifies messages based on the model and shows data visualizations.

### Instructions to run the code:
   require following pip installations :
   pip install SQLAlchemy
   pip install nltk
### Used Libraries :
   SQLlite Libraries: SQLalchemy
   NLP Libraries: NLTK
   ML Libraries: NumPy, Pandas, SciPy, SkLearn
   Model Loading and Saving Library: Pickle
   Web App and Visualization: Flask, Plotly

 
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run `python run.py` command in the app's directory to run your web app.
3. Go to http://0.0.0.0:3000/



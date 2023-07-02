# Disaster Response Pipeline Project

### Purpose
The purpose of this project is to predict the purpose of incoming messages during a disaster among many categories, to include requests for aid, food, water, medical, etc. 

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Notes
The current model relies soley on the incoming message after translation into English, not its source or the original message.  Attempts to incorporate the source did not improve the model. 

Additional research into incorporating teh original language into the model may result in an improved model.

The source data, code for formatting the data, and training a model are i nthe repositoy.  A pre-trained model is also available.  See /models/classifier.pkl for the pretrained model.

### Data Source
Data was sourced from Udacity and Figure Eight.

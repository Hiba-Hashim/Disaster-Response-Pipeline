# Disaster Response Pipeline Project
In this project I analyzed disaster data from Figure Eight to build a model for an API that classifies disaster messages.
The data set contain real messages that were sent during disaster events than it passed through a machine learning pipeline to categorize them so that you can send the messages to an appropriate disaster relief agency.

### Overview of the Dataset:
- There are 36 diffrent target categories. If the "related" value is 0 than no category is chossen.
- Some Messages requesting help and some are not. There are 4513 messages that requested help, and 21873 messages are not.
- Each message belong to one of the three genres(Direct, Social, News).
#### ETL pipeline:
Preprocessing is done in `data/process_data.py` file, where:
- Data get loaded from the csv files `data/disaster_messages.csv` and `data/disaster_categories.csv'.
- Merge both the messages and the categories datasets.
- Clean the merged data by removing Duplicated.
- Store clean merged data in `data/DisasterResponse.db`
#### Machine Learning pipeline:
ML pipeline is implemented in `models/train_classifier.py` where:
- Data get loaded from `data/DisasterResponse.db`.
- Data is split into trainging and testing sets.
- Implemented a function tokenize() to clean the messages data and tokenize it.
- Implemented Pipelines for text and machine learning processing.
- Select Parameters based on GridSearchCV.
- Store the trained classifier in `models/classifier.pkl`.
#### Flask app
Flask app is implemented in the app folder.
- Main page allows the user to write a message in the text box.
- Then its output the related categories.
### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

4. Run 'env|grep WORK' in a new terminal

5. Use the line below as a gide to write the URL inorder to access the web app

   https://WORKSPACEID-3001.WORKSPACEDOMAIN

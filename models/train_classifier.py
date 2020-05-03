#load dependencies

#imports related to storage retrieval
import sys
import pandas as pd
from sqlalchemy import create_engine
import pickle

#imports related to NLP
import re
import nltk
nltk.download(['punkt', 'stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

#imports related to ML and pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

def load_data(database_filepath):
    """
    Loads the data from the SQLite database defined by process_data.py
    input: string representing filepath to the database
    output: pandas series X (features), dataframe Y (class labels), dataframe representing disaster category
    """
    # establishes connection to database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('DisasterResponse.db',engine)#filename should match process_data
    X = df['message']
    Y = df.iloc[:,4:]
    category_names = df.columns[4::] #stimmt das so?

    return X,Y,category_names


def tokenize(text):
    """
    Tokenizes raw text messages.
    input: string containing text message.
    output: list containing tokenized text ready for the ML model.
    """
    # normalize case and remove punctuation
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    tokens = word_tokenize(text)

    # lemmatize and remove (english) stop words using list comprehension
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    cleanTokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return cleanTokens


def build_model():
    """
    Builds the classification ML model for the tokenized messages.
    input: None
    output: Python object representing the classification model.
    """
    pipeline = Pipeline([
    ('vect', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=100,min_samples_split=2)))
    ])

    #Parameters which are employed in gridsearch
    parameters = {
        'clf__estimator__n_estimators': [25,50,75,100],
        'clf__estimator__min_samples_split':[2, 4, 6],
        'clf__estimator__min_samples_leaf':[1,2,3],
        'clf__estimator__min_weight_fraction_leaf':[0.0,0.1,0.02]
    }

    #generates grid search model
    model = GridSearchCV(pipeline, param_grid=parameters,verbose=10)


    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model by comparing actual labels with predicted labels from unseen test data.
    input:Python object representing trained classification model, X (features) and Y (labels-Test data, list containing category name strings
    output: None (metrics are printed to the terminal)
    """
    #predict labels for test data
    Y_predicted = model.predict(X_test)
    #Iterate through all 36 possible labels to in order to compare prediction
    for i in range(len(category_names)):
        print("Label:",category_names[i])
        print(classification_report(Y_test.values[:, i], Y_predicted[:, i]))


def save_model(model, model_filepath):
    """
    Saves the trained model to a pickle file at provided path under highly efficient compression (reduces original file size to 1/5).
    input: Python object representing trained classification model, path for saving
    output: (None)
    """
    #compress 3 has been found to be a good compromise between speed and filesize
    joblib.dump(model, model_filepath,compress=3)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('\nPlease provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
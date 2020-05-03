import json
import plotly
import pandas as pd

import re
import nltk
nltk.download(['punkt', 'stopwords','wordnet'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Layout, Figure
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    """
    Tokenizes raw text messages.
    input: string containing text message from the user query.
    output: list containing tokenized text ready for the trained ML model.
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

# load data which was used to train and test the model
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse.db', engine)

# load the pre trained model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # generate list with plotly figure objects originating from custom functions below
    graphs = [figBarChart,figBarMessagesperCategory]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


## web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def genre_countBar(df):
    """
    Create a plotly figure of a Bar chart representing the different message channels in the data
    Input: dataset df as a pandas dataframe
    Output: plotly figure object containing the visualization
    """
    #extract data for visualization
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    data = []
    data.append(Bar(x=genre_names,y=genre_counts))
    layout = Layout(title="Number of Messages per Message Channels",
                xaxis=dict(
                    title='Message Channel',
                    tickangle=360
                ),
                yaxis=dict(
                    title='Number of Disaster Messages',
                )
            )

    return Figure(data=data, layout=layout)

def messagesPerCategory_countBar(df):
    """
    Create a plotly figure of a Bar chart representing the different messages per category.
    input: dataset df as a pandas dataframe
    output: plotly figure object containing the visualization
    """

    #select all rows from categories, summarize over them and then sort for visualization
    categories =df.iloc[:, 4:].sum().sort_values(ascending=False)

    data = []
    data.append(Bar(x=categories.index,y=categories,))
    layout = Layout(title="Number of Messages per Category",
                xaxis=dict(
                    title='Category',
                    tickangle=35
                ),
                yaxis=dict(
                    title='Number of Disaster Messages'
                )
            )
    return Figure(data=data, layout=layout)


## Execute custom functions to generate ploty figure objects for visualization
figBarChart = genre_countBar(df)
figBarMessagesperCategory = messagesPerCategory_countBar(df)

def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
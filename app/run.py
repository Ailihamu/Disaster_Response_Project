import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from sklearn.externals import joblib
from sqlalchemy import create_engine
import plotly.graph_objs 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('df_clean', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    #plot one
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    graph_one = []
    graph_one.append(plotly.graph_objs.Bar(x = genre_names, y=genre_counts))

    layout_one = dict(title = 'Distribution of Message Genres',
                xaxis = dict(title = 'Genre',),
                yaxis = dict(title = 'Count'),
                )
    
    # plot two
    graph_two = []
    graph_two.append(plotly.graph_objs.Bar(
        x = df.iloc[0:, 4:].columns.tolist(),
        y = df.iloc[0:, 4:].sum().tolist(),
          )
      )
    
    layout_two = dict(title = 'Frequency of Each Category in All Messages',
                xaxis = dict(title = 'Category',),
                yaxis = dict(title = 'total Sum'),
                )
    
    
    # plot three
    
    X = df['message']
    Y = df.iloc[0:, 4:]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    Y_pred = pd.DataFrame(model.predict(X_test), columns=Y.columns)
    
    x_fscore = []
    for col in Y.columns:
        x_fscore.append(f1_score(Y_test[col], Y_pred[col]))
        
    graph_three = []    
    graph_three.append(plotly.graph_objs.Scatter(
        x = Y.columns.tolist(),
        y = x_fscore
        ))
    
    layout_three = dict(title = 'F1 Score of Each Category',
                xaxis = dict(title = 'Category'),
                yaxis = dict(title = 'F1 Score'),
                )
    
    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))
    figures.append(dict(data=graph_two, layout=layout_two))
    figures.append(dict(data=graph_three, layout=layout_three))
    
    # encode plotly graphs in JSON
    # plot ids for the html id tag
    ids = ['figure-{}'.format(i) for i, _ in enumerate(figures)]

    # Convert the plotly figures to JSON for javascript in html template
    figuresJSON = json.dumps(figures, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html',
                           ids=ids,
                           figuresJSON=figuresJSON)


# web page that handles user query and displays model results
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


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
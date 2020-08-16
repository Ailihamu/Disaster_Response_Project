# import libraries
import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import preprocessing
from sklearn.metrics import classification_report
import pickle

def load_data(database_filepath):
    
    """
    
    Input:
    database_filepath - path of the cleaned data file 
    
    Output:
    X and Y for model training
    Category names
    
    """
    
    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql("SELECT * FROM df_clean", engine)
    X = df['message']
    Y = df.iloc[0:, 4:]
   
    category_names = Y.columns
   
    return X, Y, category_names


def tokenize(text):
    
    """
    Input: 
    text - text for tokenize
    
    output:
    clean_tokens
    
    """
    
    tokens = word_tokenize(text) #tokenize text
    
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        
    # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
   
    # build pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', RandomForestClassifier())    
    ])
    
    #selecting the main parameters for cross validation
    parameters = {
        'vect__ngram_range': ((1, 1), (1, 2)),
        'vect__max_df': (0.5, 0.75, 1.0)
      }
     
    #fit the model with grid search with choosen parameters
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model

def evaluate_model(model, X_test, Y_test, category_names):
    
    #get prediction using the trained model
    Y_pred = model.predict(X_test)
    
    #Check the scores to test the model
    Y_P = pd.DataFrame(Y_pred, columns=category_names)
    for i, cl in enumerate(category_names):
        print('Feature', i, ":", cl)
        print(classification_report(Y_test[cl], Y_P[cl]))

def save_model(model, model_filepath):
    
    #Export your model as a pickle file
    Pkl_Filename = model_filepath
    with open(Pkl_Filename, 'wb') as file:  
        pickle.dump(model, file)

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
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
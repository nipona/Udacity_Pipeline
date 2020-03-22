import sys
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
import string
import numpy as np
import joblib

nltk.download(['punkt', 'wordnet', 'stopwords'])
stop_words = nltk.corpus.stopwords.words("english")
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
remove_punc_table = str.maketrans('', '', string.punctuation)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV


def load_data(database_filepath):
    engine = create_engine('sqlite:///data/DisasterResponse.db')
    df = pd.read_sql_table('test_table4', engine)
    X = df['message']
    Y = df.drop(['message', 'genre', 'id', 'original'], axis=1)
    category_names = Y.columns.tolist()
    return X,Y,category_names



def tokenize(text):
    text = text.translate(remove_punc_table).lower()
    
    # tokenize text
    tokens = nltk.word_tokenize(text)
    
    # lemmatize and remove stop words
    return [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]


def build_model():
    forest_clf = RandomForestClassifier(n_estimators=5)
    pipeline = Pipeline([
                    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
                    ('forest', MultiOutputClassifier(forest_clf))
                    ])
    parameters = {

    'forest__estimator__n_estimators': [2,4],
    'forest__estimator__min_samples_split': [20]
    }

    cv = GridSearchCV(pipeline, parameters, cv=2, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(cv, model_filepath):
    joblib.dump(cv.best_estimator_, model_filepath)


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
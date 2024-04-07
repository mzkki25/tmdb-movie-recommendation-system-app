import pandas as pd
import keras
import nltk
import re

from keras_preprocessing.text import tokenizer_from_json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_similarity():
    data = pd.read_csv('dataset/main.csv')
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['combination'])
    similarity = cosine_similarity(count_matrix)
    return data, similarity
    
def recommends(movie):
    movie = movie.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()

    if movie not in data['movie_title'].unique():
        return 'Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies'
    else:
        movie_index = data.loc[data['movie_title'] == movie].index[0]
        similar_movies = list(enumerate(similarity[movie_index]))
        similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)[1:5]
        recommended_movies = []
        for index, _ in similar_movies:
            recommended_movie = data['movie_title'][index]
            recommended_movies.append(recommended_movie)
        return recommended_movies
    
# Konversi list of string to list ("["abc","def"]" to ["abc","def"])
def convert_list_of_str_to_list(my_list):
    return my_list.strip('[""]').split('","')

# Konversi string to list ("[1,2,3]" to [1,2,3])
def convert_str_to_list(my_list):
    return my_list.strip('[]').split(',')

def get_suggestions():
    data = pd.read_csv('dataset/main.csv')
    return list(data['movie_title'].str.capitalize())

def preprocess(text):
    stopwords = nltk.corpus.stopwords.words('english')
    stemmer = nltk.stem.PorterStemmer()
    # preprocess html tag
    text = re.sub(r'<.*?>', '', text)
    # preprocess non-alphabet
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # lowercase
    text = text.lower()
    # remove stopwords
    text = ' '.join([word for word in text.split() if word not in stopwords])
    # stemmer
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    # remove extra space
    text = re.sub(r'\s+', ' ', text)
    # remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    return text

def model():
    keras_models = keras.models.load_model('model/sentiment_analysis.h5')
    with open('model/tokenizer.json') as json_file:
        tokenizer_json = json_file.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
        
    return keras_models, tokenizer
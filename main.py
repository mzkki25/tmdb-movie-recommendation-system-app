import numpy as np
import bs4 as bs
import urllib.request as url
import tensorflow as tf

from keras_preprocessing.text import tokenizer_from_json
from keras_preprocessing.sequence import pad_sequences

from flask import Flask, render_template, request
from architecture.model import (
    recommends, 
    get_suggestions, 
    convert_list_of_str_to_list, 
    convert_str_to_list, 
    model,
    preprocess
)

app = Flask(__name__)
@app.route("/")
    
@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template(
        template_name_or_list='home.html', 
        suggestions=suggestions
    )

@app.route("/similarity", methods=["POST"])
def similarity():
    movie = request.form['name']
    if type(recommends(movie))==type('string'):
        return recommends(movie)
    else:
        m_str="---".join(recommends(movie))
        return m_str

@app.route("/recommend", methods=["POST"])
def recommend():
    keras_models, tokenizer = model()

    # Memanggil fungsi convert_to_list untuk setiap string yang perlu dikonversi menjadi list
    rec_movies = convert_list_of_str_to_list(request.form['rec_movies'])
    rec_posters = convert_list_of_str_to_list(request.form['rec_posters'])
    cast_names = convert_list_of_str_to_list(request.form['cast_names'])
    cast_chars = convert_list_of_str_to_list(request.form['cast_chars'])
    cast_profiles = convert_list_of_str_to_list(request.form['cast_profiles'])
    cast_bdays = convert_list_of_str_to_list(request.form['cast_bdays'])
    cast_bios = convert_list_of_str_to_list(request.form['cast_bios'])
    cast_places = convert_list_of_str_to_list(request.form['cast_places'])
    cast_ids = convert_str_to_list(request.form['cast_ids'])

    cast_bios = [bio.replace(r'\n', '\n').replace(r'\"', '\"') for bio in cast_bios]
    
    # Mengkombinasikan beberapa list sebagai dictionary yang dapat diteruskan ke file html sehingga dapat diproses dengan mudah dan urutan informasi akan dipertahankan
    movie_cards = {
        poster: movie 
        for poster, movie 
        in zip(rec_posters, rec_movies)
    }
    
    casts = {
        name: [cast_id, cast_char, cast_profile] 
        for name, cast_id, cast_char, cast_profile 
        in zip(cast_names, cast_ids, cast_chars, cast_profiles)
    }
    
    cast_details = {
        name: [cast_id, cast_profile, cast_bday, cast_place, cast_bio] 
        for name, cast_id, cast_profile, cast_bday, cast_place, cast_bio 
        in zip(cast_names, cast_ids, cast_profiles, cast_bdays, cast_places, cast_bios)
    }

    # web scrapping untuk mendapatkan ulasan pengguna dari situs IMDB
    source = url.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(request.form['imdb_id'])).read()
    soup = bs.BeautifulSoup(source, 'html.parser')
    soup_result = soup.find_all("div", {"class":"text show-more__control"})

    reviews_list = [] 
    persentase_reviews = [] 
    for reviews in soup_result: 
        if reviews.string: 
            reviews_list.append(reviews.string) 
            review_preprocessed = preprocess(reviews.string)
            review_pad = pad_sequences(tokenizer.texts_to_sequences([review_preprocessed]), maxlen=200)
            review_pad = tf.convert_to_tensor(review_pad)
            persentase_reviews.append(keras_models.predict(review_pad)[0])

    # Menggabungkan ulasan dan komentar menjadi ke dalam dictionary
    movie_reviews = {
        reviews_list[i]: persentase_reviews[i][0] for i in range(len(reviews_list))
    }    

    # Mengirimkan semua data ke file html
    return render_template(
        template_name_or_list='recommend.html',
        
        title=request.form['title'],
        overview=request.form['overview'],
        vote_average=request.form['rating'],
        genres=request.form['genres'],
        release_date=request.form['release_date'],
        runtime=request.form['runtime'],
        status=request.form['status'],
        
        poster=request.form['poster'],
        vote_count=request.form['vote_count'],
        
        movie_cards=movie_cards,
        reviews=movie_reviews,
        casts=casts,
        cast_details=cast_details,
    )

if __name__ == '__main__':
    app.run(debug=True)

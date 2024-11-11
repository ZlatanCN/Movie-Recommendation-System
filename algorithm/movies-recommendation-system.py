#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load data
file_path = 'dataset/TMDB_movie_dataset_v11.csv'
df = pd.read_csv(file_path)

# Preprocess data
df = df[df['vote_average'] != 0]
df.reset_index(inplace=True)
df = df.drop(['index', 'id', 'vote_count', 'status', 'release_date', 'revenue', 'backdrop_path',
              'budget', 'homepage', 'imdb_id', 'original_title', 'overview', 'poster_path',
              'tagline', 'production_companies', 'production_countries', 'spoken_languages', 'keywords'], axis=1)
df['org_title'] = df['title']
df['genres'] = df['genres'].fillna('unknown')
df = df.drop_duplicates()

# Encode genres
genre_l = df['genres'].apply(lambda x: x.split(','))
genre_l = pd.DataFrame(genre_l)
genre_l['genres'] = genre_l['genres'].apply(lambda x: [y.strip().lower().replace(' ', '') for y in x])
MLB = MultiLabelBinarizer()
genre_encoded = MLB.fit_transform(genre_l['genres'])
genre_encoded_df = pd.DataFrame(genre_encoded, columns=MLB.classes_)
genre_encoded_df = genre_encoded_df.reset_index()
mod_df = df.drop(['genres'], axis=1)
mod_df = mod_df.reset_index()
df = pd.concat([mod_df, genre_encoded_df], axis=1).drop('index', axis=1)

# Encode features
df['title'] = df['title'].apply(lambda x: x.strip().lower().replace(' ', ''))
df['original_language'] = df['original_language'].apply(lambda x: x.strip().lower().replace(' ', ''))
df.loc[~((df['original_language'] == 'en') | (df['original_language'] == 'fr') |
         (df['original_language'] == 'es') | (df['original_language'] == 'de') |
         (df['original_language'] == 'ja')), 'original_language'] = 'else'
OHE = OneHotEncoder(sparse_output=False)
df['adult'] = df['adult'].astype('str')
adult_enc = OHE.fit_transform(df[['adult']])
adult_enc_df = pd.DataFrame(adult_enc, columns=OHE.get_feature_names_out())
adult_enc_df = adult_enc_df.drop('adult_True', axis=1)
lang_enc = OHE.fit_transform(df[['original_language']])
lang_enc_df = pd.DataFrame(lang_enc, columns=OHE.get_feature_names_out())
mod_df = df.drop(['adult', 'original_language'], axis=1)
df = pd.concat([mod_df, adult_enc_df, lang_enc_df], axis=1)

# Normalize data
SC = StandardScaler()
df_norm = SC.fit_transform(df.drop(['title', 'org_title'], axis=1))
df_norm_df = pd.DataFrame(df_norm, columns=[x for x in df.columns if x not in ['title', 'org_title']])
df = pd.concat([df[['title', 'org_title']], df_norm_df], axis=1)

# Get recommendations
df = df.drop_duplicates(subset=['title'])
df = df.set_index(['title'])
df_fin = df.drop('org_title', axis=1)
movie_name = 'the dark knight'
movie_name = movie_name.strip().lower().replace(' ', '')
new_df = df_fin.loc[[movie_name]].values.reshape(1, -1)
df_other = df_fin.loc[df_fin.index != movie_name, :]
df_titles = df.loc[df.index != movie_name, 'org_title']
cosine_sim_matrix = cosine_similarity(new_df, df_other)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=[movie_name], columns=df_titles)
sorted_row = cosine_sim_df.loc[movie_name].sort_values(ascending=False)[0:20]

print(sorted_row.index)
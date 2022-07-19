import pandas as pd
import numpy as np
import Recommenders as Recommenders
import streamlit as st
import pickle 

song_df_1 = pd.read_csv('triplets_file.csv')
song_df_2 = pd.read_csv('song_data.csv')


song_df = pd.merge(song_df_1, song_df_2.drop_duplicates(
    ['song_id']), on='song_id', how='left')



song_df['song'] = song_df['title']+' - '+song_df['artist_name']
song_df = song_df.head(10000)

song_grouped = song_df.groupby(['song']).agg(
    {'listen_count': 'count'}).reset_index()



grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = (song_grouped['listen_count'] / grouped_sum) * 100
song_grouped.sort_values(['listen_count', 'song'], ascending=[0, 1])



song_list = pickle.load(open('songs.pkl','rb'))
song_l = song_list['song'].values

ir = Recommenders.item_similarity_recommender_py()
ir.create(song_df, 'user_id', 'song')



st.title('Music Recommender Engine')

selected_song_name = st.selectbox(
     'Search song to get Recommendations',
     (song_l))



if st.button('Recommend'):
     st.write(selected_song_name)


def recommend(selected_song_name):
 rs=ir.get_similar_items([selected_song_name])
 st.dataframe(rs)

recommend(selected_song_name)




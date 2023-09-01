import streamlit as st
import requests
from datetime import datetime
import os
import cv2
from kw_indexing import query_search, schema_formulation, create_index, precision_recall_f1_dcg_metric

st.set_page_config('SubtleSearch', layout='wide')

# user_query = input('Type your query here')

def format_time(time):
    '''
    Takes as input the timestamp and video path.
    Returns the frame count of the time passed
    '''
    # declaring the starting of the video to calculate time passed
    vid_start_time = datetime.strptime('00:00:00.000', '%H:%M:%S.%f')
    time_object = datetime.strptime(time, '%H:%M:%S.%f')

    # calculating time passed (to reach the given point of time)
    time_passed = time_object - vid_start_time
    time_in_seconds = time_passed.total_seconds()

    return int(time_in_seconds)

def video_files(title):
    '''
    Takes as input the title of a video.
    Returns the relative path to the video.
    '''
    vid_path = 'Assignment2/data/transcripts'
    for vid_file in os.listdir(vid_path):
        if title in vid_file:
            if vid_file.endswith('.mp4'):
                vid_file_path = vid_path + '/' + vid_file
                return vid_file_path
            
def organise_results(results):
    '''
    Takes the results as input.
    Arranges the videos such that docIDs are keys and the timestamps are values.
    Returns the results grouped with respect to videos.
    '''
    arranged_results = {}
    for result in results:
        if result[1] not in arranged_results:
            arranged_results[result[1]] = [result[2]]
        else:
            arranged_results[result[1]].append(result[2])

    return arranged_results
            

# search bar of the web application

with st.container():
    search_left, s_mid, s_right = st.columns((1, 2, 1))
    with search_left, s_right:
        st.empty()
    with s_mid:
        st.title('SubtleSearch')
        st.write('A Place to Scrutinize the Subtitles')

        user_query = st.text_input(label='Type your query here', placeholder='Type your query here', on_change=None, label_visibility='collapsed')
        st.write('###')

schema = schema_formulation()

with st.container():
    or_and_search, search_field, search_accuracy, displayed_results = st.columns((1, 2, 1, 0.7))
    with or_and_search:
        chosen = st.radio('Condition for results: ', ['AND', 'OR'], horizontal=True, index=0)
    with search_field:
        multifield = st.radio('Search in: ', ['Titles and Subtitles', 'Titles only', 'Subtitles only'], horizontal=True, index=2)
        # describe the actions which happen for each option in the query_search fn
    with search_accuracy:
        match_kind = st.radio('Find results which are ', ['Exact', 'Approximate'], horizontal=True, index=0)
    with displayed_results:
        topN = st.selectbox('Top results to be displayed: ', [10, 25, 50 ,100, 'All'])

st.write('##')

# using the user query to search the saved index and return top 10 results

try:
    st.spinner('Loading search results...')
    location, full_timestamp_caption_map = create_index(schema)

    if user_query: # runs only if there is an input from a user
        scoring_algo = st.text_input(label='Method to be used for ranking results: ', placeholder="'Frequency', 'Tfidf', or 'BM25f'")
        results, relevance_scores, k = query_search(schema, location, user_query, scoring_algo, topN, chosen, multifield, match_kind) # type: ignore
        print('Number of results: ', len(results))

        if k == None:
            num_results = len(results)
        else:
            num_results = k

        

        st.write('---')
        st.write('##')

        videos_folder = 'Assignment2/data/transcripts/'

        arranged_results = organise_results(results) # we get results grouped on the basis of videos; keys=docID, values=timestamp
        for doc in arranged_results: # doc is docID; timestamps are arranged in the order in which they appear in the results
            with st.container():
                # the order in 'result' is content, docID, start_time, title
                docID, videoSpace, material = st.columns((0.25, 1, 1.5))
                with docID:
                    st.write(doc)
                vid_title = '' 
                for result in results: # it matches the docID and gets the title of the doc
                    if result[1] == doc:
                        vid_title += result[3]
                        break
                with videoSpace:
                    video_path = videos_folder + vid_title + '.mp4'
                    vid_loc = st.video(video_path, start_time=format_time(arranged_results[doc][0])) #start_time takes the timestamp which appears first in the list for the 'doc' key
                with material:
                    st.header(vid_title) # title
                    for result in results:
                        if result[1] == doc:
                            st.write(f':blue[{result[2]}]: {result[0]}')
            st.write('---')
            st.write('##')

    
except IndexError:
    st.empty()






!pip install whoosh

from whoosh.fields import Schema, TEXT, NUMERIC, DATETIME, ID
from whoosh.index import create_in, open_dir
from whoosh.qparser import QueryParser
from whoosh import scoring
from whoosh.analysis import LowercaseFilter
import pandas as pd
import os.path
from math import log2

from srt_formatting import srt_filename_content, full_time_caption_list

 

def schema_formulation():
    # schema defines the information which should be indexed and stored (optional) into a file
    schema = Schema(title = TEXT(stored=True),
                    docID = NUMERIC(stored=True),
                    start_time = ID(stored=True), # it is not accepting DATETIME type. If required, start_time will have to processed so that it fits the format of DATETIME
                    content = TEXT(stored=True)
                    # video_thumbnail = STORED)
    )

    return schema

def create_index(schema):
    '''
    Creates an index based on the schema passed to it
    Returns: location where the index is created
    '''
    # defines the location where the indexed file will be saved
    if not os.path.exists('Assignment2/data'):
        os.mkdir('Assignment2/data')
    index_dir = 'Assignment2/data'

    ix = create_in(index_dir, schema)

    # opening and writing in the file
    writer = ix.writer()

    subtitles_path = 'Assignment2/data/raw_subtitles' # folder where srt files are stored
    fnames, srt_corpus = srt_filename_content(subtitles_path)
    full_timestamp_caption_map = full_time_caption_list(srt_corpus) # returns a list of dictionaries, where each dict contains timestamp-caption pairs of an srt file

    # formatting the filename (removing the extensions)
    docID_fname = {} # a dict mapping from docID to filenames
    for f in range(len(fnames)):
        docID_fname[f+1] = fnames[f] 

    for id in docID_fname:
        docID_fname[id] = docID_fname[id].replace('.srt', '')

    # print(full_timestamp_caption_map[0])

    # creating the index using all the required fields
    for doc in range(len(docID_fname)):
        for timestamp in full_timestamp_caption_map[doc]: # timestamp takes timestamp values for a specific srt file
            time = timestamp
            caption = full_timestamp_caption_map[doc][timestamp]
            writer.add_document(title = docID_fname[doc+1],
                                docID = doc+1,
                                start_time = time,
                                content = caption)

    # submitting and closing the file
    writer.commit()

    return index_dir, full_timestamp_caption_map


'''searching a query in the created index'''
def query_search(schema, index_dir, user_query, scoring_algo, N = 10, or_and_option='AND', multifield='Subtitles only', matching='Exact'):
    '''
    Searches for the user query in the created index
    Returns: results in a dataframe format
    '''
    ix = open_dir(index_dir)

    # creating a query parser
    from whoosh.qparser import MultifieldParser, OrGroup, AndGroup, FuzzyTermPlugin

    '''giving OR, AND conditions to search operations'''
    if or_and_option == 'OR':
        or_and_value = OrGroup
    else:
        or_and_value = AndGroup
            
    '''searching in multiple fields'''
    if multifield == 'Titles only':
        fieldname_arg = 'title'
    elif multifield == 'Titles and Subtitles':                                          # multifield == 'Titles and Subtitles'
        fieldname_arg = ['title', 'content']
    else:
        fieldname_arg = 'content'


    if len(fieldname_arg) == 2:
        query = MultifieldParser(fieldname_arg, schema, group=or_and_value)
    else:
        query = QueryParser(fieldname_arg, schema, group=or_and_value) # type: ignore #group=syntax.AndGroup


    '''search accuracy options'''
    if matching == 'Approximate':
        query.add_plugin(FuzzyTermPlugin())
        user_query = user_query+'~2/1' # '2' is the edit distance for fuzzy search, and '1' is the number of letters in the fixed prefix
    
    qterm = query.parse(user_query)

    # creating a searcher object   
    if scoring_algo == 'Frequency':
        ranker = scoring.Frequency
    elif scoring_algo == 'Tfidf':
        ranker = scoring.TF_IDF
    else:
        ranker = scoring.BM25F  # default is ranking using bm25f

    searcher = ix.searcher(weighting = ranker) # opening the Search object
    # specifying number of results to be displayed
    if N != 'All':
        topN = N
    else:
        topN = None
    results = searcher.search(qterm, terms = True, limit = topN)

    # getting the scores of the results
    
     # since None means having no limits on the number of results displayed, hence we pass len(results) to cal DCG
    result_scores = []
    for result in results:
        result_scores.append(result.score)
    
    
    q_results = [] # a list of lists, where each list is made up of the values of different fields in 'result'
    for result in results:
        # print('Caption: ', result['content'], 
        #     'Document: ', result['docID'],
        #     'Start Time: ', result['start_time'],
        #     'Title: ', result['title']
        #     )
        # we can see that we can easily access the results

        result_info = [] # stores values of different fields for a specific result
        for field in result:
            result_info.append(result[field])

        q_results.append(result_info)

    searcher.close()

    # print(q_results[0])


    return q_results, result_scores, topN


# function to evaluate the performance of different scoring methods
def precision_recall_f1_dcg_metric(metric, user_query, q_results, full_corpus, relevance_scores, k): # q_results, qterm are from query_search ; full_corpus is the full_timestamp_map 
   
    if metric != 'DCG':
        relevant = 0
        
        for result in q_results: # q_results is a list for lists, where each list is a result and contains values of diff fields
            if user_query in result[0]:
                relevant+=1
        # the above code iterates thru all the results and checks how many of them are relevant

        try:
            retrieved = 0
            for file in q_results: # counting the number of retrieved results
                retrieved+=1

            precision = relevant/retrieved 

    
            # to get the actual number of relevant results in the corpus
            actual_relevant = 0
            for file in full_corpus: # file is a dict
                for timestamp in file: # timestamp is the key
                    if user_query in file[timestamp]:
                        actual_relevant+=1

            recall = relevant/actual_relevant
            
            f1_value = 2*precision*recall/(precision+recall)


            if metric == 'Precision':
                print(f'Precision: {precision}')
            elif metric == 'Recall':
                print(f'Recall: {recall}')
            elif metric == 'F1 Score':
                print(f'F1 Score: {f1_value}')
        except ZeroDivisionError:
            pass

    else:
        # Discounted Cumulative Gain is a metric used to evaluate the ranking of results 

        # k gives the number of results to be used to calculate this metric
        # this can be kept equal to the number of results displayed
        dcg_score = relevance_scores[0]

        for i, _ in enumerate(relevance_scores, start=2):
            while i <= k:
                dcg_score += relevance_scores[i]/log2(i)
    
        print(f'DCG value: {dcg_score}')





if __name__ == '__main__':
    print('Hellooo')
    schema = schema_formulation()

    # creating an index and getting its location
    index_path = create_index(schema)

    # searches for user query 
    user_query = input('Type your query here: ')
    scoring_algo = input('Metric to be used for scoring: ')
    search_results = query_search(schema, index_path, user_query, scoring_algo)
    print(search_results)


import numpy as np


# tokenList is the list 'lemme_tokenlist' from preprocessing.py
# def demo_tfidf():
#     tokenList = ['local', 'also', 'give', 'picture', 'second', 'derivative', 'depend', 'ck', 'respect', 'solution', 'procedure', 'phase', 'close', 'yi', 'mean', 'close', 'set', 'significantly', 'two', 'take', 'object', 'compress', 'll', 'define', 'area', 'minimizes', 'give', 'side', 'close', 'allow', 'predefined', 'specific', 'several', 'square', 'close', 'ignore', 'best', 'distance', 'crucial', 'may', 'center', 'dataset', 'every', 'store', 'area', 'represent', 'partial', 'since', 'application', 'algorithm', 'tell', 'give', 'strong', 'blue', 'somewhere', 'slide', 'select', 'representative', 'hand', 'euclidean', 'randomly', 'next', 'original', 'pixel', 'non', 'consist', 'process', 'later', 'denote', 'numerical', 'y2', 'representative', 'problem', 'final', 'centroid', 'initialization', 'approach', 'average', 'would', 'minimize', 'way', 'initial', 'keep', 'optimal', 'segmentation', 'case', 'necessarily', 'effect', 'make', 'look', 'user', 'assign', 'step', 'take', 'assign', 'result', 'account', 'center', 'suggest', 'follow', 'quality', 'small', 'y5', 'much', 'input', 'repeat', 'clear', 'various', 'y3', 'go', 'many', 'globe', 'graphical', 'less', 'thing', 'formula', 'happen', 'n', 'call', 'previous', 'recompute', 'probably', 'within', 'find', 'logic', 'computation', 'want', 'compression', 'might', 'encode', 'convergence', 'affect', 'y6', 'whether', 'use', 'square', 'divide', 'vector', 'vector', 'optimize', 'yk', 'cluster', 'reassign', 'restriction', 's', 'mixed', 'actual', 'random', 'instance', 'specify', 'x1', 'distance', 'implement', 'sense', 'easy', 'choose', 'overcome', 'color', 'instance', 'minimize', 'assign', 'group', 'picture', 'id', 'objective', 'think', 'data', 'ci', 'time', 'determine', 'either', 'depend', 'function', 'get', 'belongs', 'easily', 'arbitrarily', 'move', 'choice', 'appropriate', 'xi', 'optimization', 'expect', 'good', 'anymore', 'measure', 'example', 'different', 'fix', 'stop', 'algorithm', 'certain', 'information', 'pick', 'quite', 'nearest', 'forced', 'step', 'homogeneous', 'see', 'issue', 'particular', 'image', 'sum', 'pick', 'find', 'way', 'pixel', 'proximity', 'distinguish', 'former', 'one', 'do', 'change', 'mm', '10', 'use', 'first', 'new', 'base', 'chooses', 'feature', 'design', 'location', 'ability', 'let', 'predict', 'require', 'xm', 'y1', 'follow', 'general', 'x2', 'seem', 'outlier', 'computable', 'y4', 'convex', 'iteratively', 'assignment', 'space', 'minimum', 'like', 'rgb', 'well', 'therefore', 'us', 'end', 'already', 'change', 'yellow', 'good', 'among', 'assume', 'outlier', 'applicability', 'decide', 'point', 'categorical', 'encodes', 'result', 'inner', 'global', 'sufficiently', 'cluster', 'turn', 'derivative', 'phase', 'three', 'thing', 'iteration', 'need', 'application', 'total', 'series', '255', 'c1', 'current', 'kmeans', 'unique', 'interpret', 'goal', 'time', 'show', 'computed', 'xn', 'leave', 'uh', 'value', 'identify', 'chosen', 'point', 'begin', 'compute', 'optimize', 'another', 'small', 'converge', 'zero', 'issue', 'object', 'mean', 'applicable', 'number', 'learn', 'um', 'belong', 'quick', 'video', 'important', 'say', 'basically', 'cluster', 'k', 'partition', 'c2', 'form', 'sum', 'move']
#     list1 = tokenList[:60]
#     list2 = tokenList[60:120]
#     list3 = tokenList[120:180]
#     list4 = tokenList[180:240]
#     list5 = tokenList[240:]

#     docs = [list1, list2, list3, list4, list5]
#     string_docs = [' '.join(list) for list in docs]

#     tfidf_instance = TfidfVectorizer()

#     docterm_matrix = tfidf_instance.fit_transform(string_docs)
#     # this returns a doc-term matrix where the number of terms is the number of unique tokens from each string (doc) in the collection of strings

#     feature_names = tfidf_instance.get_feature_names_out()  # feature_names are the same as the unique terms

#     for i, doc in enumerate(string_docs):
#         sorted_words = docterm_matrix[i].indices[np.argsort(docterm_matrix[i].data)]
#         # argsort sorts the data values and returns the indices (here, an integer for a unique term) of the data values
#         keywords = [feature_names[j] for j in sorted_words[:10]]
#         print(f'Keywords for document {i+1}: {keywords}')

'''
Use of TF-IDF to extract keywords is done above
'''

'''loads all the srt files content'''

import os 

def srt_filename_content(path):
    srt_corpus = [] # for storing srt file contents

    filenames = [] # stores names of srt files
    for file in os.listdir(path):
        if not file.endswith('.srt'): # if file name does not end with .srt, then it will be skipped
            continue
        else:
            filenames.append(file)
            with open(path+'/'+file) as srt_file:
                text = srt_file.read()
                srt_corpus.append(text)

    return filenames, srt_corpus


'''mapping docIDs and filenames'''
def assigning_docID(filenames): # this function is probably obsolete
    docID_filename_map = {}
    for id, doc in zip(range(len(filenames)), filenames):
        docID_filename_map[id+1] = doc

    return docID_filename_map


'''creating a timestamp-caption mapping'''
def timestamp_caption_mapping(srt_string):
    # this function takes srt content for a specific file from srt_corpus as input, and returns its time-caption mapping 
    
    raw_content = srt_string.split('\n') # splits the string into a list of substrings; newline character is removed 
    # print(raw_content)

    time_caption_map = {} # stores all timestamp-caption pairs for a srt file 
    line_index = 0
    while line_index < len(raw_content):
        if raw_content[line_index].isdigit():
            start_time = raw_content[line_index + 1].split(' --> ')[0] # extracts the start time from the two substrings
            start_time = start_time.replace(',', '.') # changing the format of timestamp to a standard one
            text = ''
            line_index += 2
            while line_index < len(raw_content) and raw_content[line_index] != '':
                text += raw_content[line_index] + ' '
                line_index += 1
            text = text.strip()
            time_caption_map[start_time] = text # creates a mapping of the time and associated caption
        else:
            line_index += 1

    return time_caption_map


def full_time_caption_list(srt_corpus):
    overall_timestamp_caption_list = [] # stores time-caption maps for srt files of all videos

    for file in srt_corpus:
        time_caption_dict = timestamp_caption_mapping(file)
        overall_timestamp_caption_list.append(time_caption_dict)

    return overall_timestamp_caption_list




if __name__ == '__main__':

    '''demonstrating use of TF-IDF to extract keywords'''
    # demo_tfidf()

    '''loads all the srt files content'''
    path = 'data/subtitles' # folder where srt files are stored
    filenames, srt_corpus = srt_filename_content(path)

    '''giving doc IDs to the files'''
    docID_filename_map = assigning_docID(filenames)

    '''creating a timestamp-caption mapping'''
    overall_timestamp_caption_list = full_time_caption_list(srt_corpus)

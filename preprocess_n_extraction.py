import srt 
import re
import nltk
# nltk.download('punkt')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# nltk.download('stopwords')
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet2021
# nltk.download('wordnet2021')
# nltk.download('averaged_perceptron_tagger')
from spellchecker import SpellChecker
import pandas as pd


def convertSRTtotext(srt_content):     # if srt file is available
    
    print(srt_content) # returns all the content in a single string
    print(type(srt_content))

    sub = srt.parse(srt_content) # creates a Subtitle object
    subtitleList = list(sub)

    # .content gives the text in each item of the list
    text = ' '.join(subtitleList[i].content for i in range(len(subtitleList)))
    text = text.replace('\n', ' ') # replaces newline characters with spaces to give a paragraph look to the text
    return text


'''tokenisation'''
def tokenising(text):
    tokens = word_tokenize(text)
    print(tokens)
    print(len(tokens))
    print('\n\nUnique set of tokens:\n')
    unique_tokens = set(tokens) # gives set of unique words
    print(unique_tokens)
    print(len(unique_tokens))

    return unique_tokens


'''removing stop-words'''
def removing_stopwords(tokenlist):
    refined_tokenlist = []  # this list contains the token list after removing the stopwords
    stop_words = stopwords.words('english')
    for term in tokenlist:
        if term not in stop_words:
            refined_tokenlist.append(term)

    print(refined_tokenlist)
    print(len(refined_tokenlist))
    return refined_tokenlist


'''noise removal and punctuations'''
def removing_noise_punctuations(tokenlist):
    for i in range(len(tokenlist)):
        tokenlist[i] = re.sub(r"[\W\s+]tokenlist","", tokenlist[i])

    refined_tokenlist = set(tokenlist)
    print('New unique list of tokens: \n', refined_tokenlist) # to get unique tokens
    print(len(refined_tokenlist))

    refined_tokenlist = list(filter(None, refined_tokenlist)) # to remove empty string, which is also considered as a unique token in the previous step
    print(len(refined_tokenlist))

    return refined_tokenlist


'''spelling corrections'''

def spellingCorrections(tokenlist):
    ''' this function considers many correct tokens as incorrect. 
    So will have to find a way to add those words to its dictionary, and if not possible, use some other spellchecker.'''
    sp_checker = SpellChecker()

    correct_tokenlist = []
    for token in tokenlist:
        # this is done to handle exceptions due to some words wrongly converted to 'None' while spelling correction
        correct_token = sp_checker.correction(token)
        if correct_token is None:
            correct_token = token
        correct_tokenlist.append(correct_token)
    
    print(correct_tokenlist)
    print(len(correct_tokenlist))
    
    # this shows the words which have been considered incorrect and hence corrected by the library
    changed = [f'{original}: {corrected}' for original, corrected in zip(refined_tokenlist, correct_tokenlist) if original != corrected]
    count = 0
    for i in changed:
        print(i)
        count+=1
    print(f'Number of tokens changed: {count}')

    # return list(set(correct_tokenlist))
    return correct_tokenlist


'''stemming'''
# spelling correction can be used after stemming as stemming is likely to produce incorrect token

def stemming(tokenlist):
    tokenStemmer = PorterStemmer()
    stem_tokenlist = []
    for token in tokenlist:
        stem = tokenStemmer.stem(token)
        stem_tokenlist.append(stem)

    # print(stem_tokenlist)
    print(len(set(stem_tokenlist)))

    return stem_tokenlist


'''POS tagging'''

def POS_tagging(tokenlist):
    tagged = nltk.pos_tag(tokenlist)
    # for token in tokenlist:
    #     taggedToken = nltk.pos_tag(token) # taggedToken is a tuple of 2 elements, token and its tag
    #     tagged.append(taggedToken)

    return tagged


'''lemmatisation'''

def POS_to_wordnet(tag):
    # this function converts POS tags to wordnet compatible form so that the tags can be used in lemmatisation
    if tag.startswith('JJS'):
        tag = wordnet2021.ADJ_SAT
    elif tag.startswith('V') or tag.startswith('M'):
        tag = wordnet2021.VERB
    elif tag.startswith('R'):
        tag = wordnet2021.ADV
    elif tag.startswith('N'):
        tag = wordnet2021.NOUN
    elif tag.startswith('J'):
        tag = wordnet2021.ADJ
    else:
        tag = None
    
    return tag

def lemmatising(taggedList):
    # this function iterates through the taggedList and converts each token to its lemma
    lemmatised_tokenlist = []
    for token, tag in taggedList:
        if POS_to_wordnet(tag) == None: # if mapped tag is None, then do not lemmatise the token as the default lemmatisation considers the token to be a noun
            lemmatised_tokenlist.append(token)
        else:
            lemmatised_tokenlist.append(wnl.lemmatize(token, str(POS_to_wordnet(tag))))

    return lemmatised_tokenlist



if __name__ == '__main__':

    '''parsing srt file contents'''
    stitle_path = 'Assignment2/data/subtitles/W6V1 k-means.srt'
    with open(stitle_path) as srt_file:
        content = srt_file.read()

    text = convertSRTtotext(content)
 

    '''
    Now, we perform the text preprocessing to analyse the words in the data.
    '''
    # doing segmentation may be beneficial for retrieving sentences

    '''lower-casing'''
    lower_text = text.lower()
    print(lower_text) # an issue is that some words are written as 'k means algorithm'. So 'k' will be treated as separate word, which is wrong

    '''tokenisation'''
    unique_tokens = tokenising(lower_text) # gives set of unique words

    '''stopword removal'''
    refined_tokenlist = removing_stopwords(unique_tokens)

    '''noise removal and punctuations'''
    cleaned_tokenlist = removing_noise_punctuations(refined_tokenlist)

    '''spelling corrections'''
    correct_tokenlist = spellingCorrections(cleaned_tokenlist)
    print(len(set(correct_tokenlist)))

    '''stemming'''
    # spelling correction can be used after stemming as stemming is likely to produce incorrect token
    stem_tokenlist = stemming(correct_tokenlist)

    '''POS tagging'''
    tagged_tokenlist = POS_tagging(stem_tokenlist) # POS_tagging returns a list of tuples 
    print(tagged_tokenlist)
    print(len(tagged_tokenlist))

    '''lemmatisation'''
    wnl = WordNetLemmatizer()
    lemme_tokenlist = lemmatising(tagged_tokenlist)
    print(lemme_tokenlist)
    print(len(set(lemme_tokenlist)))





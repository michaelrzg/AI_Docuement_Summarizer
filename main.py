# Extractive Text Summarization vs Abstractive Text Summarization
# Michael Rizig

#imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest

# import spacy's eng corpus
nlp = spacy.load('en_core_web_sm')

def extractive(text, n_sentences):
    """
    This function returns a summary of the input text with n_sentences via extractive summarization.
    Extracts the sentences with the highest sentence scores.

    @type text: string
    @param text: input text to be summarized
    @type n_sentences: int
    @param n_sentences: number of sentenes desired in summary
    @rtype: string
    @returns: string summart oof input text
    """
    # create our spacy object
    input = nlp(text)
    # generate our tokens from the document and remove stopwords
    tokens = [token.text.lower() for token in input 
            if not token.is_stop and 
            not token.is_punct and 
            token.text !='\n']

    # get list of frequency for each word
    word_freq = Counter(tokens)
    # get max frequency value
    max_freq = max(word_freq.values())
    # divide each word by the max frequency value
    for word in word_freq.keys():
        word_freq[word] = word_freq[word]/max_freq
    # create list of sentences from input
    sentences = [sent.text for sent in input.sents]
    # create list to hold scores for each senence
    sentence_scores = {}
    # for ach sentence get a sentence score by adding all word scores
    for sent in sentences:
        for word in sent.split():
            if word.lower() in word_freq.keys():
                if sent not in sentence_scores.keys():
                    sentence_scores[sent] = word_freq[word]
                else:
                    sentence_scores[sent] +=word_freq[word]
            
    # return the passed in number of sentences with hightest scores
    n = nlargest(n_sentences,sentence_scores,key=sentence_scores.get)
    return "".join(n)

#TODO: 
def abstractive(text):
    """
    This function returns a summary of the input text with n_sentences via Abstractive Summarization.

    @type text: string
    @param text: input text to be summarized
    
    """
    pass

input = "President-elect Donald Trump has named Pam Bondi, a former attorney general of Florida, as his next pick for U.S. attorney general. Bondi, who served as Florida's top prosecutor from 2011 to 2019, is a longtime ally of Trump's and was one of his lawyers during his first impeachment trial. She leads the legal arm of the America First Policy Institute, a think tank set up by former staffers from Trump's first presidency. In a statement announcing the selection, Trump said: \"Pam was a prosecutor for nearly 20 years, where she was very tough on Violent Criminals, and made the streets safe for Florida Families. Then, as Florida's first female Attorney General, she worked to stop the trafficking of deadly drugs, and reduce the tragedy of Fentanyl Overdose Deaths, which have destroyed many families across our Country. She did such an incredible job, that I asked her to serve on our Opioid and Drug Abuse Commission during my first Term — We saved many lives!\" Trump added: \"For too long, the partisan Department of Justice has been weaponized against me and other Republicans — Not anymore. Pam will refocus the DOJ to its intended purpose of fighting Crime, and Making America Safe Again.\""
print(extractive(input,5))

#TODO: Add abstractive text summarization usage here


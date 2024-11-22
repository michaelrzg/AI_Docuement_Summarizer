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
from transformers import BartForConditionalGeneration, BartTokenizer

def abstractive(text):
    """
    This function returns a summary of the input text with n_sentences via Abstractive Summarization.

    @type text: string
    @param text: input text to be summarized
    @rtype: string
    @returns: string summary of input text
    """
    # import bart
    bart = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    encoding = bart.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(encoding[0], skip_special_tokens=True)
    return summary

# MAIN

input = "KYIV, Ukraine — Russia fired an experimental intermediate-range ballistic missile at Ukraine overnight, Russian President Vladimir Putin said in a TV speech Thursday, warning that the Kremlin could use it against military installations of countries that have allowed Ukraine to use their missiles to strike inside Russia. Putin said the new missile, called \"Oreshnik,\" Russian for \"hazel,\" used a nonnuclear warhead. Ukraine's air force said a ballistic missile hit the central Ukrainian city of Dnipro, saying it was launched from the Astrakhan region in southeastern Russia, more than 770 miles away. Ukrainian officials said it and other rockets damaged an industrial facility, a rehabilitation center for people with disabilities and residential buildings. Three people were injured, according to regional authorities. \"This is an obvious and serious increase in the scale and brutality of this war,\" Ukrainian President Volodymyr Zelenskyy wrote on his Telegram messaging app. The attack came during a week of intense fighting in the nearly three years of war since Russia invaded Ukraine, and it followed U.S. authorization earlier this week for Ukraine to use its sophisticated weapons to strike targets deep inside Russia. Putin said Ukraine had carried out attacks in Russia this week using long-range U.S.-made Army Tactical Missile System (ATACMS) and British-French Storm Shadow missiles. He said Ukraine could not have carried out these attacks without NATO involvement. \"Our test use of Oreshnik in real conflict conditions is a response to the aggressive actions by NATO countries towards Russia,\" Putin said. He also warned: \"We believe that we have the right to use our weapons against military facilities of the countries that allow to use their weapons against our facilities.\""
print("Extractive Summary: ")
print(extractive(input,3))
print("Abstractive Summary with Bart: ")
print(abstractive(input))

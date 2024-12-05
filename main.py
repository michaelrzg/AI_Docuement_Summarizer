# Extractive Text Summarization vs Abstractive Text Summarization
# Michael Rizig

# install in order to run:
# pip install protobuf sentencepiece torch spacy rogue-score datasets bert-score
# python -m spacy download en

#imports
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from heapq import nlargest
from rouge_score import rouge_scorer
from datasets import load_dataset
from statistics import mean
from bert_score import BERTScorer
# import spacy's eng corpus
nlp = spacy.load('en_core_web_sm')

# import bert
from transformers import BertTokenizer, BertModel
# import bart
from transformers import BartForConditionalGeneration, BartTokenizer
# import t5
from transformers import T5ForConditionalGeneration, T5Tokenizer


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
    return "\n".join(n)

def abstractive_BART(text):
    """
    This function utilizes BART go return a summary of the input text with n_sentences via Abstractive Summarization.

    @type text: string
    @param text: input text to be summarized
    @rtype: string
    @returns: string summary of input text
    """
    # import our trained bartbart
    bart = BartForConditionalGeneration.from_pretrained("bart_cnn_dailymail_finetuned")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn",clean_up_tokenization_spaces=True)

    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    encoding = bart.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)

    summary = tokenizer.decode(encoding[0], skip_special_tokens=True)
    return "\n"+ summary


def abstractive_BERT_BART(text):
    """
    This function utilizies BERT and BART to return a summary of the input text with n_sentences via Abstractive Summarization.
    BERT used for encoding and BART used for decoding.

    @type text: string
    @param text: input text to be summarized
    @rtype: string
    @returns: string summary of input text
    """
    # BERT tokenizer and encoder
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # our trained BART decoder
    bart = BartForConditionalGeneration.from_pretrained("bart_cnn_dailymail_finetuned")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    encoding = bart.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(encoding[0], skip_special_tokens=True)
    return summary

def abstractive_T5(text):
    """
    This function utilizies T5 to return a summary of the input text with n_sentences via Abstractive Summarization.
    T5 used for encoding and decoding.

    @type text: string
    @param text: input text to be summarized
    @rtype: string
    @returns: string summary of input text
    """
    # BERT tokenizer and encoder
    tokenizer = T5Tokenizer.from_pretrained('t5-small',legacy=False)
    # BART decoder
    T5 = T5ForConditionalGeneration.from_pretrained("t5-small")
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)
    encoding = T5.generate(inputs, max_length=150, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(encoding[0], skip_special_tokens=True)
    return summary

def clean_summary(summary):
    """
    This function cleans the summary of unused tokens (from bert/bart passing)

    @type summary: string
    @param summary: summary to be cleaned
    @rtype: string
    @returns: cleaned summary
    """
    # Remove [unusedX] tokens
    return summary.replace("[unused", "").replace("]", "")

def generate_summaries(input):
    """
    This function generates and prints the extractive, finetuned BART, BERT-BART, and T5 Summaries for a given input article

    @type input: string
    @param input: input text to be summarized
    """
    print("\n\nExtractive Summary: \n")
    
    #generate extractive summary
    extractive_summary = extractive(input,4)
    print(extractive_summary)
    print("Scores for extractive summary:")

    # Generate ROGUE Scores
    extractive_scores = scorer.score(input, extractive_summary)
    for key in extractive_scores:
        print(f'{key}: {extractive_scores[key]}')



    # Abstractive Summary with BART
    print("\n\nAbstractive Summary with BART: ")
    BART_summary = abstractive_BART(input)
    print( BART_summary+ "\n")
    print("Scores for BART:")
    
    #generate ROGUE scores
    BART_scores = scorer.score(input, BART_summary)
    for key in BART_scores:
        print(f'{key}: {BART_scores[key]}')



    # Abstractive Summary with BERT Encodings and BART Decoding
    print("\n\nAbstractive Summary with BERT + BART: ")
    BERT_BART_summary = clean_summary(abstractive_BERT_BART(input))
    print(BERT_BART_summary + "\n")
    print("Scores for BERT+BART:")
    
    #generate ROGUE scores
    BERT_BART_scores = scorer.score(input, BERT_BART_summary)
    for key in BERT_BART_scores:
        print(f'{key}: {BERT_BART_scores[key]}')


    print("\n\nAbstractive Summary with BERT + T5: ")
    T5_summary = abstractive_T5(input)
    print( T5_summary+ "\n")
    print("Scores for T5:")

    #generate ROGUE scores
    T5_scores = scorer.score(input, T5_summary)
    for key in T5_scores:
        print(f'{key}: {T5_scores[key]}')
    

def generate_statistics(dataset,limit):
    """
    This function generates the performance metrics statistics for our
    extractive, finetuned bart, bert-bart, and T5 models summarizaton based on a dataset

    @type dataset: dataset
    @param dataset: dataset of summaries
    @type limit: integer
    @param limit: how many samples from the dataset to include
    @rtype: string
    @returns: string containing statistics
    """
    extractive_precision = []
    extractive_recall = []
    extractive_fmeasure=[]

    BART_precision = []
    BART_recall = []
    BART_fmeasure=[]

    BERT_BART_precision = []
    BERT_BART_recall = []
    BERT_BART_fmeasure=[]

    T5_precision = []
    T5_recall = []
    T5_fmeasure=[]
    
    e= []
    b = []
    bb=[]
    t = []
    
    count =0
    dataset = dataset.select(range(limit))
    for input in dataset:   
        #generate extractive summary
        extractive_summary = extractive(input["article"],4)
        e.append(extractive_summary)
        # Generate ROGUE Scores
        extractive_scores = scorer.score(input["article"], extractive_summary)
        
        # get precision for rogue1, rogue2, and rogueL
        extractive_precision.append((extractive_scores['rouge1'].precision,extractive_scores['rouge2'].precision,extractive_scores['rougeL'].precision))
        # get recall for rogue1, rogue2, and rogueL
        extractive_recall.append((extractive_scores['rouge1'].recall,extractive_scores['rouge2'].recall,extractive_scores['rougeL'].recall))
        # get fmeasure for rogue1, rogue2, and rogueL
        extractive_fmeasure.append((extractive_scores['rouge1'].fmeasure,extractive_scores['rouge2'].fmeasure,extractive_scores['rougeL'].fmeasure))

        # Abstractive Summary with BART
        BART_summary = abstractive_BART(input["article"])
        b.append(BART_summary)
        #generate ROGUE scores
        BART_scores = scorer.score(input["article"], BART_summary)
        
         # get precision for rogue1, rogue2, and rogueL
        BART_precision.append((BART_scores['rouge1'].precision,BART_scores['rouge2'].precision,BART_scores['rougeL'].precision))
        # get recall for rogue1, rogue2, and rogueL
        BART_recall.append((BART_scores['rouge1'].recall,BART_scores['rouge2'].recall,BART_scores['rougeL'].recall))
        # get fmeasure for rogue1, rogue2, and rogueL
        BART_fmeasure.append((BART_scores['rouge1'].fmeasure,BART_scores['rouge2'].fmeasure,BART_scores['rougeL'].fmeasure))

        # Abstractive Summary with BERT Encodings and BART Decoding
        BERT_BART_summary = clean_summary(abstractive_BERT_BART(input["article"]))
        bb.append(BERT_BART_summary)
        #generate ROGUE scores
        BERT_BART_scores = scorer.score(input["article"], BERT_BART_summary)
        
   

        # get precision for rogue1, rogue2, and rogueL
        BERT_BART_precision.append((BERT_BART_scores['rouge1'].precision,BERT_BART_scores['rouge2'].precision,BERT_BART_scores['rougeL'].precision))
        # get recall for rogue1, rogue2, and rogueL
        BERT_BART_recall.append((BERT_BART_scores['rouge1'].recall,BERT_BART_scores['rouge2'].recall,BERT_BART_scores['rougeL'].recall))
        # get fmeasure for rogue1, rogue2, and rogueL
        BERT_BART_fmeasure.append((BERT_BART_scores['rouge1'].fmeasure,BERT_BART_scores['rouge2'].fmeasure,BERT_BART_scores['rougeL'].fmeasure))

        T5_summary = abstractive_T5(input["article"])
        t.append(T5_summary)
        #generate ROGUE scores
        T5_scores = scorer.score(input["article"], T5_summary)
         # get precision for rogue1, rogue2, and rogueL
        T5_precision.append((T5_scores['rouge1'].precision,T5_scores['rouge2'].precision,T5_scores['rougeL'].precision))
        # get recall for rogue1, rogue2, and rogueL
        T5_recall.append((T5_scores['rouge1'].recall,T5_scores['rouge2'].recall,T5_scores['rougeL'].recall))
        # get fmeasure for rogue1, rogue2, and rogueL
        T5_fmeasure.append((T5_scores['rouge1'].fmeasure,T5_scores['rouge2'].fmeasure,T5_scores['rougeL'].fmeasure))
        
        count+=1
        print("Progress: " , (count/limit)*100 , "%")
    e_p,e_r,e_f = bert_scorer.score(e,[input["article"] for input in dataset])
    b_p,b_r, b_f = bert_scorer.score(b,[input["article"] for input in dataset])
    bb_p,bb_r,bb_f = bert_scorer.score(bb,[input["article"] for input in dataset])
    t_p,t_r,t_f = bert_scorer.score(t,[input["article"] for input in dataset])
    return f"""
ROGUE1 Precision: 
Extractive : {mean([x[0] for x in extractive_precision])}
BART : {mean([x[0] for x in BART_precision])}
BERT_BART : {mean([x[0] for x in BERT_BART_precision])}
T5 : {mean([x[0] for x in T5_precision])}

ROGUE2 Precision: 
Extractive : {mean([x[1] for x in extractive_precision])}
BART : {mean([x[1] for x in BART_precision])}
BERT_BART : {mean([x[1] for x in BERT_BART_precision])}
T5 : {mean([x[1] for x in T5_precision])}

ROGUEL Precision: 
Extractive :  {mean([x[2] for x in extractive_precision])}
BART :  {mean([x[2] for x in BART_precision])}
BERT_BART :  {mean([x[2] for x in BERT_BART_precision])}
T5 :  {mean([x[2] for x in T5_precision])}

ROGUE1 Recall: 
Extractive : {mean([x[0] for x in extractive_recall])}
BART : {mean([x[0] for x in BART_recall])}
BERT_BART : {mean([x[0] for x in BERT_BART_recall])}
T5 : {mean([x[0] for x in T5_recall])}

ROGUE2 Recall: 
Extractive : {mean([x[1] for x in extractive_recall])}
BART : {mean([x[1] for x in BART_recall])}
BERT_BART : {mean([x[1] for x in BERT_BART_recall])}
T5 : {mean([x[1] for x in T5_recall])}

ROGUEL Recall: 
Extractive :  {mean([x[2] for x in extractive_recall])}
BART :  {mean([x[2] for x in BART_recall])}
BERT_BART :  {mean([x[2] for x in BERT_BART_recall])}
T5 :  {mean([x[2] for x in T5_recall])}

ROGUE1 fMeasure: 
Extractive : {mean([x[0] for x in extractive_fmeasure])}
BART : {mean([x[0] for x in BART_fmeasure])}
BERT_BART : {mean([x[0] for x in BERT_BART_fmeasure])}
T5 : {mean([x[0] for x in T5_fmeasure])}

ROGUE2 fMeasure: 
Extractive : {mean([x[1] for x in extractive_fmeasure])}
BART : {mean([x[1] for x in BART_fmeasure])}
BERT_BART : {mean([x[1] for x in BERT_BART_fmeasure])}
T5 : {mean([x[1] for x in T5_fmeasure])}

ROGUEL fMeasure: 
Extractive :  {mean([x[2] for x in extractive_fmeasure])}
BART :  {mean([x[2] for x in BART_fmeasure])}
BERT_BART :  {mean([x[2] for x in BERT_BART_fmeasure])}
T5 :  {mean([x[2] for x in T5_fmeasure])}

BERTScore Precision:
Extractive :  {e_p.mean()}
BART :  {b_p.mean()}
BERT_BART :  {bb_p.mean()}
T5 :  {t_p.mean()}

BERTScore Recall:
Extractive :  {e_r.mean()}
BART :  {b_r.mean()}
BERT_BART :  {bb_r.mean()}
T5 :  {t_r.mean()}

BERTScore fMeasure:
Extractive :  {e_f.mean()}
BART :  {b_f.mean()}
BERT_BART :  {bb_f.mean()}
T5 :  {t_f.mean()}
"""


# MAIN
dataset = load_dataset("cnn_dailymail", "3.0.0")
#intilize scoring 
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
bert_scorer = BERTScorer(model_type='bert-base-uncased')
# input article from npr
input = "KYIV, Ukraine — Russia fired an experimental intermediate-range ballistic missile at Ukraine overnight, Russian President Vladimir Putin said in a TV speech Thursday, warning that the Kremlin could use it against military installations of countries that have allowed Ukraine to use their missiles to strike inside Russia. Putin said the new missile, called \"Oreshnik,\" Russian for \"hazel,\" used a nonnuclear warhead. Ukraine's air force said a ballistic missile hit the central Ukrainian city of Dnipro, saying it was launched from the Astrakhan region in southeastern Russia, more than 770 miles away. Ukrainian officials said it and other rockets damaged an industrial facility, a rehabilitation center for people with disabilities and residential buildings. Three people were injured, according to regional authorities. \"This is an obvious and serious increase in the scale and brutality of this war,\" Ukrainian President Volodymyr Zelenskyy wrote on his Telegram messaging app. The attack came during a week of intense fighting in the nearly three years of war since Russia invaded Ukraine, and it followed U.S. authorization earlier this week for Ukraine to use its sophisticated weapons to strike targets deep inside Russia. Putin said Ukraine had carried out attacks in Russia this week using long-range U.S.-made Army Tactical Missile System (ATACMS) and British-French Storm Shadow missiles. He said Ukraine could not have carried out these attacks without NATO involvement. \"Our test use of Oreshnik in real conflict conditions is a response to the aggressive actions by NATO countries towards Russia,\" Putin said. He also warned: \"We believe that we have the right to use our weapons against military facilities of the countries that allow to use their weapons against our facilities.\""

#generate_summaries(input)
print(generate_statistics(dataset["test"], 100))
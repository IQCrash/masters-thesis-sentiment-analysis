import json
import pandas as pd
import numpy as np
import re
import unidecode
import nltk

from bs4 import BeautifulSoup 
from autocorrect import Speller
from nltk.corpus import stopwords
from nltk import FreqDist
from sklearn.model_selection import train_test_split
from nltk.tokenize import word_tokenize
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Function to delete all the occurrences of newlines, tabs, and combinations like: \\n, \\.
def delete_newlines_and_tabs(text):
    
    # Replacing all the occurrences of \n,\\n,\t,\\ with a space.
    clean_text = text.replace('\\n', ' ').replace('\n', ' ').replace('\t',' ').replace('\\', ' ').replace('(', ' ').replace(')', ' ')
    return clean_text

# Function to remove all the occurrences of html tags from the text using BeautifulSoup
def strip_html_tags(text):
    
    # Initiating BeautifulSoup object soup.
    soup = BeautifulSoup(text, "html.parser")
    
    # Get all the text other than html tags.
    stripped_text = soup.get_text(separator=" ")
    return stripped_text

# Function to remove all the occurrences of links
def remove_links(text):
    
    # Use RegEx to identify links
    remove_link = re.sub(r"(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])", " ", text)
        
    return remove_link


# Function to remove repeated characters and punctuations
def reduce_character_repeatation(text):

    # Pattern matching for all case alphabets
    Pattern_alpha = re.compile(r"([A-Za-z])\1{1,}", re.DOTALL)
    
    # Limiting all the repeatation to two characters
    Formatted_text = Pattern_alpha.sub(r"\1\1", text) 
    
    # Pattern matching for all the punctuations that can occur
    Pattern_Punct = re.compile(r'([.,\#!$%^&*?;:{}=_`~()+-])')
    
    # Limiting punctuations in previously formatted string to only one.
    Combined_Formatted = Pattern_Punct.sub(r'\1', Formatted_text)
    
    # Replace spaces that occur more than two times with that of one occurrence
    Final_Formatted = re.sub(' {2,}',' ', Combined_Formatted)
    return Final_Formatted


# Function for removing special characters
def remove_special_characters(text):
    
    # The formatted text after removing not necessary punctuations.
    Formatted_Text = re.sub(r"[^a-zA-Z:$-,%.?!]+", ' ', text)
    Formatted_Text = re.sub(r"[\$\!\?\€\&]", ' ', Formatted_Text)
    
    # Remove strings starting with an apostrophe
    Formatted_Text = re.sub(r"'\w+\b", '', Formatted_Text)

    # Remove punctuation marks (., ,)
    Formatted_Text = re.sub(r'[^\w\s]', '', Formatted_Text)

    # Remove extra whitespaces
    Formatted_Text = re.sub(r"\s+", ' ', Formatted_Text).strip()

    return Formatted_Text


# Function to remove accented characters
def remove_accented_characters(text):

    text = unidecode.unidecode(text)
    return text

# Function to remove extra whitespaces
def remove_whitespace(text):
    
    pattern = re.compile(r'\s+') 
    Without_whitespace = re.sub(pattern, ' ', text)
    text = Without_whitespace.replace('?', ' ? ').replace(')', ') ')
    
    # Replace multiple occurences of dots 
    text = text.replace('..', '.')
    text = text.replace('..', '. ')
    return text



with open('contraction.json', 'r') as f:
    CONTRACTION_MAP = json.load(f)
    
# The code for expanding contraction words
def expand_contractions(text, contraction_mapping =  CONTRACTION_MAP):
    """expand shortened words to the actual form.
       e.g. don't to do not
        
       Example: 
       Input : ain't, aren't, can't, cause, can't've
       Output :  is not, are not, cannot, because, cannot have 
    
     """
    # Tokenizing text
    list_Of_tokens = text.split(' ')
    
    # Check whether Word is in list_Of_tokens or not.
    for Word in list_Of_tokens: 
        # Check whether found word is in dictionary "Contraction Map" or not
         if Word in CONTRACTION_MAP: 
                # If Word is present in both dictionary & list_Of_tokens, replace that word with the key value.
                list_Of_tokens = [item.replace(Word, CONTRACTION_MAP[Word]) for item in list_Of_tokens]
                
    # Converting list of tokens to String.
    String_Of_tokens = ' '.join(str(e) for e in list_Of_tokens) 
    return String_Of_tokens

# The code for removing stopwords
stoplist = stopwords.words('english') 
stoplist = set(stoplist)

def remove_stopwords(text):
  
    # Tokenize the text into individual words
    words = word_tokenize(text)
    
    no_StopWords = [word for word in words if word.lower() not in stoplist ]
    
    # Join the words back into a single string
    words_string = ' '.join(no_StopWords)    
    return words_string

def remove_dates_and_numbers(text):
    # Define the regular expression pattern
    pattern = r'\b\d+(?:st|nd|rd|th)\b'

    # Remove dates and numbers with ordinal indicators
    modified_text = re.sub(pattern, '', text)
    
    return modified_text

# Function to remove dots and commas    
def remove_dots_commas(text):
    clean_text = text.replace(' , ', ', ').replace(' . ', '. ').replace('  ', ' ')
    # .replace('. com', '.com')
    # .replace('(',' ').replace(')',' ')
    return clean_text


# Function to evaluate accuracy, precision, recall and f1-score
def calculate_results(y_true, y_pred):
    # Calculate model accuracy
    model_accuracy = accuracy_score(y_true, y_pred) * 100
    
    # Calculate model precision, recall and f1 score
    model_precision, model_recall, model_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")
    model_results = {"Accuracy": model_accuracy,
                     "Precision": model_precision,
                     "Recall": model_recall,
                     "f1": model_f1}
    return model_results

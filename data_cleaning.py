#!/usr/bin/env python
# coding: utf-8

# In[40]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the TSV files
train_file_path = r'C:\Users\prits\Downloads\Data\ghc_train.tsv'
test_file_path = r'C:\Users\prits\Downloads\Data\ghc_test.tsv'

train_df = pd.read_csv(train_file_path, sep='\t')
test_df = pd.read_csv(test_file_path, sep='\t')

# Display the first few rows of the data
print("Train DataFrame Head:")
display(train_df.head())
print("Test DataFrame Head:")
display(test_df.head())

# Inspect the data
print("Train DataFrame Info:")
train_df.info()
print("Train DataFrame Description:")
display(train_df.describe())

print("Test DataFrame Info:")
test_df.info()
print("Test DataFrame Description:")
display(test_df.describe())

# Handle missing values
train_df.dropna(inplace=True)
test_df.dropna(inplace=True)
# Or fill missing values (example: fill with mean for numerical columns)
# train_df.fillna(train_df.mean(), inplace=True)
# test_df.fillna(test_df.mean(), inplace=True)


# # REMOVING URL

# In[41]:



def clean_data(dataframe):
#replace URL of a text
    test_df['text'] = test_df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')

clean_data(test_df)
print(test_df['text']);


# In[42]:


def clean_data(dataframe):
#replace URL of a text
    train_df['text'] = train_df['text'].str.replace('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ')

clean_data(train_df)
print(train_df['text']);


# # Abbreviation Treatment 

# In[43]:


import re
import pandas as pd

# Dictionary of abbreviations and their full forms
abbreviations = {
    'u': 'you',
    'r': 'are',
    'omg': 'oh my god',
    'lol': 'laugh out loud',
    'brb': 'be right back',
    'idk': 'I don’t know',
    'tbh': 'to be honest',
    'btw': 'by the way',
    'afaik': 'as far as I know',
    'bbl': 'be back later',
    'bfn': 'bye for now',
    'bff': 'best friends forever',
    'cya': 'see you',
    'ftw': 'for the win',
    'fyi': 'for your information',
    'gtg': 'got to go',
    'imo': 'in my opinion',
    'imho': 'in my humble opinion',
    'irl': 'in real life',
    'jk': 'just kidding',
    'lmao': 'laughing my ass off',
    'lmk': 'let me know',
    'nvm': 'never mind',
    'omw': 'on my way',
    'rofl': 'rolling on the floor laughing',
    'smh': 'shaking my head',
    'tba': 'to be announced',
    'tbd': 'to be decided',
    'ttyl': 'talk to you later',
    'txt': 'text',
    'w/e': 'whatever',
    'w/o': 'without',
    'w8': 'wait',
    'yolo': 'you only live once',
    'plz': 'please',
    'thx': 'thanks',
    'xoxo': 'hugs and kisses',
    'NATO':'North Atlantic Treaty Organization.'
    # Add more abbreviations as needed
}

def chat_word(text):
    new_text=[]
    for word in text.split():
        if word.upper() in abbreviations :
            new_text.append(abbreviations [word.upper()])
        else:
            new_text.append(word)
            
    return " ".join(new_text)

test_df['text']=test_df['text'].apply(chat_word)
test_df.head()


# In[44]:


train_df['text']=train_df['text'].apply(chat_word)
train_df.head()


# In[45]:



specific_text = test_df.loc[19, 'text']
print(specific_text )


# # removing unwanted tags such as @icareviews

# In[46]:


import pandas as pd
import re

def remove_pattern(text):
    # Define the pattern to match
    pattern = r'@\w+\b'  # Matches "@" followed by one or more word characters (\w+)

    # Use the sub() function from the re module to replace matched patterns with an empty string
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text



test_df['text'] = test_df['text'].apply(remove_pattern)
test_df.head()


# In[47]:


train_df['text'] = train_df['text'].apply(remove_pattern)
print(train_df)


# In[48]:


specific_text = test_df.loc[72, 'text']
print(specific_text) 


# # labelling the text type

# In[49]:


import pandas as pd

def label_content(text):
    # Define lists of inappropriate words and phrases
    inappropriate_keywords = [
        'fuck', 'shit', 'nigger', 'slut', 'bitch', 'pussy', 'dick', 'asshole', 'faggot', 
        'cunt', 'bastard', 'whore', 'rape', 'pedophile', 'nazi'
        # Add more keywords as needed
    ]
    
    # Convert text to lowercase for case-insensitive matching
    text_lower = text.lower()
    
    # Check if any inappropriate keywords are in the text
    if any(keyword in text_lower for keyword in inappropriate_keywords):
        return 'Inappropriate'
    else:
        return 'Appropriate'

# Apply the label_content function to the 'text' column
test_df['label'] = test_df['text'].apply(label_content)

# Display the DataFrame
print(test_df.head())


    


# In[50]:


train_df['label'] = train_df['text'].apply(label_content)

# Display the DataFrame
print(train_df.head())


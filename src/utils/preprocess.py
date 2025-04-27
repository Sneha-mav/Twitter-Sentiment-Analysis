import re
import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import os
from tqdm import tqdm
import argparse

nltk.download('stopwords')

class TextPreprocessor:

    def __init__(self):
        self.stop_words=set(stopwords.words('english'))
        self.stemmer=PorterStemmer()
        self.url_pattern=re.compile(r'https?://\S+|www\.\S+')
        self.non_alpha_pattern=re.compile(r'[^a-zA-Z\s]')
        self.user_mention_pattern = re.compile(r'@\w+')

    def clean_text(self, text):
        text=str(text).lower()
        text=self.url_pattern.sub(' ', text)
        text=self.user_mention_pattern.sub(' ', text)
        text=self.non_alpha_pattern.sub(' ', text)
        words=text.split()
        stemmed_words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        return ' '.join(stemmed_words)

def load_dataset(data_path):

    df=pd.read_csv(data_path,encoding='ISO-8859-1')
    df.columns=['target','id','date','flag','user','text']
    df['target']=df['target'].replace(4,1)
    return df[['target','text']]

def dataset_preprocessing():

    preprocessor=TextPreprocessor()
    df['cleaned_text']=df['text'].progress_apply(preprocessor.clean_text)
    return df[['target','cleaned_text']]

def pipeline_creation(input_path):

    print(f'Loading dataset from {input_path}')
    df=load_dataset(input_path)
    print("Dataset loaded successfully!")

    tqdm.pandas()
    print('Performing preprocessing on the dataset...')
    df=dataset_preprocessing(df)
    print('Preprocessing completed successfully!')
    return df

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=str,required=True,help="Path to the input dataset file(raw csv)")
    args=parser.parse_args()

    pipeline_creation(args.input)
##################################################
# Class for creating a dictionary of words
# Source: https://www.geeksforgeeks.org/word-embeddings-in-nlp/
# Source: Assignment 3 Machine_Translation.ipynb
##################################################

import pandas as pd
import re
import torch
from typing import List
from torch.utils.data import Dataset


# source for Vocabulary class:
# Assignment 3 Machine_Translation.ipynb
class Vocabulary:
    def __init__(self, word_lookup_csv, def_idx=0, def_word='<unk>'):
        self.word_to_idx = pd.read_csv(word_lookup_csv)
        self.idx_to_word = self.word_to_idx.copy()
        self.default_idx = def_idx
        self.default_word = def_word

        # source for dataframe help:
        # https://saturncloud.io/blog/how-to-search-pandas-data-frame-by-index-value-and-value-in-any-column/
        self.word_to_idx.set_index('word', inplace=True)
        self.idx_to_word.set_index('idx', inplace=True)

    def __len__(self):
        return len(self.word_to_idx)

    def to_word(self, idx: int) -> str:
        if idx < 0:
            return self.default_word
        try:
            word = self.idx_to_word.loc[idx]['word']
        except KeyError:
            word = self.default_word
        return word

    def to_idx(self, word: str) -> int:
        if word is None:
            return self.default_idx
        try:
            idx = self.word_to_idx.loc[word.lower()]['idx']
        except KeyError:
            idx = self.default_idx

        return idx


################################################
# Custom Dataset Class
# Source: https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Source: https://www.codecademy.com/article/how-to-use-pytorch-dataloader-custom-datasets-transformations-and-efficient-techniques
#################################################
class BillDataset(Dataset):
    def __init__(self, df):
        self.data = df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        print(row)
        text = torch.tensor(row['text'])
        summary = torch.tensor(row['summary'])
        return text, summary


#########################################
# Data Encoder Class                    #
#########################################
class DataEncoder:
    def __init__(self, vocab, start_idx, stop_idx):
        self.vocab = vocab
        self.start_token = vocab.to_word(start_idx)
        self.stop_token = vocab.to_word(stop_idx)

    def encode_data_to_csv(self, data_csv_path: str, encoded_csv_path: str):
        orig_bill_df = pd.read_csv(data_csv_path)

        # create lists of indices
        text_idx = []
        summary_idx = []
        for i in range(len(orig_bill_df)):
            text_idx.append(self.text_to_idx_str(orig_bill_df['text'][i]))
            summary_idx.append(self.text_to_idx_str(orig_bill_df['summary'][i]))

        # use lists to create dict, then dataframe, then dataset
        bill_dict = {'text': text_idx, 'summary': summary_idx}
        bill_df = pd.DataFrame(bill_dict)
        bill_df.to_csv(encoded_csv_path)

    def text_to_idx_str(self, text: str) -> str:
        tokens = tokenize_str(text)

        # add start/stop tokens
        tokens.insert(0, self.start_token)
        tokens.append(self.stop_token)

        # create idx tensor
        num_tokens = len(tokens)
        idx_str = ""

        for i in range(num_tokens):
            idx_str += f"{self.vocab.to_idx(tokens[i])} "
        return idx_str


###############################################
# Static Data Processing Methods              #
###############################################
def get_encoded_bill_data(encoded_data_path, def_idx):
    encoded_str_df = pd.read_csv(encoded_data_path)
    encoded_str_texts = encoded_str_df['text']
    encoded_str_summaries = encoded_str_df['summary']

    num_samples = len(encoded_str_texts)

    texts = []
    summaries = []

    for i in range(num_samples):
        texts.append(idx_str_to_int_list(encoded_str_texts[i], def_idx))
        summaries.append(idx_str_to_int_list(encoded_str_summaries[i], def_idx))

    bill_dict = {'text': texts, 'summary': summaries}
    bill_df = pd.DataFrame(bill_dict)
    return BillDataset(bill_df)


def create_word_lookup_csv(csv_all_data_path: str, word_lookup_csv: str, special_tokens=None):
    if special_tokens is None:
        special_tokens = ['<unk>', '<start>', '<stop>']
    df = pd.read_csv(csv_all_data_path)
    texts, summaries = df['text'], df['summary']
    word_set = set()
    if len(texts) != len(summaries):
        print("error with data lengths")
    for i in range(len(texts)):
        word_set.update(tokenize_str(texts[i].lower()))
        word_set.update(tokenize_str(summaries[i].lower()))
    word_dict = create_word_dict(word_set, special_tokens)
    word_df = pd.DataFrame(word_dict)
    word_df.to_csv(word_lookup_csv, index=False)


def create_clean_csv(orig_csv_path, new_csv_path):
    df = pd.read_csv(orig_csv_path)

    keys = ['text', 'summary']
    values = []

    text = df['diff_text']
    summary = df['summary']

    clean_texts = []
    clean_summaries = []
    for i in range(len(text)):
        clean_texts.append(remove_angle_brackets(text[i], i))
        clean_summaries.append(remove_angle_brackets(summary[i], i))

    values.append(clean_texts)
    values.append(clean_summaries)

    # source for dict
    # https://www.geeksforgeeks.org/how-to-create-a-dictionary-in-python/
    clean_dict = {k: v for k, v in zip(keys, values)}

    # source for df
    # https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
    clean_df = pd.DataFrame(clean_dict)
    clean_df.to_csv(new_csv_path)


###############################################
# Helper Data Processing Methods              #
###############################################
# copied from assignment3:
def tokenize_str(my_str: str) -> List[str]:
    tokens = re.findall(r'\b\w+\b', my_str)
    return tokens


def create_word_dict(word_set: set, special_tokens: list) -> dict:
    words = special_tokens
    idxs = [i for i in range(len(words))]
    idx = len(words)
    for word in word_set:
        words.append(word)
        idxs.append(idx)
        idx += 1
    word_dict = {'idx': idxs, 'word': words}
    return word_dict


def idx_str_to_int_list(idx_str: str, def_idx: int) -> list:
    idx_list = idx_str.strip().split()
    int_list = []
    for i in range(len(idx_list)):
        idx = idx_list[i]
        if idx.isdigit():
            int_list.append(int(idx))
        else:
            int_list.append(def_idx)
    return int_list


def remove_angle_brackets(my_str, idx):
    my_str = my_str.replace("</line>", " ")
    counter = 0
    while True:
        if my_str.find("<") == -1 or my_str.find(">") == -1:
            return my_str
        if counter > 100000:
            print("Possibly infinite loop at data idx:", idx)
            return my_str

        b1 = my_str.rfind("<")
        b2 = my_str.rfind(">")

        if b2 < b1:
            print("Found issue with missing '>' at data idx", idx)
            my_str = my_str[0:b1] + my_str[b1 + 1:-1]
            print("Removed extra '<'")

        else:
            text_between = my_str[b1:b2 + 1]
            my_str = my_str.replace(text_between, "")

        counter += 1

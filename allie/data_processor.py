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
from collections import Counter


# source for Vocabulary class:
# Assignment 3 Machine_Translation.ipynb
class Vocabulary:
    def __init__(self, word_lookup_csv, def_idx=1, def_word='<unk>'):
        # source for word embedding help:
        # https://www.geeksforgeeks.org/word-embeddings-in-nlp/
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
        return row['text'], row['summary']


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
        
        # when encoding data, always include all data and numbers
        for i in range(len(orig_bill_df)):
            text_idx.append(self.text_to_idx_str(orig_bill_df['text'][i], False))
            summary_idx.append(self.text_to_idx_str(orig_bill_df['summary'][i], True))

        # use lists to create dict, then dataframe, then dataset
        bill_dict = {'text': text_idx, 'summary': summary_idx}
        bill_df = pd.DataFrame(bill_dict)
        bill_df.to_csv(encoded_csv_path)

    def text_to_idx_str(self, text: str, add_extra_start: bool) -> str:
        tokens = tokenize_str(text)

        # add start/stop tokens
        tokens.insert(0, self.start_token)
        if add_extra_start:
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
def get_encoded_bill_data(encoded_data_path, train_percent, max_len_text,
                          max_len_summary, pad_idx, def_idx, stop_idx, len_vocab):
    encoded_str_df = pd.read_csv(encoded_data_path)
    encoded_str_texts = encoded_str_df['text']
    encoded_str_summaries = encoded_str_df['summary']

    num_samples = len(encoded_str_texts)
    num_train = int(num_samples * train_percent)

    train_texts = []
    train_summaries = []
    valid_texts = []
    valid_summaries = []

    truncated_texts = 0
    truncated_sums = 0

    for i in range(num_samples):
        text_tensor, is_text_complete = idx_str_to_tensor(encoded_str_texts[i], max_len_text, pad_idx, def_idx, stop_idx, len_vocab)
        sum_tensor, is_sum_complete = idx_str_to_tensor(encoded_str_summaries[i], max_len_summary, pad_idx, def_idx, stop_idx, len_vocab)
        if not is_text_complete:
            truncated_texts += 1
        if not is_sum_complete:
            truncated_sums += 1
        if i < num_train:
            train_texts.append(text_tensor)
            train_summaries.append(sum_tensor)
        else:
            valid_texts.append(text_tensor)
            valid_summaries.append(sum_tensor)

    train_bill_dict = {'text': train_texts, 'summary': train_summaries}
    train_bill_df = pd.DataFrame(train_bill_dict)
    valid_bill_dict = {'text': valid_texts, 'summary': valid_summaries}
    valid_bill_df = pd.DataFrame(valid_bill_dict)

    print(f"Total Truncated Texts: {truncated_texts} -> {truncated_texts / num_samples}")
    ex_0 = train_bill_df['text'][0]
    print(f"Total Truncated Summaries: {truncated_sums} -> {truncated_sums / num_samples}")
    
    return BillDataset(train_bill_df), BillDataset(valid_bill_df)


# help with counter()
# https://docs.python.org/3/library/collections.html#collections.Counter
def create_word_lookup_csv(csv_all_data_path: str, word_lookup_csv: str, special_tokens=None, n_common=None):
    if special_tokens is None:
        special_tokens = ['<pad>', '<unk>', '<start>', '<stop>']
    df = pd.read_csv(csv_all_data_path)
    texts, summaries = df['text'], df['summary']
    word_counter = Counter()
    if len(texts) != len(summaries):
        print("error with data lengths")
    for i in range(len(texts)):
        word_counter.update(tokenize_str(texts[i].lower()))
        word_counter.update(tokenize_str(summaries[i].lower()))
    word_list = special_tokens
    idx_list = [i for i in range(len(word_list))]
    idx = len(word_list)
    most_common = word_counter.most_common()
    for word_tuple in most_common:
        word_list.append(word_tuple[0])
        idx_list.append(idx)
        idx += 1
    word_dict = {'idx': idx_list, 'word': word_list}
    word_df = pd.DataFrame(word_dict)
    if n_common != None:
        # source for help with df:
        # https://datascienceparichay.com/article/pandas-select-first-n-rows-dataframe/
        word_df = word_df.head(n_common)
    
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

    # source for dataframe
    # https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
    clean_df = pd.DataFrame(clean_dict)
    clean_df.to_csv(new_csv_path)


###############################################
# Helper Data Processing Methods              #
###############################################
# tokenize_str copied from assignment3:
def tokenize_str(my_str: str) -> List[str]:
    # regex help:
    # https://regex101.com/
    tokens = re.findall(r'\b[a-zA-Z0-9\']+\b', my_str)
    return tokens


def idx_str_to_tensor(idx_str: str, max_len: int, pad_idx: int, def_idx: int, stop_idx: int, len_vocab: int) -> torch.Tensor:
    idx_list = idx_str.strip().split()
    idx_tensor = torch.zeros(max_len)
    if pad_idx != 0:
        # tensor is filled with 0s, so expect that def_idx = 0
        print("error, pad idx must be 0")
    for i in range(len(idx_list)):
        if i == (max_len - 1):
            #print(f"Truncated data. Max Length={max_len}. Actual Length={len(idx_list)}")
            idx_tensor[i] = stop_idx
            return idx_tensor, False
        idx = idx_list[i]
        if idx.isdigit():
            idx = int(idx)
            if idx < len_vocab:
                idx_tensor[i] = idx
            else:
                idx_tensor[i] = def_idx
    return idx_tensor, True


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


#####################################
# Sources
# Assignment 3 Machine_Translation.ipynb
# https://www.geeksforgeeks.org/word-embeddings-in-nlp/
# https://docs.python.org/3/library/collections.html#collections.Counter
# https://datascienceparichay.com/article/pandas-select-first-n-rows-dataframe/
# https://www.geeksforgeeks.org/how-to-create-a-dictionary-in-python/
# https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html
#######################################

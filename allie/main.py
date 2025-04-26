from data_processor import DataEncoder, Vocabulary, create_word_lookup_csv, create_clean_csv, get_encoded_bill_data
from torch.utils.data import DataLoader
import yaml


# GLOBALS
config_path = "config/config.yaml"

# these paths are for the original data
data_csv = "data/all_bill_data.csv"
train_csv = "data/train_bill_data.csv"

# these paths will be used to generate cleaner data
clean_data_csv = "data/data_clean.csv"
clean_train_csv = "data/train_data_clean.csv"
word_lookup_csv = "data/word_lookup.csv"
encoded_train_csv = "data/encoded_train.csv"

# special tokens!
special_tokens = ['<unk>', '<start>', '<stop>']


def clean_all_data():
    create_clean_csv(data_csv, clean_data_csv)
    create_clean_csv(data_csv, clean_train_csv)


def create_word_lookup():
    create_word_lookup_csv(clean_data_csv, word_lookup_csv, special_tokens)


def create_encoded_data():
    vocab = Vocabulary(word_lookup_csv)
    d = DataEncoder(vocab, 1, 2)
    d.encode_data_to_csv(clean_train_csv, encoded_train_csv)


def get_config(config_file):
    # get configurations
    # source: assignment 4 code
    with open(config_file, "r") as file:
        config_dict = yaml.safe_load(file)
    return config_dict


if __name__ == '__main__':
    # only need to create the csv files once,
    # then comment these lines out
    #clean_all_data()
    #create_word_lookup()
    #create_encoded_data()

    # get vocab
    vocab = Vocabulary(word_lookup_csv)

    # get configs
    config = get_config(config_path)
    device = 'cpu'
    max_len_text, max_len_summary = config['max_len_text'], config['max_len_summary']

    # get data
    bill_data = get_encoded_bill_data(encoded_train_csv, 0)
    dataloader = DataLoader(bill_data, batch_size=config['batch_size'], shuffle=True)




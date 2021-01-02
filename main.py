import argparse
import os
import glob
import sys
from tqdm import tqdm
from preprocess import *

parser = argparse.ArgumentParser(description="Basic Kor Chatbot")

parser.add_argument('-path',type=str, default='data_in/ChatBotData.csv', help='Training Data Path')
parser.add_argument('-vocab_path',type=str, default='data_in/vocabualry.csv', help='Vocabulary Data Path')
args = parser.parse_args()

def preprocess_data(PATH, VOCAB_PATH):
    inputs, outputs = load_data(PATH)
    char2idx, idx2char, vocab_size = load_vocabulary(PATH, VOCAB_PATH)
    index_inputs, input_seq_len = enc_processing(inputs, char2idx)
    index_outputs, output_seq_len = dec_output_processing(outputs, char2idx)
    index_targets = dec_target_processing(outputs, char2idx)

    data_configs = {}
    data_configs['char2idx'] = char2idx
    data_configs['idx2char'] = idx2char
    data_configs['vocab_size'] = vocab_size
    data_configs['pad_symbol'] = PAD
    data_configs['std_symbol'] = STD
    data_configs['end_symbol'] = END
    data_configs['unk_symbol'] = UNK

    DATA_IN_PATH = './data_in/
    TRAIN_INPUTS = 'train_inputs.npy'
    TRAIN_OUTPUTS = 'train_outputs.npy'
    TRAIN_TARGETS = 'traiN_targets.npy'
    DATA_CONFIGS = 'data_configs.json'

    np.save(open(DATA_IN_PATH+TRAIN_INPUTS,'wb'),index_inputs)
    np.save(open(DATA_IN_PATH+TRAIN_OUTPUTS,'wb'),index_outputs)
    np.save(open(DATA_IN_PATH+TRAIN_TARGETS,'wb'),index_targets)

    json.dump(data_configs, open(DATA_IN_PATH+DATA_CONFIGS,'w'))

if __name__ == "__main__":
    preprocess_data(args.path, args.vocab_path)    
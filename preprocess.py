import os
import re
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from konlpy.tag import Okt

FILTERS="([~,.!?\"':;)(])"
PAD = "<PAD>"
STD = "<SOS>"
END = "<END>"
UNK = "<UNK>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD, STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

MAX_SEQUENCE = 25

def load_data(path):
    data_df = pd.read_csv(path, header=0)
    question, answer = list(data_df['Q']), list(data_df['A'])

    return question, answer

def data_tokenizer(data):
    words = []
    for sentence in data:
        sentence = re.sub(CHANGE_FILTER,"",sentence)
        for word in sentence.split():
            words.append(word)

    return [word for word in words if word]

def prepro_like_morphlized(data):
    morph_analyzer = Okt()
    result_data = list()
    for seq in data:
        morphlized_seq = morph_analyzer.morphs(seq.replace(' ',''))
        result_data.append(morphlized_seq)

    return result_data

def load_vocabulary(path, vocab_path):
    vocabulary_list = list()
    if not os.path.exists(vocab_path):
        if (os.path.exists(path)):
            data_df = pd.read_csv(path, encoding='utf-8')
            question, answer = list(data_df['Q']), list(data_df['A'])
            data = []
            data.extend(question)
            data.extend(answer)

            words = data_tokenizer(data)
            words = list(set(words))
            words[:0] = MARKER # HEAD APPEND

        with open(vocab_path, 'w',encoding='utf-8') as vocabulary_file:
            for word in words:
                vocabulary_file.write(word+'\n')

    with open(vocab_path,'r',encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())
        
    word2idx, idx2word = make_vocabulary(vocabulary_list)

    return word2idx, idx2word, len(word2idx)

def make_vocabulary(vocabulary_list):
    word2idx = {w:i for i,w in enumerate(vocabulary_list)}
    idx2word = {i:w for i,w in enumerate(vocabulary_list)}

    return word2idx, idx2word

# word2idx, idx2word, vocab_size = load_vocabulary(PATH, VOCAB_PATH)

def enc_processing(value, dictionary):
    sequences_input_index = []
    sequences_length = []

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = []

        for word in sequence.split():
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
            else:
                sequence_index.extend([dictionary[UNK]])

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        
        sequences_length.append(len(sequences_length))

        # PADDING POST 
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_input_index.append(sequence_index)

    return np.asarray(sequences_input_index), sequences_length

def dec_output_processing(value, dictionary):
    sequences_output_index = []
    sequences_length = []

    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[STD]] + [dictionary[word] for word in sequence.split()]

        if len(sequence_index) > MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE]
        sequences_length.append(len(sequence_index))
        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_output_index.append(sequence_index)
    
    return np.asarray(sequences_output_index), sequences_length

def dec_target_processing(value, dictionary):
    sequences_target_index = []
    for sequence in value:
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        sequence_index = [dictionary[word] for word in sequence.split()]
        
        if len(sequence_index) >= MAX_SEQUENCE:
            sequence_index = sequence_index[:MAX_SEQUENCE-1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]

        sequence_index += (MAX_SEQUENCE - len(sequence_index)) * [dictionary[PAD]]
        sequences_target_index.append(sequence_index)

    return np.asarray(sequences_target_index)

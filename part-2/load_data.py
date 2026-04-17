import os, random, re, string
from collections import Counter
from tqdm import tqdm
import pickle

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import nltk
nltk.download('punkt')
from transformers import T5TokenizerFast
import torch

PAD_IDX = 0

class T5Dataset(Dataset):

    def __init__(self, data_folder, split):
        '''
        Skeleton for the class for performing data processing for the T5 model.

        Some tips for implementation:
            * You should be using the 'google-t5/t5-small' tokenizer checkpoint to tokenize both
              the encoder and decoder output.
            * You want to provide the decoder some beginning of sentence token. Any extra-id on the
              T5Tokenizer should serve that purpose.
            * Class behavior should be different on the test set.
        '''
        self.tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
        self.split = split
        self.process_data(data_folder, split, self.tokenizer)

    def process_data(self, data_folder, split, tokenizer):
        nl_path = f'{data_folder}/{split}.nl'
        sql_path = f'{data_folder}/{split}.sql'

        self.nl_data = load_lines(nl_path)

        if split != 'test':
            self.sql_data = load_lines(sql_path)
        else:
            self.sql_data = None

        self.encoder_inputs = []
        for text in self.nl_data:
            tokens = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
            self.encoder_inputs.append(tokens.squeeze(0))

        self.decoder_inputs = []
        self.decoder_targets = []

        if self.sql_data is not None:
            for sql in self.sql_data:
                decoder_input_text = sql
                tokens = tokenizer.encode(decoder_input_text, return_tensors='pt', max_length=512, truncation=True)
                tokens = tokens.squeeze(0)

                bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
                decoder_input_ids = torch.cat([torch.tensor([bos_id]), tokens[:-1]])

                decoder_target_ids = tokens.clone()

                self.decoder_inputs.append(decoder_input_ids)
                self.decoder_targets.append(decoder_target_ids)

    def __len__(self):
        return len(self.nl_data)

    def __getitem__(self, idx):
        encoder_input = self.encoder_inputs[idx]

        if self.split == 'test':
            return encoder_input, None, None, None
        else:
            decoder_input = self.decoder_inputs[idx]
            decoder_target = self.decoder_targets[idx]
            return encoder_input, decoder_input, decoder_target

def normal_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for training and evaluation with the
    development or validation set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Returns: To be compatible with the provided training loop, you should be returning
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * decoder_inputs: Decoder input ids of shape BxT' to be fed into T5 decoder.
        * decoder_targets: The target tokens with which to train the decoder (the tokens following each decoder input)
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = []
    decoder_inputs_list = []
    decoder_targets_list = []

    for encoder_ids, decoder_input, decoder_target in batch:
        encoder_ids_list.append(encoder_ids)
        decoder_inputs_list.append(decoder_input)
        decoder_targets_list.append(decoder_target)

    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Pad decoder inputs and targets
    decoder_inputs = pad_sequence(decoder_inputs_list, batch_first=True, padding_value=PAD_IDX)
    decoder_targets = pad_sequence(decoder_targets_list, batch_first=True, padding_value=PAD_IDX)

    # Get initial decoder input (just the first token of decoder input)
    initial_decoder_inputs = decoder_inputs[:, 0:1]

    return encoder_ids, encoder_mask, decoder_inputs, decoder_targets, initial_decoder_inputs

def test_collate_fn(batch):
    '''
    Collation function to perform dynamic padding for inference on the test set.

    Inputs:
        * batch (List[Any]): batch is a list of length batch_size, where each index contains what
                             the dataset __getitem__ function returns.

    Recommended returns:
        * encoder_ids: The input ids of shape BxT to be fed into the T5 encoder.
        * encoder_mask: Mask of shape BxT associated with padding tokens in the encoder input
        * initial_decoder_inputs: The very first input token to be decoder (only to be used in evaluation)
    '''
    encoder_ids_list = []
    for encoder_ids, _, _, _ in batch:
        encoder_ids_list.append(encoder_ids)

    # Pad encoder inputs
    encoder_ids = pad_sequence(encoder_ids_list, batch_first=True, padding_value=PAD_IDX)
    encoder_mask = (encoder_ids != PAD_IDX).long()

    # Get BOS token as initial decoder input
    tokenizer = T5TokenizerFast.from_pretrained('google-t5/t5-small')
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    initial_decoder_inputs = torch.full((encoder_ids.shape[0], 1), bos_id, dtype=torch.long)

    return encoder_ids, encoder_mask, initial_decoder_inputs

def get_dataloader(batch_size, split, data_folder='data'):
    dset = T5Dataset(data_folder, split)
    shuffle = split == "train"
    collate_fn = normal_collate_fn if split != "test" else test_collate_fn

    dataloader = DataLoader(dset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return dataloader

def load_t5_data(batch_size, test_batch_size, data_folder='data'):
    train_loader = get_dataloader(batch_size, "train", data_folder)
    dev_loader = get_dataloader(test_batch_size, "dev", data_folder)
    test_loader = get_dataloader(test_batch_size, "test", data_folder)

    return train_loader, dev_loader, test_loader


def load_lines(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
    return lines

def load_prompting_data(data_folder):
    # TODO
    return train_x, train_y, dev_x, dev_y, test_x
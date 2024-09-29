# Imports from transformers
import json
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoTokenizer

def squad_json_to_dataframe_dev(input_file_path, record_path=['data', 'paragraphs', 'qas', 'answers'],
                                verbose=1):
    """
    input_file_path: path to the squad json file.
    record_path: path to deepest level in json file default value is
    ['data','paragraphs','qas','answers']
    verbose: 0 to suppress it default is 1
    """
    if verbose:
        print("Reading the json file")
    file = json.loads(open(input_file_path).read())
    if verbose:
        print("processing...")
    # parsing different level's in the json file
    m = pd.json_normalize(file, record_path[:-1])
    r = pd.json_normalize(file, record_path[:-2])

    # combining it into single dataframe
    idx = np.repeat(r['context'].values, r.qas.str.len())
    m['context'] = idx
    main = m[['id', 'question', 'context', 'answers']].set_index('id').reset_index()
    if verbose:
        print("shape of the dataframe is {}".format(main.shape))
        print("Done")
    return main

def get_preprocess_function(tokenizer):
    """
    Return the preprocess function of a tokenizer
    :param tokenizer:
    :return:
    """
    def preprocess_function(examples):
        questions = examples['question']
        contexts = examples['context']
        answers = examples['answers']

        inputs = tokenizer(questions, contexts, max_length=384, truncation=True, padding='max_length',
                           return_offsets_mapping=True)

        start_positions = []
        end_positions = []

        for i in range(len(questions)):
            if not answers[i]:  # Si pas de réponse
                start_positions.append(0)  # Valeur par défaut
                end_positions.append(0)  # Valeur par défaut
                continue

            # Only the firs answer is used
            first_answer = answers[i][0]['text']
            first_answer_start = answers[i][0]['answer_start']

            offsets = inputs['offset_mapping'][i]

            start_token = None
            end_token = None
            for idx, (start, end) in enumerate(offsets):
                if start <= first_answer_start < end:
                    start_token = idx
                if start < first_answer_start + len(first_answer) <= end:
                    end_token = idx
                    break

            if start_token is not None and end_token is not None:
                start_positions.append(start_token)
                end_positions.append(end_token)
            else:
                start_positions.append(-1)  # Default value
                end_positions.append(-1)  # Default value

        inputs.pop('offset_mapping')  # offset_mapping is not necessary to train the model

        # Converting everything into tensors
        inputs.update({
            'start_positions': torch.tensor(start_positions),
            'end_positions': torch.tensor(end_positions)
        })
        for column in inputs.keys():
            inputs[column] = torch.tensor(inputs[column])
        return inputs
    return preprocess_function

def get_tokenized_dataset_form_json(model_name, input_file_path = "dev-v2.0.json"):
    """
    Read and tokenize a dataset from a json file
    :param model_name:
    :param input_file_path:
    :return:
    """
    record_path = ['data', 'paragraphs', 'qas', 'answers']
    #Create the tokenizer/preprocess function for the dataset
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    preprocess_function = get_preprocess_function(tokenizer)

    dataset = squad_json_to_dataframe_dev(input_file_path=input_file_path, record_path=record_path)
    dataset = Dataset.from_pandas(dataset)
    tokenized_dataset = dataset.map(preprocess_function, batched=True,
                                    remove_columns=['id', 'question', 'context', 'answers'])
    tokenized_dataset = tokenized_dataset.select(range(4000))  # Decrease the size of the dataset to have a longer training
    return tokenized_dataset

def split_dataset(tokenized_dataset, test_size=0.2):
    """
    Split a tokenized dataset into train and test sets
    :param tokenized_dataset:
    :param test_size:
    :return:
    """
    splited_tokenized_dataset = tokenized_dataset.train_test_split(test_size=test_size, seed=42)
    tokenized_dataset_train = splited_tokenized_dataset['train']
    tokenized_dataset_validation = splited_tokenized_dataset['test']
    return tokenized_dataset_train, tokenized_dataset_validation
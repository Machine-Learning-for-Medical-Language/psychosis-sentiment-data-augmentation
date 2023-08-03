import os
from os.path import join
from fastapi import Body, FastAPI
from pydantic import BaseModel
from typing import List, Tuple, Dict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    glue_convert_examples_to_features
)
from transformers.data.processors.utils import InputFeatures, InputExample
from torch.utils.data.dataset import Dataset
import numpy as np
import logging
from time import time

app = FastAPI()
model_name = 'models/seed_18/model'
max_length = 128
batch_size = 64

all_domains=['appearance', 'interpersonal',  'mood', 'occupation', 'substance', 'thought_content', 'thought_process']

# Results are 0, 1, 2, which map to original labels of 0, 1, 9, which in turn map to Positive, Neutral, Negative
labels_map = {0: 'Neutral', 1: 'Positive', 2: 'Negative'}

class Document(BaseModel):
    document: str

class SentenceDocument(BaseModel):
    sentence: str

class SentenceResults(BaseModel):
    ''' Result is a mapping from an element in the domain array to a value in the labels_map '''
    results: List[Dict[str, str]]

class ClassificationDocumentDataset(Dataset):
    def __init__(self, features, label_list):
        self.features = features
        self.label_list = label_list
    def __len__(self):
        return len(self.features)
    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]
    def get_labels(self):
        return self.label_list
    @classmethod
    def from_instance_list(cls, inst_list, tokenizer, label_list, max_length=128):
        examples = []
        for (ind,inst) in enumerate(inst_list):
            guid = 'instance-%d' % (ind)
            examples.append(InputExample(guid=guid, text_a=inst[0], text_b=inst[1], label=None))
        features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list = label_list,
            output_mode='classification',
        )
        return cls(features, label_list)

# def create_instance_string(doc_text: str, offsets : List[int]):
#     start = max(0, offsets[0]-100)
#     end = min(len(doc_text), offsets[1]+100)
#     raw_str = doc_text[start:offsets[0]] + ' <e> ' + doc_text[offsets[0]:offsets[1]] + ' </e> ' + doc_text[offsets[1]:end]
#     return raw_str.replace('\n', ' ')

@app.on_event("startup")
async def startup_event():
    args = ['--output_dir', 'save_run/', '--per_device_eval_batch_size', str(batch_size), '--do_predict'] #, '--report_to', 'none']
    parser = HfArgumentParser((TrainingArguments,))
    training_args, = parser.parse_args_into_dataclasses(args=args)

    app.state.training_args = training_args
    config = AutoConfig.from_pretrained(model_name)
    app.state.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                  config=config)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, cache_dir=join(model_name, 'cache'), config=config)

    app.state.trainer = Trainer(
        model=model,
        args=app.state.training_args,
        compute_metrics=None,
    )

@app.post("/psych/sentiment/process_document")
async def process_document(doc: Document):
    doc_txt = doc.document
    # break into sentences:
    sents = []
    
    # ...

    # process sentences:
    return process_sentences(sents)

@app.post("/psych/sentiment/process_sentence")
async def process_sentence(sent: SentenceDocument):
    return process_sentences([sent.sentence])[0]

def process_sentences(sents: List[str], domains = all_domains):
    instances = []
    results = []

    for sent in sents:
        results.append({})
        for domain in domains:
            instances.append( (sent, domain) )
    

    dataset = ClassificationDocumentDataset.from_instance_list(instances, app.state.tokenizer, label_list = ["0", "1", "9"])
    predictions = app.state.trainer.predict(test_dataset=dataset).predictions
    for pred_ind,prediction in enumerate(predictions):
        sent_ind = pred_ind // len(dataset)
        domain_ind = pred_ind % len(domains)
        output = np.argmax(prediction)
        label = labels_map[output]
        results[sent_ind][domains[domain_ind]] = label

    return results

@app.post("/psych/sentiment/process_sentence_domain")
def process_sentence_domain(sent: str = Body(), domain: str = Body()):
    return process_sentences([sent], domains = [domain.lower()])[0]

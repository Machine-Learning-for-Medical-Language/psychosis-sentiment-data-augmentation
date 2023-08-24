import os
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Tuple, Dict
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
import numpy as np
import logging
from PyRuSH import RuSH
import logging

from cnlpt.api.cnlp_rest import get_dataset, initialize_cnlpt_model

app = FastAPI()
model_name = 'mlml-chip/psych-domains'
max_length = 128
batch_size = 64

all_domains=['appearance', 'interpersonal',  'mood', 'occupation', 'thought_content', 'thought_process', 'substance']

logger = logging.getLogger(__name__)

class DocumentResults(BaseModel):
    results: List[Dict[str,bool]]
    sentences: List[Tuple[int, int]]

class SentenceDocument(BaseModel):
    sentence: str

class Document(BaseModel):
    text: str

class SentenceResults(BaseModel):
    ''' Result is a mapping from an element in the domain array to a boolean value representing whether that domain is present in this sentence '''
    results: List[Dict[str, bool]]

@app.on_event("startup")
async def startup_event():
    initialize_cnlpt_model(app, model_name)
    
    # sentence segmenter
    app.state.rush = RuSH('/PHShome/tm685/psychosis-sentiment-data-augmentation/domains/conf/rush_rules.tsv')

@app.post("/psych/domain/process_document")
async def process_document(doc: Document):
    # break into sentences:
    print("Length of document string is %d" % len(doc.text))
    sents = []
    sent_spans = app.state.rush.segToSentenceSpans(doc.text)
    ret_spans = []
    for sent_span in sent_spans:
        sent = doc.text[sent_span.begin:sent_span.end]
        sents.append(sent)
        ret_spans.append( (sent_span.begin, sent_span.end))

    # process sentences:
    print("sending %d sentences to the process sentences method" % (len(sents)))
    results = process_sentences(sents)

    ret_val = DocumentResults(results=results, sentences=ret_spans)
    return ret_val

@app.post("/psych/domain/process_sentence")
async def process_sentence(sent: SentenceDocument) -> SentenceResults:
    return SentenceResults(process_sentences([sent.sentence])[0])

def process_sentences(sents: List[str]) -> List[Dict[str,bool]]:
    instances = []
    results = []

    print("Received %d sents in process_sentences" % (len(sents)))

    for sent in sents:
        instances.append(sent)

    print("Sending %d instances to the get_dataset method" % (len(instances)))

    dataset = get_dataset(instances, app.state.tokenizer, max_length=max_length)

    print("Received %d rows in the dataset" % (dataset.num_rows))

    predictions = app.state.trainer.predict(test_dataset=dataset).predictions

    assert len(predictions[0]) == len(instances), 'Model returned %d predictions for %d sentences' % (len(predictions[0]), len(instances))

    print('Model returned %d predictions for %d sentences' % (len(predictions[0]), len(instances)))

    for sent_ind in range(len(sents)):
        sent_result = dict()
        for domain_ind,task_name in enumerate(app.state.config.finetuning_task):
            domain_output = predictions[domain_ind][sent_ind].argmax()
            sent_result[task_name] = app.state.config.label_dictionary[task_name][domain_output]
            # sent_result[all_domains[domain_ind]] = True if domain_output == 1 else False
        results.append(sent_result)
    return results

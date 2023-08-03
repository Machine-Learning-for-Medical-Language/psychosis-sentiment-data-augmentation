import csv
import sys
import json
import random
from os.path import join

def clean_string(txt: str):
    while txt[0] == '"' and txt[-1] == '"':
        txt = txt[1:-1]
    if txt[-1] == "'":
        txt = txt[:-1]
    return txt

def main(args):
    if len(args) < 2:
        sys.stderr.write("2 required argument(s): <input csv file> <output dir>\n")
        sys.exit(-1)

    random.seed(718)
    domains = ['Appearance', 
                'Content (Thought Content)',
                'Interpersonal',
                'Mood',
                'Occupation',
                'Process (Thought Process)',
                'Substance']
    metadata = { "version" : "1.0",
                 "task" : "psych-domain",
                 "tasks" : domains,
                 "output_mode": "mtl",
                 }
    # labels = ['No', 'Yes']

    # train_instances = []
    # dev_instances = []
    train_data = { 'metadata': metadata, 'data': []}
    dev_data = {'metadata': metadata, 'data': []}

    with open(args[0], 'rt') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            # Default label is 'No', change to 'Yes' if we find it annotated for this sentence
            sent_id = row['SentenceID']
            labels = {x:'No' for x in domains}
            text = clean_string(row['Text'])
            inst = {'text': text, 'sentId': sent_id }
            inst.update(labels)

            for dom_ind in range(6):
                dom_str = 'Domain%d' % (dom_ind)
                dom = row[dom_str]
                # inst[dom] = 'No'
                if dom in labels:
                    inst[dom] = 'Yes'

            if random.random() > 0.8:
                # dev_instances.append(inst)
                dev_data['data'].append(inst)
            else:
                # train_instances.append(inst)
                train_data['data'].append(inst)


    # for inst in train_instances:
    #     train_data['data'].append(inst)

    # for inst in dev_instances:
    #     dev_data['data'].append(inst)

    with open(join(args[1], 'training.json'), 'wt') as fout:
        fout.write(json.dumps(train_data))
    with open(join(args[1], 'dev.json'), 'wt') as fout:
        fout.write(json.dumps(dev_data))

if __name__ == '__main__':
    main(sys.argv[1:])

import sys
import os
import requests
from os.path import join

def main(args):
    if len(args) < 3:
        sys.stderr.write('Required argument(s): <directory path> <domain classifier port> <sentiment classifier port>\n')
        sys.exit(-1)

    domain_url = 'http://localhost:%d/psych/domain/process_document' % int(args[1])
    sentiment_url = 'http://localhost:%d/psych/sentiment/process_sentence_domain/' % int(args[2])

    for fn in os.listdir(args[0]):
        print("Processing %s" % (fn,))
        with open(join(args[0], fn), 'rt') as f:
            text = f.read()

        r = requests.post(domain_url, json=text)
        if r.status_code != 200:
            sys.stderr.write('Error making reST call:\n%s\n' % (r.text))
            sys.exit(-1)
        
        results = r.json()
        sent_spans = results['sentences']
        predictions = results['results']

        print("Domain classifier returned %d sentence spans with %d predictions" % (len(sent_spans), len(predictions)))

        for sent_ind in range(len(sent_spans)):
            for domain,prediction in predictions[sent_ind].items():
                if prediction == True:
                    begin,end = sent_spans[sent_ind]
                    sent_txt = text[begin:end]

                    # make a call to the sentiment classifier with this text and domain
                    r = requests.post(sentiment_url, json={'sent': sent_txt, 'domain':domain})
                    results = r.json()
                    print("%s: (%d, %d) %s" % (fn, begin, end, results))

if __name__ == '__main__':
    main(sys.argv[1:])

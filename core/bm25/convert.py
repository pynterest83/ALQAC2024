import sys
sys.path.append('../')
from preprocessing import utils
import argparse

def match(item, law_data, type):
    if type == 'train':
        candidates = item['bm25_candidates'][:20]
    elif type == 'infer':
        candidates = item['bm25_candidates']
    new_item_candidates = []
    for candidate in candidates:
        new_candidate = {}
        new_candidate['a_id'] = candidate
        for law in law_data:
            if law['id'] == candidate:
                new_candidate['a_text'] = law['text']
                break
        new_candidate['bm25_score'] = item['bm25_scores'][candidate]
        new_item_candidates.append(new_candidate)
    return new_item_candidates

def convert(data, out_put, law_data, type):
    new_data = []
    for item in data:
        new_item = {}
        new_item['q_id'] = item['question_id']
        if item.get('segment-text') == None:
            new_item['q_text'] = item['segmented-text']
        if item.get('segmented-text') == None:
            new_item['q_text'] = item['segment-text']
        if item.get('relevant_articles') != None:
            new_item['pos_id'] = item['relevant_articles'][0]['id']
            for law in law_data:
                if law['id'] == new_item['pos_id']:
                    new_item['pos_text'] = law['text']
                    break
        new_item['candidates'] = match(item, law_data, type)
        
        new_data.append(new_item)
    utils.save_json(new_data, out_put)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, help='input data')
    parser.add_argument('--output', type=str, help='output data')
    parser.add_argument('--law_data', type=str, help='law data')
    parser.add_argument('--type', type=str, help='train or infer')
    args = parser.parse_args()
    data = utils.load_json(args.data)
    law_data = utils.load_json(args.law_data)
    convert(data, args.output, law_data, args.type)
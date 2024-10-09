from elasticsearch import Elasticsearch
import sys
sys.path.append('../')
from preprocessing import utils
import argparse

def mapping(client, _index):
    _mapping = {
        "settings": {
            "number_of_shards": 1,
            "index": {
                "similarity": {
                    "default": {
                        "type": "BM25",
                    }
                }
            }
        },
        "mappings": {
            "properties": {
                "paragraph": {
                    "type": "text"
                },
                "id": {
                    "type": "text"
                }
            }
        }
    }

    response = client.indices.create(
        index=_index,
        body=_mapping
    )

def indexing(client, _index, input_list):
    for i in range(len(input_list)):
        _id = input_list[i]["id"]
        _paragraphs = input_list[i]["text"].replace("\n\n", " ")
        client.index(index=_index, body={"id": _id, "paragraph": _paragraphs})
        print("Indexed " + str(i + 1) + "/" + str(len(input_list)) + " cases", end='\r', flush=True)

def searching(client, _index, _query:str, num_results:int):
    query = {
        "query": {
            "match": { "paragraph": _query }
        },
        "sort": [
            { "_score": "desc" }
        ],
        "track_scores": True,
        "size": num_results
    }

    return client.search(index=_index, body=query)

def get_result(client, _index, query, num_results):
    results = []
    response = searching(client, _index, query, num_results)
    for hit in response['hits']['hits']:
        results.append({
            "id": hit["_source"]["id"],
            "score": hit["_score"]
        })
    return results

def get_bm25_result(client, _index, query_data, file_name, num_results):
    results = []
    for query in query_data:
        result = get_result(client, _index, query["text"], num_results) # a list
        query["bm25_candidates"] = list(map(lambda x: x["id"], result))
        query["bm25_scores"] = dict(map(lambda x: (x["id"], x["score"]), result))
        results.append(query)
        print("Processed " + str(len(results)) + "/" + str(len(query_data)) + " queries", end='\r', flush=True)
    
    utils.save_json(results, file_name)

if __name__ == "__main__":
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Rank documents using BM25.')
    parser.add_argument('--elastic_link', type=str, required=True, help='Elasticsearch link.')
    parser.add_argument('--elastic_api_key', type=str, required=True, help='Elasticsearch API key.')
    parser.add_argument('--index_name', type=str, required=True, help='Index name.')
    parser.add_argument('--law_path', type=str, required=True, help='Path to law data.')
    parser.add_argument('--query_path', type=str, required=True, help='Path to query data.')
    parser.add_argument('--num_results', type=int, required=True, help='Number of results wanted.', default=2249)
    parser.add_argument('--output_path', type=str, required=True, help='Path for output.')

    args = parser.parse_args()

    # Initialize Elasticsearch client
    client = Elasticsearch(
        args.elastic_link,
        api_key=args.elastic_api_key
    )

    # Load data
    law_data = utils.load_json(args.law_path)
    query_data = utils.load_json(args.query_path)

    _index = args.index_name
    if not client.indices.exists(index=_index):
        mapping(client, _index)
        print("Index created")
    if client.indices.exists(index=_index):
        print("Index existed")
        # check if the index is empty
        response = client.search(index=_index, body={"query": {"match_all": {}}})
        if response['hits']['total']['value'] == 0:
            indexing(client, _index, law_data)
            print("Indexing completed")
        
        print("Ranking test started")
        get_bm25_result(client, _index, query_data, args.output_path, args.num_results)
        print("Ranking test completed")
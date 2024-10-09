import sys
import argparse
import csv
sys.path.append('../')
from preprocessing import utils
from ragatouille import RAGPretrainedModel
import numpy as np

def evaluate(model, data, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['qid', 'article_id', 'label', 'bm25_score_eng', 'colbert_score'])
        
        for query in data:
            try:
                results = model.search_encoded_docs(query['q_text'], k=2249)
                for result in results:
                    article_id = result['document_metadata']['a_id']
                    for candidate in query['candidates']:
                        if candidate['a_id'] == article_id:
                            csvwriter.writerow([query['q_id'], article_id, -1, candidate['bm25_score'], result['score']])
                    
            except KeyError as e:
                print(f"Missing key in data: {e}")
            except Exception as e:
                print(f"An error occurred: {e}")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Infer with the Colbert model.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data to infer.")
    parser.add_argument("--law-path", type=str, required=True, help="Path to the law data.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Colbert model.")
    parser.add_argument("--output-path", type=str, required=True, help="Path for the output CSV file.")
    args = parser.parse_args()

    # Load the law data
    infer_data = utils.load_json(args.data_path)    
    law_data = utils.load_json(args.law_path)

    # Load the model from the specified path
    RAG = RAGPretrainedModel.from_pretrained(args.model_path)
    RAG.encode([sample['text'] for sample in law_data], document_metadatas=[{"a_id": sample['id']} for sample in law_data])

    evaluate(RAG, infer_data, args.output_path)
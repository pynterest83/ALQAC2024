from sklearn.metrics.pairwise import cosine_similarity
import argparse
import sys
sys.path.append('../')
from preprocessing import utils
import csv
from sentence_transformers import SentenceTransformer

def evaluate(model, data):
    new_data = []
    for query in data:
        new_query = {}
        new_query['q_id'] = query['q_id']
        new_query['q_text'] = query['q_text']
        
        try:
            q_text = query['q_text']
            cands = query['candidates']
            
            # Encode question and all candidate answers in batch
            texts = [q_text] + [can['a_text'] for can in cands]
            embeddings = model.encode(texts)
            
            q_emb = embeddings[0].reshape(1, -1)
            cand_embs = embeddings[1:]
            
            # Compute cosine similarities
            similarities = cosine_similarity(q_emb, cand_embs)[0]
            
            for i in range(len(cands)):
                cands[i]['cosine_similarity'] = float(similarities[i])
        except KeyError as e:
            print(f"Missing key in data: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
        new_query['candidates'] = cands
        new_data.append(new_query)
    
    return new_data

def write_csv(data, csv_file):
    # Writing to CSV
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # Writing the header
        writer.writerow(["qid", "article_id", "bm25_score", "bert_score"])

        # Writing the data
        for item in data:
            q_id = item["q_id"]
            for candidate in item["candidates"]:
                writer.writerow([q_id, candidate["a_id"], candidate["score"], candidate["cosine_similarity"]])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate model and output results to CSV.")
    parser.add_argument("--data-path", type=str, required=True, help="Path to the data to infer.")
    parser.add_argument("--model-path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--output-csv-file", type=str, required=True, help="Path to the output CSV file.")
    args = parser.parse_args()

    # Load model
    model = SentenceTransformer(args.model_path)

    # Load data
    data = utils.load_json(args.data_path)

    # Evaluate
    evaluated_data = evaluate(model, data)

    # Write to CSV
    write_csv(evaluated_data, args.output_csv_file)
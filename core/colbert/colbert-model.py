import sys
import argparse
sys.path.append('../')
from preprocessing import utils
from ragatouille import RAGTrainer

def create_dataset(data):
    triplets = []
    for sample in data:
        if sample.get('pos_text') == None:
            continue
        query = sample['q_text']
        positive = sample['pos_text']
        for candidate in sample['candidates']:
            if candidate['a_id'] != sample['q_id']:
                negative = candidate['a_text']
                triplets.append((query, positive, negative))
    return triplets

def train(trainer, train_triplets, text_corpus):
    trainer.prepare_training_data(raw_data=train_triplets, data_out_path=args.data_out_path, all_documents=text_corpus, num_new_negatives=0, mine_hard_negatives=True)
    trainer.train(batch_size=4,
                  nbits=16,
                  maxsteps=500000,
                  use_ib_negatives=True,
                  dim=128,
                  learning_rate=1e-5,
                  doc_maxlen=512,
                  use_relu=False,
                  warmup_steps="auto",
                )
if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train LawColBERT model.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--law-path", type=str, required=True, help="Path to the law text corpus.")
    parser.add_argument("--data-out-path", type=str, required=True, help="Path to save processed training data.")
    args = parser.parse_args()

    trainer = RAGTrainer(model_name="LawColBERT", pretrained_model_name="colbert-ir/colbertv2.0", language_code="en")

    # Load training data and law text corpus
    train_data = utils.load_json(args.train_path)
    text_corpus = [sample["text"] for sample in utils.load_json(args.law_path)]  # Assuming utils.load_json can be used to load the law text corpus
    train_triplets = create_dataset(train_data)

    train(trainer, train_triplets, text_corpus)
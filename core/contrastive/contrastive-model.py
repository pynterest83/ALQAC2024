import sys
sys.path.append('../')
from preprocessing import utils
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainingArguments, SentenceTransformerTrainer, BatchSamplers
import torch
from datasets import Dataset
import argparse
from sentence_transformers.evaluation import TripletEvaluator

def create_dataset(data):
    anchor = []
    positive = []
    negative = []
    for item in data:
        anchor.append(item['q_text'])
        positive.append(item['pos_text'])
        negative.append(item['neg_text'])
    dataset = Dataset.from_dict({'anchor': anchor, 'positive':positive,
                                'negative':negative})
    return dataset

def train(model, loss, train_dataset, dev_dataset, test_dataset, output_dir):
    args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir='./output/triplet-loss-model',
        # Optional training parameters:
        num_train_epochs=10,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=False,  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers.NO_DUPLICATES,  # losses that use "in-batch negatives" benefit from no duplicates
        # Optional tracking/debugging parameters:
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=100,
        run_name="triplet-model-alqac",  # Will be used in W&B if `wandb` is installed
    )

    dev_evaluator = TripletEvaluator(
        anchors=dev_dataset["anchor"],
        positives=dev_dataset["positive"],
        negatives=dev_dataset["negative"],
        name="triplet-dev"
    )
    dev_evaluator(model)

    test_evaluator = TripletEvaluator(
        anchors=test_dataset["anchor"],
        positives=test_dataset["positive"],
        negatives=test_dataset["negative"],
        name="triplet-dev"
    )
    test_evaluator(model)

    # Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset={
            dev_dataset,
            test_dataset
        },
        loss=loss,
        evaluator={
            dev_evaluator,
            test_evaluator
        }
    )
    trainer.train()

    # Save the trained model
    model.save_pretrained(output_dir)

if __name__ == "__main__":
    # Ensure the script runs on a GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the pre-trained SentenceTransformer model
    model = SentenceTransformer('keepitreal/vietnamese-sbert').to(device)

    # Loss function
    loss = losses.TripletLoss(model)

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Train a SentenceTransformer model with triplet loss.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to the training data.")
    parser.add_argument("--dev-path", type=str, required=True, help="Path to the development data.")
    parser.add_argument("--test-path", type=str, required=True, help="Path to the test data.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the trained model.")
    args = parser.parse_args()

    # Load datasets
    train_data = utils.load_json(args.train_path)
    dev_data = utils.load_json(args.dev_path)
    test_data = utils.load_json(args.test_path)

    # Assuming create_dataset function is defined and un-commented
    train_dataset = create_dataset(train_data)
    dev_dataset = create_dataset(dev_data)
    test_dataset = create_dataset(test_data)

    # Train the model
    train(model, loss, train_dataset, dev_dataset, test_dataset, args.output_dir)
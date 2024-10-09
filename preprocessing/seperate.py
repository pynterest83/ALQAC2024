import argparse
import utils
import random

def split(data, train_percent, dev_percent):
    # Shuffle data
    random.shuffle(data)

    # Calculate split indices
    total_len = len(data)
    train_size = int(total_len * train_percent)
    dev_size = int(total_len * dev_percent)

    # Split the data
    train_data = data[:train_size]
    dev_data = data[train_size:train_size + dev_size]
    test_data = data[train_size + dev_size:]

    return train_data, dev_data, test_data

if __name__ == '__main__':
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Split data into train, dev, and test sets.')
    parser.add_argument('--data_path', type=str, default='train.json')
    parser.add_argument('--train_weight', type=float, default=0.8)
    parser.add_argument('--dev_weight', type=float, default=0.1)
    parser.add_argument('--train_output', type=str, default='data_train.json')
    parser.add_argument('--dev_output', type=str, default='data_dev.json')
    parser.add_argument('--test_output', type=str, default='data_test.json')

    args = parser.parse_args()

    # Load data
    data = utils.load_json(args.data_path)

    # Split data
    train_data, dev_data, test_data = split(data, args.train_weight, args.dev_weight)

    # Save the split data into separate files
    utils.save_json(train_data, args.train_output)
    utils.save_json(dev_data, args.dev_output)
    utils.save_json(test_data, args.test_output)

    print(f"Data split into: {len(train_data)} train, {len(dev_data)} dev, and {len(test_data)} test instances.")
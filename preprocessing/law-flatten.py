import argparse
import utils

# Flatten
def match(mapped_id, articles):
    new_articles = []
    for article in articles:
        new_id = f"{mapped_id}-{article['id']}"
        new_article = {
            "id": new_id,
            "text": article['text']
        }
        new_articles.append(new_article)
    return new_articles

def convert(data, mapping):
    new_data = []
    for item in data:
        law_id = item['id']
        mapped_id = next((key for key, value in mapping.items() if value == law_id), None)
        if mapped_id:
            new_articles = match(mapped_id, item['articles'])
            new_data.extend(new_articles)
    return new_data

if __name__ == '__main__':
    # Create the parser
    parser = argparse.ArgumentParser(description='Flatten law documents based on ID mapping.')

    # Add arguments
    parser.add_argument('--law_path', default = 'law.json')
    parser.add_argument('--mapping_path', default = 'law-id-mapping.json')
    parser.add_argument('--output_path', default = 'law-flatten.json')

    # Parse arguments
    args = parser.parse_args()

    # Use the parsed arguments
    file_law_path = args.law_path
    file_id_mapping_path = args.mapping_path
    file_law_flatten = args.output_path

    # Load data
    data = utils.load_json(file_law_path)
    mapping = utils.load_json(file_id_mapping_path)

    # Flatten data
    mapped_data = convert(data, mapping)

    # Save flattened data
    utils.save_json(mapped_data, file_law_flatten)
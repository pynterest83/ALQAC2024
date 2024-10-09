import argparse
import utils
import py_vncorenlp

def segment_text(text):
    output = rdrsegmenter.word_segment(text)
    output = ' '.join(output)
    return output

def segment_corpus(data):
    for i in range(len(data)):
        data[i]["segment-text"] = segment_text(data[i]["text"])
        print("Processed " + str(i + 1) + "/" + str(len(data)) + " cases", end='\r', flush=True)
    return data

if __name__ == '__main__':
    
    # Setup argument parser
    parser = argparse.ArgumentParser(description='Segment text in the dataset.')
    parser.add_argument('--lib_path', type=str, required=True, help='Path to the VnCoreNLP library.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the input data file.')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save the segmented data.')

    args = parser.parse_args()

    # Download and setup VnCoreNLP
    py_vncorenlp.download_model(args.lib_path)
    rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=args.lib_path)

    # Load data
    data = utils.load_json(args.data_path)

    # Segment data
    segmented_data = segment_corpus(data)

    # Save segmented data
    utils.save_json(segmented_data, args.output_path)

    print(f"Segmentation completed. Output saved to {args.output_path}")
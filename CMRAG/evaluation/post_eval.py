""" Evaluate the accuracy of different types of sources or questions """
import argparse
import os

from utils.utils import load_json, save_json
from dataset.load_data import load_testqa, load_img_root


def setup_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_name",
        type=str,
        default="MMLongBench",
        choices=["MMLongBench", "ViDoSeek", "SlideVQA"],
        help="Name of the dataset to evaluate."
    )
    parser.add_argument(
        '--save_file',
        type=str,
        default='results_hotpotqa_num7404_top2_Qwen2.5-7B-Instruct.json',
        help='The name of the input file.'
    )

    args = parser.parse_args()
    return args


def main():
    args = setup_args()

    # Count the numerbs of files and images
    root = load_img_root(data_name=args.data_name)
    num_files = len(os.listdir(root))
    num_images = 0
    for d in os.listdir(root):
        for f in os.listdir(os.path.join(root, d)):
            if f.endswith(('.png', '.jpg', '.jpeg')) and "_" not in f and len(f.split('.')) == 2:
                num_images += 1
    print(f"Dataset {args.data_name} has {num_files} files and {num_images} images.")

    # Count the number of questions and how many pages of each question are required to answer
    qa_data = load_testqa(data_name=args.data_name, num_samples=-1)
    num_questions = len(qa_data)
    print(f"Dataset {args.data_name} has {num_questions} questions.")
    page_counts = [len(item['doc_page']) if isinstance(item['doc_page'], list) else 1 for item in qa_data]
    avg_pages = sum(page_counts) / num_questions
    print(f"On average, each question requires {avg_pages:.2f} pages to answer.")

    # # Load the prediction results
    # pred_data = load_json(args.save_file)
    # print(f"Loaded {len(pred_data)} samples from {args.save_file}")

    # # Load original testqa data to get source_type and query_type
    # qa_data = load_testqa(data_name=args.data_name, num_samples=-1)
    # print(f"Loaded {len(qa_data)} samples from original testqa data")
    # # Count all unique source_type and query_type
    # query_types = [item['query_type'] for item in qa_data]
    # query_types = list(set(query_types))
    # source_types = [item['source_type'] for item in qa_data]
    # source_types = list(set(source_types))
    # print(f"Unique query types: {query_types}")
    # print(f"Unique source types: {source_types}")




if __name__ == "__main__":
    main()
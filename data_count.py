import os
from sklearn.model_selection import train_test_split


def split_and_save_integrated_data(datasets, output_dir, train_ratio=0.7, dev_ratio=0.15):
    # Combine all triplets from multiple datasets
    all_lines = []
    for dataset in datasets:
        with open(dataset, 'r', encoding='utf-8') as f:
            all_lines.extend([line.strip() for line in f if line.strip()])

    # Shuffle and split the combined dataset
    train_lines, temp_lines = train_test_split(all_lines, test_size=1 - train_ratio, random_state=42)
    dev_lines, test_lines = train_test_split(temp_lines, test_size=0.5, random_state=42)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Save each split
    def save_split(data, filename):
        with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
            f.write('\n'.join(data) + '\n')

    save_split(train_lines, 'train_triplets.txt')
    save_split(dev_lines, 'dev_triplets.txt')
    save_split(test_lines, 'test_triplets.txt')


# File paths for the datasets
datasets = [
    "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14lap/total_triplets.txt",
    "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/14res/total_triplets.txt",
    "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/15res/total_triplets.txt",
    "SemEval-Triplet-data/ASTE-Data-V2-EMNLP2020/16res/total_triplets.txt"
]

# Directory for saving the integrated splits
output_directory = "SemEval-Triplet-data/Integrated-Splits"

# Perform the split and save
split_and_save_integrated_data(datasets, output_directory)

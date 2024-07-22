import random
import pickle
import argparse
from sklearn.model_selection import train_test_split

BASE_DIR = "..."

def load_data(total_samples, multipliers, random_seed=42):
    random.seed(random_seed)

    data_file_names = [
        f"{BASE_DIR}/data/common_crawl_dedup.pkl",
        f"{BASE_DIR}/data/github_dedup.pkl",
        f"{BASE_DIR}/data/books_dedup.pkl",
        f"{BASE_DIR}/data/wikipedia_dedup.pkl",
        f"{BASE_DIR}/data/c4_dedup.pkl",
        f"{BASE_DIR}/data/stackexchange_dedup.pkl",
        f"{BASE_DIR}/data/arxiv_dedup.pkl"
    ]

    num_files = len(data_file_names)
    assert len(multipliers) == num_files, "Number of multipliers must match the number of data files."

    base_samples = total_samples / num_files 

    zero_indices = [i for i, x in enumerate(multipliers) if x == 0]
    non_zero_indices = [i for i, x in enumerate(multipliers) if x != 0]

    if zero_indices:
        for i in zero_indices:
            multipliers[i] = base_samples / total_samples / 10
    
    remaining_weight = 1 - sum(multipliers[i] for i in zero_indices)
    other_weights_sum = sum(multipliers[i] for i in non_zero_indices)
    normalized_multipliers = [x / other_weights_sum * remaining_weight if i in non_zero_indices else multipliers[i] for i, x in enumerate(multipliers)]

    print("Normalized multipliers:", normalized_multipliers)

    number_of_samples = [int(total_samples * multiplier) for multiplier in normalized_multipliers]

    data = []
    for i, file_path in enumerate(data_file_names):
        with open(file_path, 'rb') as file:
            file_data = pickle.load(file)
            if not isinstance(file_data, list):
                file_data = file_data.tolist()
            sampled_data = random.sample(file_data, number_of_samples[i])
            data.append(sampled_data)

    print("Number of samples:", number_of_samples)
    return data

def write_to_file(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(item + "\n")

def main(args):
    multipliers = [float(x) for x in args.multipliers.split(',')]
    total_samples = args.n_train + args.n_val
    data = load_data(total_samples, multipliers, random_seed=42)
    test_ratio = args.n_val / (args.n_train + args.n_val)

    train_data = []
    val_data = []

    for data_set in data:
        t_data, v_data = train_test_split(data_set, test_size=test_ratio, shuffle=True, random_state=42)
        train_data.extend(t_data)
        val_data.extend(v_data)
    random.shuffle(train_data)

    write_to_file(train_data, args.train_file_path)
    write_to_file(val_data, args.val_file_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Load data, reweight and split into train and validation sets.")
    parser.add_argument("--n_train", type=int, required=True, help="Number of training samples")
    parser.add_argument("--n_val", type=int, required=True, help="Number of validation samples")
    parser.add_argument("--train_file_path", type=str, required=True, help="Path to save the training data")
    parser.add_argument("--val_file_path", type=str, required=True, help="Path to save the validation data")
    parser.add_argument("--multipliers", type=str, required=True, help="Comma-separated list of data multipliers")
    
    args = parser.parse_args()
    main(args)

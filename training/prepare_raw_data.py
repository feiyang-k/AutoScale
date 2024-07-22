from datasets import load_dataset
import random
import os
import pickle

def sample_and_save_subset_streaming(subset_name, num_samples=8000000, max_length=1000, output_dir="/home/ubuntu/dc4fm"):
    dataset = load_dataset("togethercomputer/RedPajama-Data-1T", subset_name, split='train', streaming=True)
    output_file_path = os.path.join(output_dir, f"{subset_name}.pkl")
    samples = []

    try:
        count = 0
        for sample in dataset:
            text = sample['text']

            if len(text) < max_length:
                continue
            
            if subset_name == "wikipedia":
                if "'language': 'en'" not in sample['meta']:
                    continue
                text = text[:max_length]
            elif subset_name in ["github", "arxiv"]:
                start_idx = random.randint(0, len(text) - max_length)
                text = text[start_idx:start_idx + max_length]
            else:
                text = text[:max_length]

            samples.append(text)
            count += 1
            if count >= num_samples:
                break
            if count % 10000 == 0:
                print(f"Processed {count} samples for {subset_name}")

    except Exception as e:
        print(f"Error occurred after processing {count} samples: {str(e)}")

    with open(output_file_path, 'wb') as file:
        pickle.dump(samples, file)
    
    print(f"{subset_name} processed and saved up to the last valid sample before the error.")

subsets = ["common_crawl", "c4", "github", "arxiv", "wikipedia", "stackexchange"]
for subset in subsets:
    sample_and_save_subset_streaming(subset_name=subset)

print("Attempted processing of all subsets.")

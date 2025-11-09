import argparse
import json
import random

def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into known and unknown image IDs")
    parser.add_argument('--dataset_name', type=str, default='FHM', choices=['FHM', 'MAMI', 'HSOL', 'Sent140'], help='Name of the dataset (e.g., FHM, MAMI, etc.)')
    parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder for the model')
    
    return parser.parse_args()

def load_data(anno_path):
    data = []
    with open(anno_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def shuffle_data(data):
    random.seed(42)
    random.shuffle(data)
    return data

def split_data(data, data_type):
    video_id_split = {}
    if data_type == "test" or data_type == "dev":
        known_ratio_list = [0]
    else:
        known_ratio_list = [50, 100, 250]

    for known_ratio in known_ratio_list:
        data_len = len(data)
        pnu_known = []
        pnu_unknown = []
        for row in data[:known_ratio]:
            pnu_known.append(row["image_id"])
        for row in data[known_ratio:]:
            pnu_unknown.append(row["image_id"])

        pu_known = []
        pu_unknown = []
        for row in data:
            if row["label"] == 1 and row["image_id"] in pnu_known:
                pu_known.append(row["image_id"])
            else:
                pu_unknown.append(row["image_id"])

        video_id_split[str(known_ratio)] = {
            "pu_known": {
                "image_id": pu_known,
                "length": len(pu_known),
            },
            "pu_unknown": {
                "image_id": pu_unknown,
                "length": len(pu_unknown),
            },
            "pnu_known": {
                "image_id": pnu_known,
                "length": len(pnu_known),
            },
            "pnu_unknown": {
                "image_id": pnu_unknown,
                "length": len(pnu_unknown),
            }
        }

    return video_id_split

def save_split_data(video_id_split, output_path):
    with open(output_path, "w") as f:
        json.dump(video_id_split, f)

def main():
    args = parse_args()
    for data_type in ["train", "test", "dev"]:
        dataset_path = f"{args.root_folder}/datasets/{args.dataset_name}/{data_type}.jsonl"
        output_path =  f"{args.root_folder}/datasets/{args.dataset_name}/image_id_split_{data_type}.json"
        
        data = load_data(dataset_path)
        shuffled_data = shuffle_data(data)
        video_id_split = split_data(shuffled_data, data_type)
        save_split_data(video_id_split, output_path)

if __name__ == "__main__":
    main()

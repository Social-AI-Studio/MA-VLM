import subprocess
import os
import json
import time
import torch
import argparse

# Function to load image ID split
def load_round_upbound(root_folder, dataset_name, known_number, topk):
    image_id_split_path = f"{root_folder}/datasets/{dataset_name}/image_id_split_train.json"
    with open(image_id_split_path, "r") as f:
        image_id_split = json.load(f)
    unknown = image_id_split[str(known_number)]["pnu_unknown"]["image_id"]
    return len(unknown) // topk

# Function to run a Python script with specified arguments
def run_script(script_name, args, round_number, inference_data_type):
    command = [
        "python3", script_name,
        "--root_folder", args.root_folder,
        "--dataset_name", args.dataset_name,
        "--pseudo_label_type", args.pseudo_label_type,
        "--known_number", str(args.known_number),
        "--round_number", str(round_number),
        "--topk", str(args.topk),
        "--batch_size", str(args.batch_size),
        "--num_workers", str(args.num_workers),
        "--lr", str(args.lr),
        "--gamma", str(args.gamma),
        "--pos_unknown_value", str(args.pos_unknown_value),
        "--neg_unknown_value", str(args.neg_unknown_value),
        "--pi_p", str(args.pi_p),
        "--epochs", str(args.epochs),
        "--seed", str(args.seed),
        "--device", str(args.device),
        "--inference_data_type", str(inference_data_type),
        "--model_id", str(args.model_id)
    ]

    try:
        print(f"Running {script_name} with round_number = {round_number}", flush=True)
        subprocess.run(command, check=True)
        print(f"{script_name} completed successfully.", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}", flush=True)
        return False
    return True

# Function to refresh the environment
def refresh_environment():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Refreshing environment...", flush=True)
    time.sleep(2)  # Simulating some refresh time

# Main function
def main():

    args = parse_args()
    checkpoints_dir = os.path.join(args.root_folder, "checkpoints", args.dataset_name)
    meta_data_classifier_ma_vlms_output_dir = os.path.join(args.root_folder, "scripts/meta_data/Classifier_MA_VLMs_output", args.dataset_name)
    meta_data_classifier_output_dir = os.path.join(args.root_folder, "scripts/meta_data/Classifier_output", args.dataset_name)
    for dir_path in [checkpoints_dir, meta_data_classifier_ma_vlms_output_dir, meta_data_classifier_output_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)    
         
    round_upbound = load_round_upbound(args.root_folder, args.dataset_name, args.known_number, args.topk)
    for round_number in range( round_upbound + 1): 
        # Step 1: Train classifier
        print("-" * 90, flush=True)
        print(f"Start Round {round_number}:", flush=True)
        if not run_script(f"{args.root_folder}/scripts/Classifier_train.py", args, round_number, "train"):
            break  

        refresh_environment()

        # Step 2: Inference classifier
        if not run_script(f"{args.root_folder}/scripts/Classifier_inference.py", args, round_number, "train"):
            break  

        refresh_environment()

        # Step 3: Combine with MA_VLMs to generate pseudo label
        if not run_script(f"{args.root_folder}/scripts/pseudo_label_generation.py", args, round_number, "train"):
            break  
        refresh_environment()

        # Evaluate the initial results for SupOnly
        if(int(round_number) == 0 or int(round_number) == round_upbound):
            if not run_script(f"{args.root_folder}/scripts/Classifier_inference.py", args, round_number, "test"):
                break
            refresh_environment()
            

    print("All scripts completed successfully.", flush=True)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Run MAST-PNU workflow")

    # Common arguments
    parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder for the model')
    parser.add_argument('--known_number', type=int, required=True,  choices=[50, 100, 250], help='Number of known data.')
    parser.add_argument('--dataset_name', type=str, default="FHM", choices=['FHM', 'MAMI', 'HSOL', 'Sent140'], help='Name of the dataset (e.g., FHM, MAMI, etc.)')
    parser.add_argument('--pseudo_label_type', type=str, default="classifier_ma_vlm", choices=['classifier_ma_vlm', 'classifier', 'ma_vlm'], help='Type of pseudo-labeling model.')
    parser.add_argument('--gamma', type=float, default=0.1, help='Gamma value for PNULoss')
    parser.add_argument('--pos_unknown_value', type=float, default=0.67, help='Soft label for Positive Agreed-Unknown.')
    parser.add_argument('--neg_unknown_value', type=float, default=0.33, help='Soft label for Negative Agreed-Unknown.')
    
    parser.add_argument('--topk', type=int, default=500, help="Top-k most confident unknown data selected in each round.")
    parser.add_argument('--pi_p', type=float, default=0.5, help='pi_p value for PNULoss.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and evaluation.')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading.')
    parser.add_argument('--lr', type=float, default=5e-6, help='Learning rate for optimizer.')
    
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    parser.add_argument('--device', type=str, default='cuda:0', help='Device for training (cuda or cpu).')
    parser.add_argument('--model_id', type=str, default="openai/clip-vit-large-patch14", help="Pretrained model identifier (e.g., Clip vit large)")

    return parser.parse_args()

if __name__ == "__main__":
    main()

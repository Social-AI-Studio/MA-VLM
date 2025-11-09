import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, CLIPModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
import pandas as pd
from PIL import Image
import random, json, os, csv, re
import numpy as np

# Set seed function
def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# Dataset class
class MAST_PNU_Dataset(Dataset):
    def __init__(self, dataframe, unknown, pseudo_labeled_unknown, root_folder="", transform=None, split="train"):
        self.transform = transform
        self.split = split
        self.unknown = unknown
        self.pseudo_labeled_unknown = pseudo_labeled_unknown
        self.root_folder = root_folder
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        all_data = []
        for row in dataframe:
            image_id =  row["image_id"]
            if split == "train":
                if image_id in self.unknown and image_id not in self.pseudo_labeled_unknown:
                    all_data.append({
                        "image_id": image_id,
                        "image_path": row["image_path"],
                        "text": row["text"],
                        "label": row["label"]
                    })
            else:
                all_data.append({
                    "image_id": image_id,
                    "image_path": row["image_path"],
                    "text": row["text"],
                    "label": row["label"]
                })

        self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        image_id = entry["image_id"]
        image_path = os.path.join(self.root_folder, entry["image_path"])
        
        # Check if the image exists, otherwise use a placeholder (black image)
        if os.path.exists(image_path):
            image = Image.open(image_path).convert("RGB")
        else:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))  # Create a black image
        
        # Apply transformations if provided
        if self.image_transform:
            image = self.image_transform(image)

        text = entry["text"]
        label = entry["label"]

        return image_id, image, text, label

# Collate function

# JSONL reader
def read_jsonl(anno_path):
    data = []
    with open(anno_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

# Evaluation function
def evaluate(model, data_loader, device, save_every_n_batches=10, output_path=None):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    batch_counter = 0
    csv_rows = []

    with torch.no_grad():
        for image_id, images, text_inputs, targets in data_loader:
            batch_counter += 1

            images = images.to(device)
            targets = targets.to(device)
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            logits = model(images, input_ids, attention_mask)

            probs = torch.sigmoid(logits).cpu()
            all_probs.extend(probs.tolist())

            preds = (probs >= 0.5).long()
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.cpu().tolist())

            for image_id, target, pred, prob in zip(image_id, targets, preds, probs):
                csv_rows.append([str(image_id).replace(".0", ""), int(target), int(pred), float(prob)])

            if batch_counter % save_every_n_batches == 0 and output_path:
                with open(output_path, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerows(csv_rows)
                csv_rows = []

    with open(output_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    acc = accuracy_score(all_labels, all_preds)
    m_f1 = f1_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary')
    precision = precision_score(all_labels, all_preds, pos_label=1, average='binary')
    recall = recall_score(all_labels, all_preds, pos_label=1, average='binary')
    roc_auc = roc_auc_score(all_labels, all_preds)

    return acc, m_f1, f1, precision, recall, roc_auc

# CLIP Binary Classifier Model
class CLIPBinaryClassifier(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.clip = clip_model
        embed_dim = clip_model.config.projection_dim
        self.classifier = nn.Linear(embed_dim * 2, 1)

    def forward(self, pixel_values, input_ids, attention_mask):
        outputs = self.clip(pixel_values=pixel_values, input_ids=input_ids, attention_mask=attention_mask)
        image_embeds = outputs.image_embeds
        text_embeds = outputs.text_embeds
        joint_embeds = torch.cat([image_embeds, text_embeds], dim=-1)
        logits = self.classifier(joint_embeds)
        return logits

# Main function to run the model
def main(args):
    set_seed(args.seed)
    topl = args.topk
    
    # Dataset and paths
    dataset_path = os.path.join(args.root_folder, f"datasets/{args.dataset_name}/{args.inference_data_type}.jsonl")
    output_path = os.path.join(args.root_folder, f"scripts/meta_data/Classifier_output/{args.dataset_name}/round_{args.round_number}_{args.inference_data_type}.csv")
    image_id_split_path = os.path.join(args.root_folder, f"datasets/{args.dataset_name}/image_id_split_{args.inference_data_type}.json")

    # Load dataset split
    with open(image_id_split_path, "r") as f:
        image_id_split = json.load(f)
    # Pseudo-label handling
    known_number = args.known_number if args.inference_data_type == "train" else 0
    unknown = image_id_split[str(known_number)]["pnu_unknown"]["image_id"]
    unknown = [str(id) for id in unknown]

    positive_agreed_unknown, negative_agreed_unknown, disagreed_unknown = [], [], []
    pseudo_labeled_unknown = []
    for i in range(args.round_number):
        df = pd.read_csv(os.path.join(args.root_folder, f"scripts/meta_data/Classifier_MA_VLMs_output/{args.dataset_name}/{args.pseudo_label_type}_round_{i}.csv"), dtype={'image_id': str})
        positive_agreed_unknown_round, negative_agreed_unknown_round, disagreed_unknown_round = [], [], []
                
        for _, row in df.iterrows():
            image_id =  row["image_id"]
            if row["pseudo_label"] == 1:
                positive_agreed_unknown_round.append(image_id)
            elif row["pseudo_label"] == 0:
                negative_agreed_unknown_round.append(image_id)
            elif row["pseudo_label"] == -1:
                disagreed_unknown_round.append(image_id)

        total_len = len(positive_agreed_unknown_round) + len(negative_agreed_unknown_round) + len(disagreed_unknown_round)
        positive_agreed_unknown += positive_agreed_unknown_round[:int(args.topk * len(positive_agreed_unknown_round) / total_len)]
        negative_agreed_unknown += negative_agreed_unknown_round[:int(args.topk * len(negative_agreed_unknown_round) / total_len)]
        disagreed_unknown += disagreed_unknown_round[:int(args.topk * len(disagreed_unknown_round) / total_len)]
    pseudo_labeled_unknown = positive_agreed_unknown + negative_agreed_unknown + disagreed_unknown
   
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    def collate_fn(batch):
        video_id, images, texts, labels = zip(*batch)
        images = torch.stack(images)
        text_inputs = tokenizer(list(texts), return_tensors="pt", padding='max_length', truncation=True, max_length=77)
        labels = torch.tensor(labels, dtype=torch.long)
        return video_id, images, text_inputs, labels
    clip_model = CLIPModel.from_pretrained(args.model_id)
    model = CLIPBinaryClassifier(clip_model).to(args.device)

    overall_best_model_path =  json.load(open(f"{args.root_folder}/checkpoints/{args.dataset_name}/best_round_meta.json"))["best_model_path"]
    model.load_state_dict(torch.load(overall_best_model_path))

    # Dataset
    inference_data = read_jsonl(dataset_path)
    inference_dataset = MAST_PNU_Dataset(dataframe=inference_data, unknown=unknown, pseudo_labeled_unknown=pseudo_labeled_unknown, root_folder=args.root_folder, split=args.inference_data_type)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    # Evaluation
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["image_id", "gt_label", "pseudo_label", "pseudo_label_prob"])  # Write header

    acc, m_f1, f1, precision, recall, roc_auc = evaluate(
            model,
            inference_dataloader,
            device=args.device,
            save_every_n_batches=10,
            output_path=output_path
        )

    if(args.inference_data_type == "test"):
        # Print evaluation metrics
        print(f"  Round {args.round_number} test set evalutaion: ", flush=True)
        print(f"  Accuracy = {acc:.4f}", flush=True)
        print(f"  Macro F1 = {m_f1:.4f}", flush=True)
        print(f"  Roc AUC = {roc_auc:.4f}", flush=True)
        print(f"  F1 (pos) = {f1:.4f}")
        print(f"  Precision (pos) = {precision:.4f}", flush=True)
        print(f"  Recall (pos) = {recall:.4f}", flush=True)

if __name__ == "__main__":
    # Define argument parser
    parser = argparse.ArgumentParser(description="Train CLIP Binary Classifier with MAST-PNU")

    # Add command-line arguments
    parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder for the model')
    parser.add_argument('--round_number', type=int, required=True, help='Current round.')
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
    parser.add_argument('--inference_data_type', type=str, default="train", choices=['train', 'test'], help="Inference data type: 'train' for pseudo-label generation, 'test' for evaluation.")
    parser.add_argument('--model_id', type=str, default="openai/clip-vit-large-patch14", help="Pretrained model identifier (e.g., Clip vit large)")

    # Parse the arguments
    args = parser.parse_args()

    # Run the main function with the parsed arguments
    main(args)


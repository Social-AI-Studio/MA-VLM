import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, CLIPModel
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import pandas as pd
from PIL import Image
import random, json, os, re
import numpy as np
import argparse

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def read_jsonl(anno_path):
    data = []
    with open(anno_path, "r") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


class MAST_PNU_Dataset(Dataset):
    def __init__(self, dataframe, split="train", tokenizer=None, root_folder="" , known=[], positive_agreed_unknown=[], negative_agreed_unknown=[], disagreed_unknown=[]):
        self.tokenizer = tokenizer
        self.split = split
        self.root_folder = root_folder
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),  # Converts to [0,1]
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # Prepare the data based on the split (train or test)
        if split == "train":
            all_data = []
            
            for row in dataframe:
                image_id =  row["image_id"]
                image_path = row["image_path"]
                text = row["text"]
                label = row["label"]
                
                if image_id in known:
                    all_data.append({
                        "image_path": image_path,
                        "text": text,
                        "label": label
                    })
                elif image_id in positive_agreed_unknown and image_id not in known:
                    all_data.append({
                        "image_path": image_path,
                        "text": text,
                        "label": -1
                    })
                elif image_id in negative_agreed_unknown and image_id not in known:
                    all_data.append({
                        "image_path": image_path,
                        "text": text,
                        "label": -2
                    })
                elif image_id in disagreed_unknown and image_id not in known:
                    all_data.append({
                        "image_path": image_path,
                        "text": text,
                        "label": -3
                    })
            self.data = all_data
        else:
            all_data = []
            for row in dataframe:
                all_data.append({
                    "image_path": row["image_path"],
                    "text": row["text"],
                    "label": row["label"]
                })
            self.data = all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
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
        
        return image, text, label

def train(model, train_loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for batch_idx, (images, text_inputs, targets) in enumerate(tqdm(train_loader)):
        images = images.to(device)
        targets = targets.to(device)

        input_ids = text_inputs['input_ids'].to(device)
        attention_mask = text_inputs['attention_mask'].to(device)

        logits = model(images, input_ids, attention_mask)
        loss = loss_fn(logits, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (batch_idx + 1) % 10 == 0:
            print(f"\nBatch {batch_idx + 1}/{len(train_loader)} - Loss: {total_loss/(batch_idx+1):.4f}", flush=True)
    return total_loss / len(train_loader)


def evaluate(model, dev_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, text_inputs, targets in dev_loader:
            images = images.to(device)
            targets = targets.to(device)
            input_ids = text_inputs['input_ids'].to(device)
            attention_mask = text_inputs['attention_mask'].to(device)
            logits = model(images, input_ids, attention_mask)

            probs = torch.sigmoid(logits).cpu()
            preds = (probs >= 0.5).long()
            all_preds.extend(preds.tolist())
            all_labels.extend(targets.cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    m_f1 = f1_score(all_labels, all_preds, average='macro')
    f1 = f1_score(all_labels, all_preds, pos_label=1, average='binary')  # positive_known=0 as in PU learning
    precision = precision_score(all_labels, all_preds, pos_label=1, average='binary')
    recall = recall_score(all_labels, all_preds, pos_label=1, average='binary')
    roc_auc = roc_auc_score(all_labels, all_preds)

    return acc, m_f1, f1, precision, recall, roc_auc


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


class PNULoss(nn.Module):
    def __init__(self, gamma=0.1, pos_unknown_value=0.67, neg_unknown_value=0.33, pi_p=0.5):
        super().__init__()
        self.gamma = gamma
        self.pos_unknown_value = pos_unknown_value
        self.neg_unknown_value = neg_unknown_value
        self.pi_p = pi_p
        self.positive_known = 1
        self.positive_agreed_unknown = -1
        self.negative_known = 0
        self.negative_agreed_unknown = -2
        self.disagreed_unknown = -3
        self.min_count = 1  # avoid div by zero

    def forward(self, inputs, targets):
        inputs = inputs.squeeze(-1) if inputs.dim() > 1 else inputs
        assert inputs.shape == targets.shape, "Input and target shapes must match"

        positive_mask = (targets == self.positive_known).float()
        positive_unknown_mask = (targets == self.positive_agreed_unknown).float()
        negative_mask = (targets == self.negative_known).float()
        negative_unknown_mask = (targets == self.negative_agreed_unknown).float()
        real_unknown_mask = (targets == self.disagreed_unknown).float()

        n_positive = torch.clamp(positive_mask.sum(), min=self.min_count)
        n_positive_unknown = torch.clamp(positive_unknown_mask.sum(), min=self.min_count)
        n_negative = torch.clamp(negative_mask.sum(), min=self.min_count)
        n_negative_unknown = torch.clamp(negative_unknown_mask.sum(), min=self.min_count)
        disagreed_unknown = torch.clamp(real_unknown_mask.sum(), min=self.min_count)
        
        loss_pos = F.binary_cross_entropy_with_logits(inputs, torch.ones_like(inputs), reduction='none')
        loss_pos_unk =  F.binary_cross_entropy_with_logits(inputs, torch.full_like(inputs, self.pos_unknown_value), reduction='none')
        loss_neg = F.binary_cross_entropy_with_logits(inputs, torch.zeros_like(inputs), reduction='none')
        loss_neg_unk =  F.binary_cross_entropy_with_logits(inputs, torch.full_like(inputs, self.neg_unknown_value), reduction='none')

        pu_positive_risk = self.pi_p * ( (loss_pos * positive_mask).sum() / n_positive + (loss_pos_unk * positive_unknown_mask).sum() / n_positive_unknown)/ 2
        pu_negative_estimate = (
            (loss_neg * real_unknown_mask).sum() / disagreed_unknown
            - self.pi_p *(  (loss_neg * positive_mask).sum() / n_positive + (loss_neg_unk * positive_unknown_mask).sum() / n_positive_unknown )/ 2 
        )
        if pu_negative_estimate < 0:
            pu_negative_estimate =  - 0.01 * pu_negative_estimate
        pu_risk = pu_positive_risk + pu_negative_estimate

        nu_negative_risk = self.pi_p * ((loss_neg * negative_mask).sum() / n_negative + (loss_neg_unk * negative_unknown_mask).sum() / n_negative_unknown)/ 2
        nu_positive_estimate = (
            (loss_pos * real_unknown_mask).sum() / disagreed_unknown
            - self.pi_p *(  (loss_pos * negative_mask).sum() / n_negative + (loss_pos_unk * negative_unknown_mask).sum() / n_negative_unknown )/ 2 
        )
        if nu_positive_estimate < 0:
            nu_positive_estimate =  - 0.01 * nu_positive_estimate
        nu_risk = nu_negative_risk + nu_positive_estimate

        positive_risk = ((loss_pos * positive_mask).sum() / n_positive + (loss_pos_unk * positive_unknown_mask).sum() / n_positive_unknown)/ 2
        negative_risk = ((loss_neg * negative_mask).sum() / n_negative + (loss_neg_unk * negative_unknown_mask).sum() / n_negative_unknown)/ 2
        if(self.gamma >= 0):
            return (1 - self.gamma) * (positive_risk + negative_risk) + self.gamma * pu_risk
        else:
            return (1 - (-self.gamma)) * (positive_risk + negative_risk) + (-self.gamma) * nu_risk

def main(args):
    set_seed(args.seed)
    # Load data
    root_folder = args.root_folder
    dataset_name = args.dataset_name
    pseudo_label_type = args.pseudo_label_type
    round_number = args.round_number
    known_number = args.known_number
    topk = args.topk
    # Initialize the model and optimizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    def collate_fn(batch):
        images, texts, labels = zip(*batch)
        images = torch.stack(images)
        text_inputs = tokenizer(list(texts), return_tensors="pt", padding='max_length', truncation=True, max_length=77)
        labels = torch.tensor(labels, dtype=torch.long)
        return images, text_inputs, labels

    clip_model = CLIPModel.from_pretrained(args.model_id)
    model = CLIPBinaryClassifier(clip_model).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = PNULoss(gamma=args.gamma, pos_unknown_value=args.pos_unknown_value, neg_unknown_value=args.neg_unknown_value, pi_p=args.pi_p)

    image_id_split_path =f"{root_folder}/datasets/{dataset_name}/image_id_split_train.json"
    with open(image_id_split_path, "r") as f:
        image_id_split = json.load(f)
    known = image_id_split[str(known_number)]["pnu_known"]["image_id"]
    known = [str(id) for id in known]



    if(round_number == 0):
        gamma = 0.0
        pos_unknown_value = 1.0
        neg_unknown_value = 0.0

    positive_agreed_unknown = []
    negative_agreed_unknown = []
    disagreed_unknown = []
    overall_best_macro_m_y1, macro_m_y1 = 0.0, 0.0
    for i in range( round_number + 1):
        if(i < round_number):
            macro_m_y1 = json.load(open(f"{root_folder}/checkpoints/{dataset_name}/round_{str(i)}_meta.json"))["best_epoch"]["macro_f1"]
        if(i == 0):
            overall_best_macro_m_y1 = macro_m_y1
            continue
        elif(i < round_number and macro_m_y1 >= overall_best_macro_m_y1):
            overall_best_macro_m_y1 = macro_m_y1
        elif(i < round_number):
            continue

        df = pd.read_csv(f"{root_folder}/scripts/meta_data/Classifier_MA_VLMs_output/{dataset_name}/{pseudo_label_type}_round_{str(i-1)}.csv", dtype={'image_id': str})
        positive_agreed_unknown_round = []
        negative_agreed_unknown_round = []
        disagreed_unknown_round = []
        for _, row in df.iterrows():
            image_id =  row["image_id"]
            if row["pseudo_label"] == 1:
                positive_agreed_unknown_round.append(image_id)
            elif row["pseudo_label"] == 0:
                negative_agreed_unknown_round.append(image_id)
            elif row["pseudo_label"] == -1:
                disagreed_unknown_round.append(image_id)
        total_len = len(positive_agreed_unknown_round) + len(negative_agreed_unknown_round) + len(disagreed_unknown_round)
        positive_agreed_unknown += positive_agreed_unknown_round[ : int(topk * len(positive_agreed_unknown_round) / total_len )]
        negative_agreed_unknown += negative_agreed_unknown_round[:int(topk * len(negative_agreed_unknown_round) / total_len )] 
        disagreed_unknown += disagreed_unknown_round[:int(topk * len(disagreed_unknown_round) / total_len )]

    print("The number of Known " + str(len(known)), flush=True)
    print("The number of Positive Agreed-Unknwon " + str(len(positive_agreed_unknown)), flush=True)
    print("The number of Negative Agreed-Unknwon  " + str(len(negative_agreed_unknown)), flush=True)
    print("The number of Unknown Agreed-Unknwon  " + str(len(disagreed_unknown)), flush=True)
    
    # Load train and dev datasets
    train_data = read_jsonl( f"{root_folder}/datasets/{dataset_name}/train.jsonl")
    dev_data = read_jsonl( f"{root_folder}/datasets/{dataset_name}/dev.jsonl")

    # Initialize dataset and dataloaders
    train_dataset = MAST_PNU_Dataset(dataframe=train_data, split="train", tokenizer=tokenizer, known=known, positive_agreed_unknown=positive_agreed_unknown, negative_agreed_unknown=negative_agreed_unknown, disagreed_unknown=disagreed_unknown)
    dev_dataset = MAST_PNU_Dataset(dataframe=dev_data, tokenizer=tokenizer, split="test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed)
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )

    best_macro_m_y1 = 0.0
    best_model_path = f"{root_folder}/checkpoints/{dataset_name}/round_{str(round_number)}_ckpt.pth"
    metadata_path = f"{root_folder}/checkpoints/{dataset_name}/round_{str(round_number)}_meta.json"
    
    metadata = {'best_model_path': best_model_path, 'best_epoch': {}}
    
    # Training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        loss = train(model, train_loader, optimizer, loss_fn, args.device)
        acc, m_f1, f1, precision, recall, roc_auc = evaluate(model, dev_loader, args.device)
        
        print(f"Epoch: {epoch}, Macro F1: {m_f1:.4f}", flush=True)
        metadata['epoch_' + str(epoch)] = {
            'accuracy': acc,
            'macro_f1': m_f1,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc,
        }

        # Save best model
        if m_f1 > best_macro_m_y1:
            best_macro_m_y1 = m_f1
            torch.save(model.state_dict(), best_model_path)
            metadata['best_epoch'] = {
                'accuracy': acc,
                'macro_f1': m_f1,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
            }
    
    print(f"Model saved with the best Macro-F1 score for eval set: {best_macro_m_y1:.4f}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    if best_macro_m_y1 >= overall_best_macro_m_y1:
        overall_metadata_path = f"{root_folder}/checkpoints/{dataset_name}/best_round_meta.json"
        with open(overall_metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

if __name__ == "__main__":
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

    args = parser.parse_args()
    main(args)

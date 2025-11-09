import pandas as pd
import csv, re
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def extract_label(text, positive_label):
    if positive_label in text:
        return 1
    else:
        return 0

def pseudo_label_func(classifier_pseudo_label, auditor_pseudo_label_step_0, auditor_pseudo_label_step_1, user_pseudo_label_step_0, user_pseudo_label_step_1, pseudo_label_type, inference_data_type):
    if pseudo_label_type == "classifier_ma_vlm":
        if auditor_pseudo_label_step_0 == user_pseudo_label_step_0:
            return auditor_pseudo_label_step_0
        elif classifier_pseudo_label == auditor_pseudo_label_step_1 and classifier_pseudo_label == user_pseudo_label_step_1:
            return classifier_pseudo_label
        else:
            return -1
    elif pseudo_label_type == "ma_vlms":
        if auditor_pseudo_label_step_1 == user_pseudo_label_step_1:
            return auditor_pseudo_label_step_1
        else:
            if inference_data_type == "test":
                return 0
            else:
                return -1
    elif pseudo_label_type == "classifier":
        return classifier_pseudo_label

def main(args):
    # Initialize parameters from args
    round_number = args.round_number
    inference_data_type = args.inference_data_type
    dataset_name = args.dataset_name
    known_number = args.known_number
    pseudo_label_type = args.pseudo_label_type
    
    if dataset_name in ["FHM", "HSOL"]:
        positive_label, negative_label = "Hateful", "Normal"
    elif dataset_name in ["MAMI"]:
        positive_label, negative_label = "Misogyny", "Normal"
    elif dataset_name in ["Sent140"]:
        positive_label, negative_label = "Positive", "Negative"

    root_folder = args.root_folder
    output_path = f"{root_folder}/scripts/meta_data/Classifier_MA_VLMs_output/{dataset_name}/{pseudo_label_type}_round_{round_number}.csv"

    classifier_pseudo_label_path = f"{root_folder}/scripts/meta_data/Classifier_output/{dataset_name}/round_{round_number}_train.csv"
    classifier_pseudo_labels = pd.read_csv(classifier_pseudo_label_path, dtype={'image_id': str})
    combine_pseudo_label = {}
    for _, row in classifier_pseudo_labels.iterrows():
        image_id =  row["image_id"]
        gt_label = row["gt_label"]
        pseudo_label = row["pseudo_label"]
        pseudo_label_prob  = row["pseudo_label_prob"]
        combine_pseudo_label[image_id] = {
            "gt_label": int(gt_label),
            "classifier_pseudo_label": int(pseudo_label),
            "classifier_pseudo_label_prob": pseudo_label_prob
        }

    ma_vlms_pseudo_label_path = f"{root_folder}/scripts/meta_data/MA_VLMs_output/{dataset_name}/all.csv"
    ma_vlms_pseudo_labels = pd.read_csv(ma_vlms_pseudo_label_path, dtype={'image_id': str})

    for _, row in ma_vlms_pseudo_labels.iterrows():
        image_id =  row["image_id"]
        if image_id in combine_pseudo_label:
            classifier_pseudo_label = combine_pseudo_label[image_id]["classifier_pseudo_label"]
        else:
            continue
        classifier_pseudo_label = "_for_" + str(classifier_pseudo_label)
        auditor_pseudo_label_step_1 = extract_label(row["auditor_step_1_judgement" + classifier_pseudo_label], positive_label) if row["auditor_step_1_judgement" + classifier_pseudo_label] != "No" else extract_label(row["auditor_step_0_judgement" + classifier_pseudo_label], positive_label)
        user_pseudo_label_step_1 = extract_label(row["user_step_1_judgement" + classifier_pseudo_label], positive_label) if row["user_step_1_judgement" + classifier_pseudo_label] != "No" else extract_label(row["user_step_0_judgement" + classifier_pseudo_label], positive_label)
        combine_pseudo_label[image_id]["auditor_pseudo_label_step_1"] = auditor_pseudo_label_step_1
        combine_pseudo_label[image_id]["user_pseudo_label_step_1"] = user_pseudo_label_step_1
        combine_pseudo_label[image_id]["auditor_pseudo_label_step_0"] = extract_label(row["auditor_step_0_judgement" + classifier_pseudo_label], positive_label)
        combine_pseudo_label[image_id]["user_pseudo_label_step_0"] = extract_label(row["user_step_0_judgement" + classifier_pseudo_label], positive_label)

    output = []
    for image_id in combine_pseudo_label:
        if "auditor_pseudo_label_step_1" in combine_pseudo_label[image_id] and "classifier_pseudo_label" in combine_pseudo_label[image_id]:
            classifier_pseudo_label = combine_pseudo_label[image_id]["classifier_pseudo_label"]
            auditor_pseudo_label_step_1 = combine_pseudo_label[image_id]["auditor_pseudo_label_step_1"]
            user_pseudo_label_step_1 = combine_pseudo_label[image_id]["user_pseudo_label_step_1"]
            auditor_pseudo_label_step_0 = combine_pseudo_label[image_id]["auditor_pseudo_label_step_0"]
            user_pseudo_label_step_0 = combine_pseudo_label[image_id]["user_pseudo_label_step_0"]

            fused_label = pseudo_label_func(classifier_pseudo_label, auditor_pseudo_label_step_0, auditor_pseudo_label_step_1, user_pseudo_label_step_0, user_pseudo_label_step_1, pseudo_label_type, inference_data_type)

            output.append([image_id, combine_pseudo_label[image_id]["gt_label"], fused_label, combine_pseudo_label[image_id]["classifier_pseudo_label_prob"]])

    output = sorted(output, key=lambda x: abs(x[3] - 0.5), reverse=True)
    output = [["image_id", "gt_label", "pseudo_label", "classifier_prob"]] + output

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output)

    df = pd.read_csv(output_path, dtype={'image_id': str})
    if inference_data_type == "test":
        df = df[df['pseudo_label'] != -1]
    else:
        pos_unknown_image_ids, neg_unknown_image_ids, real_unknown_image_ids = [], [], []
        for _, row in df.iterrows():
            label = row["pseudo_label"]
            image_id = row["image_id"]
            if label == 1:
                pos_unknown_image_ids.append(image_id)
            elif label == 0:
                neg_unknown_image_ids.append(image_id)
            elif label == -1:
                real_unknown_image_ids.append(image_id)
        total_len = len(pos_unknown_image_ids) + len(neg_unknown_image_ids) + len(real_unknown_image_ids)
        next_round_image_ids = (
            pos_unknown_image_ids[:int(args.topk * len(pos_unknown_image_ids) / total_len)] +
            neg_unknown_image_ids[:int(args.topk * len(neg_unknown_image_ids) / total_len)]
        )
        df = df[df['image_id'].isin(next_round_image_ids)]

    y_true = df['gt_label'].values
    y_pred = df['pseudo_label'].values
    acc = accuracy_score(y_true, y_pred)
    m_f1 = f1_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, pos_label=1)
    recall = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)
    roc_auc = roc_auc_score(y_true, y_pred)

    print(f"  Round {round_number} topk pseudo labeled unknown evalutaion: ")
    print(f"  Accuracy        = {acc:.4f}", flush=True)
    print(f"  Macro F1        = {m_f1:.4f}", flush=True)
    print(f"  F1 (pos)        = {f1:.4f}", flush=True)
    print(f"  Precision (pos) = {precision:.4f}", flush=True)
    print(f"  Recall (pos)    = {recall:.4f}", flush=True)
    print(f"  Roc Auc         = {roc_auc:.4f}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pseudo-labeling and evaluation")

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
   

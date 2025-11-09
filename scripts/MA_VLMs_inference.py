import json
import os
import csv
import torch
import re
import argparse
from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

def load_model_and_processor(model_id):
    config = Qwen2_5_VLConfig.from_pretrained(model_id)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    processor = Qwen2_5_VLProcessor.from_pretrained(model_id)
    return model, processor


def setup_dataset(dataset_name, root_folder):
    dataset_path = f"{root_folder}/datasets/{dataset_name}/all.jsonl"
    data_lists = []
    with open(dataset_path, "r") as f:
        for line in f:
            data_lists.append(json.loads(line.strip()))
    return data_lists

def get_task_instructions(dataset_name):
    if dataset_name in ["FHM", "HSOL"]:
        positive_label, negative_label = "Hateful",  "Normal"
        task_instruction = "Hate speech is content that incites discrimination, hostility, or violence against individuals or groups based on race, gender, sexual orientation, religion, or other protected characteristics."
        auditor_identity_instruction = "You are an expert auditor responsible for enforcing rigorous standards in detecting hateful speech, ensuring that no hateful content is overlooked."
        auditor_identity_instruction_short = "an expert auditor responsible for enforcing rigorous standards in detecting hateful speech"
        user_identity_instruction = "You are a platform user responsible for ensuring the integrity and fairness of hateful speech detection, ensuring the freedom of speech."
        user_identity_instruction_short = "a platform user responsible for ensuring the integrity and fairness of hateful speech detection"
    elif dataset_name in ["MAMI"]:
        positive_label, negative_label = "Misogyny",  "Normal"
        task_instruction = "Misogyny speech attitude refers to any kind of disrespect, dislike, or negative view toward women, often expressed through harmful comments, unfair treatment, or biased behavior based on gender."
        auditor_identity_instruction = "You are an expert auditor responsible for enforcing rigorous standards in detecting misogyny speech, ensuring that no misogyny content is overlooked. "
        auditor_identity_instruction_short = "an expert auditor responsible for enforcing rigorous standards in detecting misogyny speech"
        user_identity_instruction = "You are a platform user responsible for ensuring the integrity and fairness of misogyny speech detection, ensuring the freedom of speech."
        user_identity_instruction_short = "a platform user responsible for ensuring the integrity and fairness of misogyny speech detection"
    elif dataset_name in ["Sent140"]:
        positive_label, negative_label = "Negative",  "Positive"
        task_instruction = "Positive sentiment refers to expressions that convey feelings of happiness, satisfaction, approval, or goodwill. Negative sentiment refers to expressions that convey feelings of displeasure, anger, disappointment, or criticism."
        auditor_identity_instruction = "You are an expert auditor responsible for upholding strict standards in identifying negative emotional content, ensuring that all negative content is detected."
        auditor_identity_instruction_short = "an expert auditor responsible for upholding strict standards in identifying negative emotional content"
        user_identity_instruction = "You are a platform user responsible for identifying positive emotional content, ensuring that all positive content is accurately detected."
        user_identity_instruction_short = "a platform user responsible for identifying positive emotional content"
    return positive_label, negative_label, task_instruction, auditor_identity_instruction, auditor_identity_instruction_short, user_identity_instruction, user_identity_instruction_short

def format_data(root_folder, dataset_name, sample, agent_identity, discuss_step, clip_pred_label = "", auditor_step_0_output = "", user_step_0_output = ""):
    positive_label, negative_label, task_instruction, auditor_identity_instruction, auditor_identity_instruction_short, user_identity_instruction, user_identity_instruction_short = get_task_instructions(dataset_name)
    if(agent_identity == "auditor"):
        label = positive_label
        my_side_identity_instruction = auditor_identity_instruction
        my_side_step_0_output = auditor_step_0_output
        other_side_indentity = "platform user"
        other_side_identity_instruction_stort = user_identity_instruction_short
        other_side_step_0_output = user_step_0_output
    elif(agent_identity == "user"):
        label = negative_label
        my_side_identity_instruction = user_identity_instruction
        my_side_step_0_output = user_step_0_output
        other_side_indentity = "expert auditor"
        other_side_identity_instruction_stort = auditor_identity_instruction_short
        other_side_step_0_output = auditor_step_0_output

    if(discuss_step == 0):
        question =  f"""
        {my_side_identity_instruction}
        {task_instruction}
        Your task is to evaluate the given text and image, providing an explanation for why the content is considered {label}.

        Given the following:
        - Text: "{sample["text"]}"
        - Image: [The given image]

        Please adhere to the <label>{label}</label> label and provide a short reasoning.
        Output Format:
        "The input should be classified as <label>{label}</label>. Reason: [Your short reasoning]."
        """
    elif(discuss_step == 1):
        question =  f"""
        {my_side_identity_instruction} 
        {task_instruction}
        Your task is to evaluate a given text and image combination and verify the classification made by a initial auditor.

        Given the following:
        - Text: "{sample["text"]}"
        - Image: [The given image]
        - Initial Classification: <label>{clip_pred_label}</label> 

        Your initial classification is {my_side_step_0_output}. 
        However, {other_side_identity_instruction_stort} has opposite classification: {other_side_step_0_output}.
        Do you agree with the initial auditor's and {other_side_indentity}'s classification? Please provide your revised classification and a short response to {other_side_indentity}.

        Output Format:
        "The input should be classified as <label>{positive_label}</label> or <label>{negative_label}</label>. Reason: [Your short response]."
        """
    image_path = sample["image_path"]
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": ""}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": os.path.join(root_folder, image_path) if os.path.exists(os.path.join(root_folder, image_path)) else os.path.join(root_folder, "datasets", dataset_name, "/images/demo.png"),
                },
                {
                    "type": "text",
                    "text": question,
                },
            ],
        }
    ]
    
def inference_vlms(sample, model, processor , device="cuda:0"):
    text_input = processor.apply_chat_template(
        sample, tokenize=False, add_generation_prompt=True  # Use the sample without the system message
    )
    image_inputs, _ = process_vision_info(sample)

    model_inputs = processor(
        text=[text_input],
        images=image_inputs,
        return_tensors="pt",
    ).to(
        device
    ) 
    generated_ids = model.generate(**model_inputs, max_new_tokens=1024)

    trimmed_generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(model_inputs.input_ids, generated_ids)]

    output_text = processor.batch_decode(
        trimmed_generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0] 

def extract_label_and_reason(text):
    pattern = r'The input should be classified as\s*<label>([\w\s]+)</label>\.\s*Reason:\s*(.*)'
    match = re.search(pattern, text)
    if match:
        label = match.group(1) 
        reason = match.group(2) 
        return label, reason
    else:
        pattern = r'The input should not be classified as\s*<label>([\w\s]+)</label>\.\s*Reason:\s*(.*)'
        match = re.search(pattern, text)
        if match:
            label = match.group(1)  
            reason = match.group(2)  
            if("Hateful" in label):
                label = "Normal"
            else:
                label = "Hateful"
            return label, reason
        else:
            return "Normal", "No"  

def inference_data(row, dataset_name, model, processor, root_folder, positive_label, negative_label):
    
    output_row = [str(row["image_id"]), row["label"]]
    if(dataset_name == "Sent140"):
        clip_pred_labels = [positive_label, negative_label]
    else:
        clip_pred_labels = [negative_label, positive_label]
    for clip_pred_label in clip_pred_labels:
        auditor_step_0_output = inference_vlms(format_data(root_folder, dataset_name, row, "auditor", 0), model, processor)
        user_step_0_output = inference_vlms(format_data(root_folder, dataset_name, row, "user", 0), model, processor)
        auditor_step_0_judgement, auditor_step_0_reason = extract_label_and_reason(auditor_step_0_output)
        user_step_0_judgement, user_step_0_reason = extract_label_and_reason(user_step_0_output)

        if auditor_step_0_judgement != user_step_0_judgement:
            auditor_step_1_output = inference_vlms(format_data(root_folder, dataset_name, row, "auditor", 1, clip_pred_label, auditor_step_0_output, user_step_0_output), model, processor)
            user_step_1_output = inference_vlms(format_data(root_folder, dataset_name, row, "user", 1, clip_pred_label, auditor_step_0_output, user_step_0_output), model, processor)
            auditor_step_1_judgement, auditor_step_1_reason = extract_label_and_reason(auditor_step_1_output)
            user_step_1_judgement, user_step_1_reason = extract_label_and_reason(user_step_1_output)
        else:
            auditor_step_1_output, auditor_step_1_judgement, auditor_step_1_reason = "No", "No", "No"
            user_step_1_output, user_step_1_judgement, user_step_1_reason = "No", "No", "No"
        output_row += [auditor_step_0_output, auditor_step_0_judgement, auditor_step_0_reason,
        user_step_0_output, user_step_0_judgement, user_step_0_reason, auditor_step_1_output,
        auditor_step_1_judgement, auditor_step_1_reason, user_step_1_output, user_step_1_judgement, user_step_1_reason]
    
    return output_row

def main():
    args = parse_args()

    model, processor = load_model_and_processor(args.model_id)
    data_lists = setup_dataset(args.dataset_name, args.root_folder)
    data_lists = data_lists[:3]
    positive_label, negative_label, task_instruction, auditor_identity_instruction, auditor_identity_instruction_short, user_identity_instruction, user_identity_instruction_short = get_task_instructions(args.dataset_name)
    
    output = [["image_id", "gt_label", "auditor_step_0_output_for_0", "auditor_step_0_judgement_for_0", "auditor_step_0_reason_for_0", 
               "user_step_0_output_for_0", "user_step_0_judgement_for_0", "user_step_0_reason_for_0", "auditor_step_1_output_for_0", 
               "auditor_step_1_judgement_for_0", "auditor_step_1_reason_for_0", "user_step_1_output_for_0", "user_step_1_judgement_for_0", 
               "user_step_1_reason_for_0"]]

    for i, row in enumerate(data_lists):
        output_row = inference_data(row, args.dataset_name, model, processor, args.root_folder, positive_label, negative_label)
        output.append(output_row)

    with open(f"{args.root_folder}/scripts/meta_data/MA_VLMs_output/{args.dataset_name}/all.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(output)

def parse_args():
    parser = argparse.ArgumentParser(description="Multimodal Content Classification")
    parser.add_argument('--root_folder', type=str, required=True, help='Path to the root folder for the model')
    parser.add_argument('--dataset_name', type=str, default="FHM", choices=['FHM', 'MAMI', 'HSOL', 'Sent140'], help='Name of the dataset (e.g., FHM, MAMI, etc.)')
    parser.add_argument('--model_id', type=str, default="Qwen/Qwen2.5-VL-72B-Instruct", help='Pretrained model identifier (e.g., Qwen2.5-VL-7B-Instruct)')
    return parser.parse_args()

if __name__ == "__main__":
    main()

#Models.py
#Human Scar Detection (Industry Application Project)
#COSC 5437 Neural Networking
#Fangze Zhou & Serban Voinea Gabreanu
#This script contains all functions related to model evaluation and graph generation.

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, random_split, Subset
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import timm
from collections import defaultdict
import re
import random

#Note that the Models.py script has to be in the same folder as this script.
from Models import create_modern_cnn

#Evaluation method that loads the model, and computes the peformance metrics.
def get_validation_metrics(model_path, val_loader, device, num_classes, class_names, default_variant):
    try:
        #Loads checkpoints.
        checkpoint = torch.load(model_path, map_location=device)
        variant = checkpoint.get('variant', default_variant)
        use_transfer_learning = checkpoint.get('use_transfer_learning', False)

        #Recreates the model based on checkpoint data.
        if use_transfer_learning:
            print(f"Loading timm model 'convnext_{variant}' for evaluation.")
            model = timm.create_model(f'convnext_{variant}', pretrained=False, num_classes=num_classes)
        else:
            print(f"Loading custom model 'ModernConvNeXtV2-{variant}' for evaluation.")
            model = create_modern_cnn(num_classes=num_classes, variant=variant)

        #Loads the weights into the model's structure.
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()

        all_preds, all_labels = [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        report_dict = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0, output_dict=True)
        
        overall_metrics = {
            "model_name": os.path.basename(model_path).replace('_deployable.pth', ''),
            "accuracy": report_dict['accuracy'],
            "precision": report_dict['weighted avg']['precision'],
            "recall": report_dict['weighted avg']['recall'],
            "f1": report_dict['weighted avg']['f1-score']
        }
        
        return overall_metrics, report_dict

    except Exception as e:
        print(f"Error evaluating model {os.path.basename(model_path)}: {e}")
        return None, None

#Creates a bar chart and saves it for single model's per class performance. ()
def plot_per_model_graph(model_name, report_dict, save_dir):
    class_names = [label for label in report_dict.keys() if label not in ['accuracy', 'macro avg', 'weighted avg']]
    
    if not class_names:
        print(f"No per-class metrics to plot for model {model_name}.")
        return

    all_precision = [report_dict[label]['precision'] for label in class_names]
    all_recall = [report_dict[label]['recall'] for label in class_names]
    all_f1_score = [report_dict[label]['f1-score'] for label in class_names]

    #Defines how many classes to show per graph and calculate how many graphs are needed. (Good since there's a lot of classes for the dataset used in this project).
    classes_per_graph = 5
    num_classes = len(class_names)
    num_graphs = (num_classes + classes_per_graph - 1) // classes_per_graph

    #Loops to create one graph for each chunk of classes.
    for i in range(num_graphs):
        #Defines the start and end index in the current chunk of data.
        start_index = i * classes_per_graph
        end_index = start_index + classes_per_graph

        #Slices the data for the current graph
        chunk_class_names = class_names[start_index:end_index]
        chunk_precision = all_precision[start_index:end_index]
        chunk_recall = all_recall[start_index:end_index]
        chunk_f1_score = all_f1_score[start_index:end_index]

        x = np.arange(len(chunk_class_names))
        width = 0.25

        fig, ax = plt.subplots(figsize=(16, 9))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        ax.bar(x - width, chunk_precision, width, label='Precision', color='#0070C0')
        ax.bar(x, chunk_recall, width, label='Recall', color='#B4C7E7')
        ax.bar(x + width, chunk_f1_score, width, label='F1-Score', color='#C00000', hatch='..', edgecolor='white')

        for spine in ax.spines.values(): spine.set_color('white')
        ax.tick_params(axis='x', colors='white', labelsize=12, labelrotation=20)
        ax.tick_params(axis='y', colors='white', labelsize=12)
        ax.grid(color='white', linestyle='--', linewidth=0.5, axis='y', alpha=0.5)

        ax.set_ylabel('Score', fontsize=16, color='white')
        #Updates the title to show which part it is
        ax.set_title(f'Per-Class Performance: {model_name} (Part {i+1}/{num_graphs})', fontsize=20, color='white', pad=20)
        ax.set_xticks(x, chunk_class_names)
        ax.set_ylim(0, 1.1)
        ax.legend(facecolor='black', edgecolor='white', fontsize=14, labelcolor='white')

        fig.tight_layout()
        #Updates the file path to be unique for each part
        graph_path = os.path.join(save_dir, f"{model_name}_Part_{i+1}.png")
        plt.savefig(graph_path, facecolor='black', bbox_inches='tight')
        plt.close(fig)

#Compares the performance of all models.
def plot_comparison_graph(all_metrics, save_dir):
    all_metrics.sort(key=lambda x: x['f1'], reverse=True)
    
    #Splits into chunks to attempt to keep the graphs readable.
    models_per_graph = 5
    metric_chunks = [all_metrics[i:i + models_per_graph] for i in range(0, len(all_metrics), models_per_graph)]

    for i, chunk in enumerate(metric_chunks):
        model_names = [m['model_name'] for m in chunk]
        scores = {
            'Accuracy': [m['accuracy'] for m in chunk],
            'Precision': [m['precision'] for m in chunk],
            'Recall': [m['recall'] for m in chunk],
            'F1-Score': [m['f1'] for m in chunk],
        }
        
        x = np.arange(len(model_names))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(20, 12))
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        
        multiplier = 0
        for attribute, measurement in scores.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute)
            ax.bar_label(rects, padding=3, fmt='%.3f', color='white', fontsize=10)
            multiplier += 1

        for spine in ax.spines.values(): spine.set_color('white')
        ax.tick_params(axis='x', colors='white', labelsize=12, labelrotation=20)
        ax.tick_params(axis='y', colors='white', labelsize=16)
        ax.grid(color='white', linestyle='--', linewidth=0.5, axis='y', alpha=0.5)

        ax.set_ylabel('Score', fontsize=21, color='white')
        ax.set_title(f'Model Performance Comparison (Top {i*5+1}-{(i+1)*5})', fontsize=30, color='white', pad=20)
        ax.set_xticks(x + width * 1.5, model_names)
        ax.set_ylim(0, 1.15)
        ax.legend(loc='upper left', ncols=4, facecolor='black', edgecolor='white', fontsize=16, labelcolor='white')
        
        fig.tight_layout()
        graph_path = os.path.join(save_dir, f"Comparison_Part_{i+1}.png")
        plt.savefig(graph_path, facecolor='black', bbox_inches='tight')
        print(f"Comparison graph saved to: {graph_path}")
        plt.close(fig)

#Stars the graphing module.
def generate_validation_graphs(device, processed_dir, deploy_dir, graphs_dir, per_model_graphs_dir, default_variant, val_split, batch_size):
    print("\n" + "#" * 22 + " Generating Validation Graphs " + "#" * 22)
    os.makedirs(graphs_dir, exist_ok=True)
    os.makedirs(per_model_graphs_dir, exist_ok=True)

    deployed_models = sorted([f for f in os.listdir(deploy_dir) if f.endswith('.pth') or f.endswith('.pth.tar')])
    if not deployed_models:
        print("No deployed models found to evaluate."); return

    if not os.path.isdir(processed_dir):
        print(f"Error: Processed dataset not found at '{processed_dir}'."); return
    
    with open(os.path.join(processed_dir, 'class_mapping.json'), 'r') as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)
    class_names = [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_dataset = ImageFolder(processed_dir, transform=transform)
    
    print("Grouping images to create a leakage-free validation set for graphing...")
    image_groups = defaultdict(list)
    base_name_pattern = re.compile(r"(.+?)_original|(.+?)_aug_\d+")
    for i, (path, _) in enumerate(full_dataset.imgs):
        filename = os.path.splitext(os.path.basename(path))[0]
        match = base_name_pattern.match(filename)
        base_name = match.group(1) if match.group(1) else match.group(2)
        if base_name: image_groups[base_name].append(i)

    unique_base_names = list(image_groups.keys())
    random.seed(42) 
    random.shuffle(unique_base_names)
    val_split_idx = int(len(unique_base_names) * val_split)
    val_groups = unique_base_names[:val_split_idx]
    val_indices = [idx for group in val_groups for idx in image_groups[group]]
    val_dataset = Subset(full_dataset, val_indices)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=min(os.cpu_count(), 8), pin_memory=True)
    print(f"Prepared leakage-free validation set with {len(val_dataset)} images.")

    all_metrics = []
    for model_file in tqdm(deployed_models, desc="Evaluating models", unit="model"):
        model_path = os.path.join(deploy_dir, model_file)
        overall_metrics, report_dict = get_validation_metrics(model_path, val_loader, device, num_classes, class_names, default_variant)
        
        if overall_metrics and report_dict:
            all_metrics.append(overall_metrics)
            plot_per_model_graph(overall_metrics['model_name'], report_dict, per_model_graphs_dir)

    if not all_metrics:
        print("Evaluation failed for all models. No graphs will be generated."); return
        
    print(f"\nSuccessfully validated {len(all_metrics)} models. Individual graphs saved to '{per_model_graphs_dir}'.")
    
    plot_comparison_graph(all_metrics, graphs_dir)
    print("Overall comparison graphs have been generated.")
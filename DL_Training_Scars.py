##Human Scar Detection (Industry Application Project)
#COSC 5437 Neural Networking
#Fangze Zhou & Serban Voinea Gabreanu
#DL_Training_Scars.py
#
#Uses this dataset: https://github.com/mattgroh/fitzpatrick17k, and Transfer Learning.
#
#Description: A full-featured script to preprocess, train, and manage the ModernConvNeXtV2
#model, incorporating a state-of-the-art transfer learning pipeline.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
from sklearn.metrics import classification_report
from PIL import Image, ImageFile
from tqdm import tqdm
import random
from collections import defaultdict
import re
import numpy as np
import os
import sys
import shutil
import json
from datetime import datetime
import multiprocessing
from functools import partial
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True
#Note the models and validationgraphing.py files should be in the same folder as this script.
from Models import create_modern_cnn, _VARIANTS
from ValidationGraphing import generate_validation_graphs
try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    BASE_DIR = os.getcwd()

RAW_DATASET_DIR = os.path.join(BASE_DIR, 'RawDataset')
PROCESSED_DATASET_DIR = os.path.join(BASE_DIR, 'ProcessedDataset')
CHECKPOINT_DIR = os.path.join(BASE_DIR, 'Checkpoints')
DEPLOY_DIR = os.path.join(BASE_DIR, 'DeployModel')
GRAPHS_DIR = os.path.join(BASE_DIR, 'ValidationGraphs')
PER_MODEL_GRAPHS_DIR = os.path.join(BASE_DIR, 'PerModelGraphs')

#Model & Training Hyperparameters. 
FINETUNE_LR = 5e-6
HEAD_LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 25
WARMUP_EPOCHS = 5
VALIDATION_SPLIT = 0.2
DEFAULT_VARIANT = "large"
DROP_PATH_RATE = 0.3
WEIGHT_DECAY = 1e-4
SmallClassBuff = 0.3

#Helps with preprocessing the data to not make it too large.
MAX_PROCESSED_DATASET_SIZE_GB = 30
AVG_PROCESSED_IMAGE_KB = 35


#Processes and saves the single image
def process_and_save_single_image(image_info, class_names, aug_plan, proc_dir, resize_transform, aug_transform):
    img_path, label_idx = image_info
    try:
        class_name = class_names[label_idx]
        augmentations_per_image = aug_plan[class_name]
        target_class_dir = os.path.join(proc_dir, class_name)
        os.makedirs(target_class_dir, exist_ok=True)
        original_image = Image.open(img_path).convert("RGB")
        processed_original = resize_transform(original_image)
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        original_filepath = os.path.join(target_class_dir, f"{base_filename}_original.jpg")
        processed_original.save(original_filepath, "JPEG", quality=95)
        for i in range(augmentations_per_image):
            augmented_image = aug_transform(original_image)
            aug_filepath = os.path.join(target_class_dir, f"{base_filename}_aug_{i+1}.jpg")
            augmented_image.save(aug_filepath, "JPEG", quality=95)
    except (IOError, OSError):
        pass
    return 1

#GPU acceleration code. If no GPU then runs on CPU (not recommended).

def get_device():
    if torch.cuda.is_available():
        print("CUDA compatible NVIDIA GPU found. Using GPU for acceleration.")
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("MPS compatible Apple Silicon GPU found. Using GPU for acceleration.")
        return torch.device("mps")
    print("No compatible GPU found. Defaulting to CPU (may be slow).")
    return torch.device("cpu")

def save_checkpoint(state, session_name, is_best, is_interrupt=False):
    session_dir = os.path.join(CHECKPOINT_DIR, session_name)
    os.makedirs(session_dir, exist_ok=True)
    filename = 'interrupt.pth.tar' if is_interrupt else 'checkpoint.pth.tar'
    filepath = os.path.join(session_dir, filename)
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(session_dir, 'model_best.pth.tar'))
        print(">> New best model saved!")

def preprocess_and_save_images():
    print("\n" + "#" * 22 + " Intelligent Data Augmentation " + "#" * 22)
    if not os.path.isdir(RAW_DATASET_DIR) or not os.listdir(RAW_DATASET_DIR):
        print(f"Error: Raw dataset directory is empty or does not exist.")
        print(f"Please place your raw images in: {RAW_DATASET_DIR}/class_name/image.jpg")
        return

    raw_dataset = ImageFolder(RAW_DATASET_DIR)
    class_counts = {cls: count for cls, count in zip(raw_dataset.classes, np.bincount(raw_dataset.targets))}
    if not class_counts:
        print("Error: No images found in the raw dataset directory.")
        return
    max_images = max(class_counts.values())

    while True:
        try:
            overall_multiplier = int(input("Enter overall augmentation multiplier (e.g., 2): "))
            image_size = int(input("Target image size? (e.g., 224): "))
        except ValueError:
            print("Invalid input. Please enter integers.")
            continue

        final_augmentation_plan = {}
        total_final_image_count = 0
        print("\n" + "#"*20 + " PROPOSED AUGMENTATION PLAN " + "#"*20)
        doubled_target_count = max_images * 2
        print(f"Equalization Target: {max_images} | Doubled Target: {doubled_target_count} | Final Multiplier: {overall_multiplier}x")

        for cls, count in class_counts.items():
            final_augs_per_image = 0
            if count > 0:
                images_to_generate = max(0, doubled_target_count - count)
                base_augs_needed_per_image = int(np.ceil(images_to_generate / count))
                #Applies the SmallClassBuff to the scaled augmentations
                final_augs_per_image = int(base_augs_needed_per_image * overall_multiplier * SmallClassBuff)
            final_augmentation_plan[cls] = final_augs_per_image
            total_final_image_count += count * (1 + final_augs_per_image)
            print(f"  - Class '{cls}' ({count} images): Will create {final_augs_per_image} new variations per image.")

        estimated_gb = (total_final_image_count * AVG_PROCESSED_IMAGE_KB * 1024) / (1024**3)
        print("\n" + "#"*20 + " ESTIMATE & CONFIRMATION " + "#"*20)
        print(f"This plan will generate ~{total_final_image_count:,} images, estimated size: ~{estimated_gb:.2f} GB")

        proceed = False
        if estimated_gb > MAX_PROCESSED_DATASET_SIZE_GB:
            print(f"\n*** WARNING: Exceeds {MAX_PROCESSED_DATASET_SIZE_GB} GB safety limit! ***")
            if input("Are you sure? Type 'YES' to override: ") == 'YES':
                proceed = True
        elif input("\nProceed with this plan? (y/n): ").lower() == 'y':
            proceed = True

        if proceed:
            break
        else:
            print("\nOperation cancelled. Let's try again.\n")

    print("\n" + "#"*20 + " PROCESSING IMAGES (IN PARALLEL) " + "#"*20)
    if os.path.exists(PROCESSED_DATASET_DIR):
        shutil.rmtree(PROCESSED_DATASET_DIR)
    os.makedirs(PROCESSED_DATASET_DIR)
    with open(os.path.join(PROCESSED_DATASET_DIR, 'class_mapping.json'), 'w') as f:
        json.dump(raw_dataset.class_to_idx, f)

    resize_and_crop_transform = T.Compose([T.Resize(image_size), T.CenterCrop(image_size)])
    augmentation_transform = T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=45),
        T.ColorJitter(brightness=0.3, contrast=0.5, saturation=0.5, hue=0.2),
        T.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
        T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    ])

    image_list = raw_dataset.imgs
    num_processes = os.cpu_count()
    print(f"Starting parallel processing with {num_processes} workers...")
    worker_func = partial(
        process_and_save_single_image,
        class_names=raw_dataset.classes,
        aug_plan=final_augmentation_plan,
        proc_dir=PROCESSED_DATASET_DIR,
        resize_transform=resize_and_crop_transform,
        aug_transform=augmentation_transform
    )
    with multiprocessing.Pool(processes=num_processes) as pool:
        with tqdm(total=len(image_list), desc="Processing source images") as pbar:
            for _ in pool.imap_unordered(worker_func, image_list):
                pbar.update()

    final_count = sum(len(files) for _, _, files in os.walk(PROCESSED_DATASET_DIR))
    final_size_gb = sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, files in os.walk(PROCESSED_DATASET_DIR)
        for name in files
    ) / (1024**3)
    print(f"\nPreprocessing complete! Final dataset: {final_count} images | Actual size: {final_size_gb:.2f} GB")



#This method is responsible for starting and continuing the training
#process for the DL model.
def train_model(device, session_name=None, start_epoch=0):
    from torch.optim.lr_scheduler import OneCycleLR # Import inside function

    print("\n" + "#" * 22 + " Model Training " + "#" * 22)
    if not os.path.isdir(PROCESSED_DATASET_DIR) or not os.listdir(PROCESSED_DATASET_DIR):
        print(f"Error: Processed dataset not found. Run option '1' first."); return

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    dataset = ImageFolder(PROCESSED_DATASET_DIR, transform=transform)
    num_classes, class_names = len(dataset.classes), dataset.classes
    print(f"Loaded dataset: {len(dataset)} images, {num_classes} classes.")

    print("Grouping images by original source to prevent data leakage...")
    image_groups = defaultdict(list)
    base_name_pattern = re.compile(r"(.+?)_original|(.+?)_aug_\d+")

    for i, (path, _) in enumerate(dataset.imgs):
        filename = os.path.splitext(os.path.basename(path))[0]
        match = base_name_pattern.match(filename)
        base_name = match.group(1) if match.group(1) else match.group(2)
        if base_name:
            image_groups[base_name].append(i)

    unique_base_names = list(image_groups.keys())
    random.seed(42)
    random.shuffle(unique_base_names)

    val_split_idx = int(len(unique_base_names) * VALIDATION_SPLIT)
    val_groups = unique_base_names[:val_split_idx]
    train_groups = unique_base_names[val_split_idx:]

    train_indices = [idx for group in train_groups for idx in image_groups[group]]
    val_indices = [idx for group in val_groups for idx in image_groups[group]]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)

    print(f"Leakage-free split created: {len(train_dataset)} training images, {len(val_dataset)} validation images.")

    num_workers = min(os.cpu_count(), 8)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=True)

    model, optimizer, scheduler, best_f1, variant_choice, use_transfer_learning = None, None, None, 0.0, "", False
    
    if session_name:
        print(f"Attempting to resume training for session: '{session_name}'")
        checkpoint_path = os.path.join(CHECKPOINT_DIR, session_name, 'interrupt.pth.tar')
        if not os.path.exists(checkpoint_path): checkpoint_path = os.path.join(CHECKPOINT_DIR, session_name, 'checkpoint.pth.tar')
        if not os.path.exists(checkpoint_path): print(f"Error: No checkpoint found for session '{session_name}'."); return
        
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch, best_f1 = checkpoint['epoch'], checkpoint['best_f1']
        variant_choice, use_transfer_learning = checkpoint['variant'], checkpoint['use_transfer_learning']

        if use_transfer_learning: model = timm.create_model(f'convnext_{variant_choice}', pretrained=False, num_classes=num_classes)
        else: model = create_modern_cnn(num_classes=num_classes, variant=variant_choice, drop_path_rate=DROP_PATH_RATE)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)

        if use_transfer_learning:
            if start_epoch < WARMUP_EPOCHS:
                print("Resuming in warm-up phase (head only).")
                for param in model.parameters(): param.requires_grad = False
                if hasattr(model, 'head'):
                    for param in model.head.parameters(): param.requires_grad = True
                    optimizer = optim.AdamW(model.head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
                    scheduler = OneCycleLR(optimizer, max_lr=HEAD_LR, total_steps=EPOCHS * len(train_loader))
            else:
                print("Resuming in fine-tuning phase (all layers).")
                for param in model.parameters(): param.requires_grad = True
                optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
                scheduler = OneCycleLR(optimizer, max_lr=FINETUNE_LR, total_steps=EPOCHS * len(train_loader))
        else: 
            optimizer = optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
            scheduler = OneCycleLR(optimizer, max_lr=HEAD_LR, total_steps=EPOCHS * len(train_loader))

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Successfully loaded model. Resuming from epoch {start_epoch + 1}.")

    else:
        use_transfer_learning = input("Use Transfer Learning? (y/n) [y]: ").lower() != 'n'
        variant_choice = input(f"Select model variant {list(_VARIANTS.keys())} [{DEFAULT_VARIANT}]: ").strip().lower() or DEFAULT_VARIANT
        if variant_choice not in _VARIANTS: variant_choice = DEFAULT_VARIANT
        session_name = f"ConvNeXtV2_{variant_choice}_{'Finetuned' if use_transfer_learning else 'Scratch'}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        if use_transfer_learning:
            print(f"Loading pre-trained 'convnext_{variant_choice}' from timm...")
            model = timm.create_model(f'convnext_{variant_choice}', pretrained=True, num_classes=num_classes)
            print("Setting up optimizer for warm-up (head only).")
            for param in model.parameters(): param.requires_grad = False
            if hasattr(model, 'head'):
                 for param in model.head.parameters(): param.requires_grad = True
                 optimizer = optim.AdamW(model.head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
                 scheduler = OneCycleLR(optimizer, max_lr=HEAD_LR, total_steps=EPOCHS * len(train_loader))

        else:
            print("Training model from scratch...")
            model = create_modern_cnn(num_classes=num_classes, variant=variant_choice, drop_path_rate=DROP_PATH_RATE)
            optimizer = optim.AdamW(model.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
            scheduler = OneCycleLR(optimizer, max_lr=HEAD_LR, total_steps=EPOCHS * len(train_loader))
        model.to(device)

    train_targets = [dataset.targets[i] for i in train_dataset.indices]
    class_weights = 1. / torch.tensor(np.bincount(train_targets, minlength=num_classes), dtype=torch.float)
    class_weights[torch.isinf(class_weights)] = 0
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    print("\n" + "#"*22 + " Training Statistics " + "#"*22)
    print(f"Session: {session_name}, Device: {device.type.upper()}, Transfer Learning: {use_transfer_learning}")
    print(f"Trainable Params (Current): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("#"*61 + "\n")

    patience_counter, early_stopping_patience = 0, 10
    
    try:
        current_epoch = start_epoch
        for epoch in range(start_epoch, EPOCHS):
            current_epoch = epoch
            if use_transfer_learning and epoch == WARMUP_EPOCHS and epoch != 0:
                print("\n" + "*"*20 + " Warm-Up Complete " + "*"*20)
                print("Unfreezing all layers for fine-tuning...")
                for param in model.parameters(): param.requires_grad = True
                optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LR, weight_decay=WEIGHT_DECAY)
                scheduler = OneCycleLR(optimizer, max_lr=FINETUNE_LR, total_steps=EPOCHS * len(train_loader))
                print(f"All {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters are now trainable.")

            model.train()
            stage = "Warm-up" if use_transfer_learning and epoch < WARMUP_EPOCHS else "Fine-tuning"
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS} [{stage}]', colour='cyan')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                
                #Mixed Precision Training Context
                with torch.autocast(device_type=device.type):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                pbar.set_postfix(loss=f'{loss.item():.4f}')

            model.eval()
            all_preds, all_labels = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    #Uses autocast for evaluation as well for consistency
                    with torch.autocast(device_type=device.type):
                        outputs = model(inputs.to(device))
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())

            expected_labels = list(range(num_classes)) 
            report = classification_report(all_labels, all_preds, labels=expected_labels, target_names=class_names, zero_division=0, output_dict=True)
            val_f1 = report['macro avg']['f1-score']
            print(f"Epoch {epoch+1} Summary | Val Macro F1: {val_f1:.4f} | Best F1: {best_f1:.4f}")

            is_best = val_f1 > best_f1
            if is_best: best_f1, patience_counter = val_f1, 0
            else: patience_counter += 1

            save_checkpoint({'epoch': epoch + 1, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_f1': best_f1, 'variant': variant_choice, 'use_transfer_learning': use_transfer_learning}, session_name, is_best)
            if patience_counter >= early_stopping_patience: print(f"\nEarly stopping triggered after {epoch+1} epochs."); break

        print(f"\nTraining complete! Best F1: {best_f1:.4f}")
        deploy_model_from_session(session_name, auto_deploy=True)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted! Saving state...")
        save_checkpoint({'epoch': current_epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_f1': best_f1, 'variant': variant_choice, 'use_transfer_learning': use_transfer_learning}, session_name, is_best=False, is_interrupt=True)
        sys.exit(0)

def continue_training(device):
    print("\n### Select a Session to Continue Training ###")
    if not os.path.exists(CHECKPOINT_DIR) or not os.listdir(CHECKPOINT_DIR): print("\nNo training sessions found."); return
    sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
    if not sessions: print("\nNo training sessions found."); return

    for i, s in enumerate(sessions):
        status = "[PAUSED]" if os.path.exists(os.path.join(CHECKPOINT_DIR, s, 'interrupt.pth.tar')) else ""
        print(f"{i+1}. {s} {status}")

    try:
        choice = int(input(f"Select session (1-{len(sessions)}): ")) - 1
        if 0 <= choice < len(sessions): train_model(device, session_name=sessions[choice])
        else: print("Invalid selection.")
    except (ValueError, IndexError): print("Invalid selection.")

#Evaluates the model and its accuracies for each dataset class.
def evaluate_deployed_model_text(device):
    print("\n" + "#" * 22 + " Deployed Model Evaluation " + "#" * 22)
    if not os.path.exists(DEPLOY_DIR) or not os.listdir(DEPLOY_DIR): print("\nNo deployed models found."); return
    models = sorted([f for f in os.listdir(DEPLOY_DIR) if f.endswith('.pth') or f.endswith('.pth.tar')])
    if not models: print("\nNo deployed models found."); return
    
    print("\n### Select a Deployed Model to Evaluate ###")
    for i, m in enumerate(models): print(f"{i+1}. {m}")
    try:
        model_name = models[int(input(f"Select model (1-{len(models)}): ")) - 1]
        model_path = os.path.join(DEPLOY_DIR, model_name)
    except (ValueError, IndexError): print("Invalid selection."); return

    if not os.path.isdir(PROCESSED_DATASET_DIR): print(f"Error: Processed dataset not found."); return
    with open(os.path.join(PROCESSED_DATASET_DIR, 'class_mapping.json'), 'r') as f: class_to_idx = json.load(f)
    num_classes, class_names = len(class_to_idx), [k for k, v in sorted(class_to_idx.items(), key=lambda item: item[1])]

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    full_dataset = ImageFolder(PROCESSED_DATASET_DIR, transform=transform)

    print("Grouping images to create a leakage-free validation set...")
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
    val_split_idx = int(len(unique_base_names) * VALIDATION_SPLIT)
    val_groups = unique_base_names[:val_split_idx]
    val_indices = [idx for group in val_groups for idx in image_groups[group]]
    val_dataset = Subset(full_dataset, val_indices)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Prepared leakage-free validation set with {len(val_dataset)} images.")

    print(f"Loading model: {model_name}...")
    checkpoint = torch.load(model_path, map_location=device)
    variant, use_tl = checkpoint.get('variant', DEFAULT_VARIANT), checkpoint.get('use_transfer_learning', False)

    if use_tl: model = timm.create_model(f'convnext_{variant}', pretrained=False, num_classes=num_classes)
    else: model = create_modern_cnn(num_classes=num_classes, variant=variant)
    
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict); model.to(device); model.eval()

    print("Running evaluation...")
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Evaluating"):
            outputs = model(inputs.to(device)); _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(labels.cpu().numpy())
            
    print("\n" + "="*30 + " CLASSIFICATION REPORT " + "="*29)
    print(f"Model: {model_name}\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, zero_division=0))
    print("="*81)

def deploy_model_from_session(session_name=None, auto_deploy=False):
    if session_name is None:
        if not os.path.exists(CHECKPOINT_DIR) or not os.listdir(CHECKPOINT_DIR): print("\nNo training sessions found."); return
        sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
        if not sessions: print("\nNo training sessions found."); return
        print("\n### Select a Session to Deploy Its Best Model ###")
        for i, s in enumerate(sessions): print(f"{i+1}. {s}")
        try:
            session_name = sessions[int(input(f"Select session (1-{len(sessions)}): ")) - 1]
        except (ValueError, IndexError): print("Invalid selection."); return

    best_model_path = os.path.join(CHECKPOINT_DIR, session_name, 'model_best.pth.tar')
    if not os.path.exists(best_model_path):
        print(f"Error: Best model not found in '{session_name}'. Deploying latest checkpoint instead.")
        best_model_path = os.path.join(CHECKPOINT_DIR, session_name, 'checkpoint.pth.tar')
        if not os.path.exists(best_model_path): print(f"Error: No checkpoints found in '{session_name}'."); return

    deploy_path = os.path.join(DEPLOY_DIR, f"{session_name}_deployable.pth")
    shutil.copy(best_model_path, deploy_path)
    msg = "automatically deployed" if auto_deploy else "deployed successfully"
    print(f"\n>> Model from '{session_name}' {msg} to: {deploy_path}")

def deploy_latest_from_session():
    if not os.path.exists(CHECKPOINT_DIR) or not os.listdir(CHECKPOINT_DIR):
        print("\nNo training sessions found.")
        return
        
    sessions = sorted([d for d in os.listdir(CHECKPOINT_DIR) if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
    if not sessions:
        print("\nNo training sessions found.")
        return
        
    print("\n### Select a Session to Deploy Its LATEST Model ###")
    for i, s in enumerate(sessions):
        print(f"{i+1}. {s}")
        
    try:
        session_name = sessions[int(input(f"Select session (1-{len(sessions)}): ")) - 1]
    except (ValueError, IndexError):
        print("Invalid selection.")
        return

    session_dir = os.path.join(CHECKPOINT_DIR, session_name)
    latest_model_path = os.path.join(session_dir, 'interrupt.pth.tar')
    
    if not os.path.exists(latest_model_path):
        latest_model_path = os.path.join(session_dir, 'checkpoint.pth.tar')

    if not os.path.exists(latest_model_path):
        print(f"Error: No checkpoint files found in '{session_name}'.")
        return

    deploy_path = os.path.join(DEPLOY_DIR, f"{session_name}_latest_deployable.pth")
    shutil.copy(latest_model_path, deploy_path)
    print(f"\n>> LATEST model from '{session_name}' deployed successfully to: {deploy_path}")


### Main Menu System ###

def print_main_menu():
    print("\n" + "#" * 22 + " ModernCNN Training Toolkit " + "#" * 22)
    print(" 1. Pre-process & Augment Raw Image Data")
    print(" 2. Train a New Model (with Transfer Learning)")
    print(" 3. Continue a Paused Training Session")
    print(" 4. Deploy BEST Model from a Session")
    print(" 5. Deploy LATEST Model from a Session") 
    print(" 6. Evaluate Model (Text Report)")
    print(" 7. Evaluate Models with Graphs")
    print(" 8. Exit") 
    print("#" * 76)

def start_cli():
    device = get_device()
    for path in [RAW_DATASET_DIR, PROCESSED_DATASET_DIR, CHECKPOINT_DIR, DEPLOY_DIR, GRAPHS_DIR, PER_MODEL_GRAPHS_DIR]:
        os.makedirs(path, exist_ok=True)

    while True:
        print_main_menu()
        choice = input("Enter your choice: ").strip()
        if choice == '1': preprocess_and_save_images()
        elif choice == '2': train_model(device)
        elif choice == '3': continue_training(device)
        elif choice == '4': deploy_model_from_session()
        elif choice == '5': deploy_latest_from_session() 
        elif choice == '6': evaluate_deployed_model_text(device) 
        elif choice == '7': 
            generate_validation_graphs(device=device, processed_dir=PROCESSED_DATASET_DIR, deploy_dir=DEPLOY_DIR, graphs_dir=GRAPHS_DIR, per_model_graphs_dir=PER_MODEL_GRAPHS_DIR, default_variant=DEFAULT_VARIANT, val_split=VALIDATION_SPLIT, batch_size=BATCH_SIZE)
        elif choice == '8': print("\nExiting program..."); break
        else: print("\nInvalid choice. Please try again.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    start_cli()
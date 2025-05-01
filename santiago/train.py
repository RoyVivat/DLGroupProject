# Importing libraries
print("Importing libraries...")
from math import ceil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy
from collections import Counter
import mlflow
import yaml
from model import LanguageModel
from utils import diff_bill_formattter, compute_rouge
from data_tokenizers import tokenizers_dict
from transformers import get_linear_schedule_with_warmup
from utils import dynamic_pad_collate 


# Load parameters
with open('params.yml', 'r') as f:
    params = yaml.safe_load(f)

# Extract parameters
clean_diff_text = params['clean_diff_text']
vocab_size = params['vocab_size']
dim_size = params['dim_size'] 
dim_feedforward = params['dim_feedforward']
num_layers = params['num_layers']
num_heads = params['num_heads']
dropout = params['dropout']
learning_rate = params['learning_rate']
weight_decay = params['weight_decay']
max_len = params['max_len']
min_summary_length = params['min_summary_length']
epochs = params['epochs']
batch_size = params['batch_size']
sample_size = params['sample_size']
grad_accum_steps = params['grad_accum_steps']
tokenizer_name = params['tokenizer']
label_smoothing = params['label_smoothing']
warmup_steps = params['warmup_steps']
beam_size_val = params['beam_size_val']
beam_size_sample = params['beam_size_sample']
length_penalty = params['length_penalty']
device = "cuda" if torch.cuda.is_available() else "cpu"


# Load data
print("Loading data...")
df = pd.read_csv('data.csv')
# Selecting relevant columns
df = df[["diff_text", "summary"]]
# Dropping rows with missing values
df = df.dropna().reset_index(drop=True)
# Removing empty strings
df = df[df["diff_text"].str.len() > 0]
df = df[df["summary"].str.len() > 0]
# Cleaning diff text
if clean_diff_text:
    df["diff_text"] = df["diff_text"].apply(diff_bill_formattter)
else:
    df["diff_text"] = df["diff_text"].apply(lambda x: x.replace("<", " <").replace(">", "> "))
# building sequence
sequence_raw = ("<bos> " + df["diff_text"] + " <sep> " + df["summary"] + " <eos>").tolist()
# Used to overfit to training data as debug
sequence_raw = sequence_raw[:sample_size]

# Build tokenizer
tokenizer = tokenizers_dict[tokenizer_name](sequence_raw, vocab_size=vocab_size)

# tokenize the sequence
sequence, sequence_lengths = tokenizer.tokenize(sequence_raw, training=True)

# Convert to tensors
x = sequence[:, :-1].clone()
y = sequence[:, 1:].clone()

# Position skipping for bills in cross entropy
sep_id = tokenizer.sep_id
pad_id = tokenizer.pad_id
eos_id = tokenizer.eos_id
# Masking after the summary is completed
before_sep = torch.cumsum((y == sep_id), dim=1) == 0
after_eos = torch.cumsum((y == eos_id), dim=1) > 0
ignore = before_sep | after_eos | (y == pad_id)
y.masked_fill_(ignore, -100)

print("Splitting data into train/val sets...")
# Splitting the data
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.1, random_state=42)

# Create TensorDataset
train_dataset = TensorDataset(x_train, y_train)
val_dataset = TensorDataset(x_val, y_val)

print("Creating data loaders...")
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    collate_fn=lambda b: dynamic_pad_collate(b, pad_id)
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=lambda b: dynamic_pad_collate(b, pad_id)
)

print("Initializing model...")
# Loading the model
model = LanguageModel(
    tokenizer=tokenizer,
    dim_size=dim_size,
    dim_feedforward=dim_feedforward,
    num_layers=num_layers,
    num_heads=num_heads,
    dropout=dropout,
    max_len=max_len,
    device=device,
).to(device)
# Setting the optimizer and loss function
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=weight_decay
)
# Set learning rate scheduler
updates_per_epoch = ceil(len(train_loader) / grad_accum_steps)
total_updates = epochs * updates_per_epoch
warmup_steps = int(0.06 * total_updates) if warmup_steps is None else warmup_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps = warmup_steps,
    num_training_steps = total_updates,
)
# Automatic Mixed Precision (only if CUDA is available)
use_amp = device == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
# Evaluation criterion
criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing)
# Accuracy metric
accuracy = Accuracy(
    task="multiclass",
    num_classes=tokenizer.vocab_size,
    ignore_index=-100
).cpu()

# MLflow setup
mlflow.set_experiment("language_model_training")
mlflow.start_run()

# Log parameters
mlflow.log_params({
    "clean_diff_text": clean_diff_text,
    "dim_size": dim_size,
    "dim_feedforward": dim_feedforward,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "dropout": dropout,
    "learning_rate": learning_rate,
    "weight_decay": weight_decay,
    "warmup_steps": warmup_steps,
    "grad_accum_steps": grad_accum_steps,
    "weight_updates_freq": batch_size * grad_accum_steps,
    "max_len": max_len,
    "min_summary_length": min_summary_length,
    "epochs": epochs,
    "batch_size": batch_size,
    "sample_size": sample_size,
    "vocab_size": vocab_size,
    "train_size": len(train_dataset),
    "val_size": len(val_dataset),
    "tokenizer": tokenizer_name,
    "label_smoothing": label_smoothing,
})

# Log token length stats
sequence_lengths_stats = tokenizer.get_sequence_statistics(sequence_lengths)
sequence_lengths_stats = {f"sequence_length_{k.replace('%', '_perc')}": v for k, v in sequence_lengths_stats.items()}
mlflow.log_params(sequence_lengths_stats)


print("Starting training...")
# Starts model training
for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("Training...")
    optimizer.zero_grad(set_to_none=True)
    model.train()
    batch_train_losses = []
    batch_train_accuracies = []
    # Train loop
    for i, (x_train_batch, y_train_batch) in enumerate(train_loader):
        x_train_batch = x_train_batch.to(device)
        y_train_batch = y_train_batch.to(device)
        # Forward pass
        with torch.amp.autocast("cuda", enabled=use_amp):
            y_hat_train_batch = model(x_train_batch)
            y_hat_train_flat  = y_hat_train_batch.view(-1, y_hat_train_batch.size(-1))
            y_train_flat = y_train_batch.view(-1)   
            train_loss = criterion(y_hat_train_flat, y_train_flat) / grad_accum_steps
        # Backward pass
        scaler.scale(train_loss).backward()
        #Gradient accumulation
        if (i + 1) % grad_accum_steps == 0 or (i + 1) == len(train_loader):
            # clip unscaled grads 
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # Update weights
            scaler.step(optimizer)
            scaler.update()
            # Reset gradients
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        batch_train_losses.append(train_loss.item())
        batch_train_accuracies.append(accuracy(y_hat_train_flat.detach().cpu(), y_train_flat.detach().cpu()))
        print(f"Batch {i+1}/{len(train_loader)}, Train Loss: {train_loss.item()}, Train Accuracy: {batch_train_accuracies[-1]}")
        # Log batch-level metrics to MLflow
        mlflow.log_metrics({
            "batch_train_loss": train_loss.item(),
            "batch_train_accuracy": batch_train_accuracies[-1],
            "learning_rate": scheduler.get_last_lr()[0]
        }, step=epoch * len(train_loader) + i)
    print("Validating...")
    # Val loop
    model.eval()
    batch_val_losses = []
    batch_val_accuracies = []
    with torch.no_grad():
        for i, (x_val_batch, y_val_batch) in enumerate(val_loader):
            x_val_batch = x_val_batch.to(device)
            y_val_batch = y_val_batch.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                y_hat_val_batch = model(x_val_batch)
                y_hat_val_flat  = y_hat_val_batch.view(-1, y_hat_val_batch.size(-1))   # (B*L‑1, V)
                y_val_flat = y_val_batch.view(-1)   
                val_loss = criterion(y_hat_val_flat, y_val_flat)
                batch_val_losses.append(val_loss.item())
                batch_val_accuracies.append(accuracy(y_hat_val_flat.detach().cpu(), y_val_flat.detach().cpu()))
    # Compute ROUGE scores for validation set
    preds, refs = [], []
    val_indices = x_val_batch.new_tensor(range(len(x_val))).tolist()
    val_diffs   = df.loc[val_indices, "diff_text"].tolist()
    val_summaries = df.loc[val_indices, "summary"].tolist()

    model.eval()
    with torch.no_grad():
        for diff in val_diffs:
            preds.append(model.generate(diff, beam_size=beam_size_val, length_penalty=length_penalty))
    
    refs = val_summaries
    rouge_dict = compute_rouge(preds, refs)

    # Log losses and accuracies
    train_loss = torch.tensor(batch_train_losses).mean()
    train_accuracy = torch.tensor(batch_train_accuracies).mean()
    val_loss = torch.tensor(batch_val_losses).mean()
    val_accuracy = torch.tensor(batch_val_accuracies).mean()
    print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss.item()}, Train Accuracy: {train_accuracy.item()}, Val Loss: {val_loss.item()}, Val Accuracy: {val_accuracy.item()}")
    print(f"Val ROUGE-1 F1={rouge_dict['rouge1']['f1']:.4f}, Val ROUGE-2 F1={rouge_dict['rouge2']['f1']:.4f}, Val ROUGE-L F1={rouge_dict['rougeL']['f1']:.4f}")

    # Log metrics to MLflow
    mlflow.log_metrics({
        "train_loss": train_loss.item(),
        "train_accuracy": train_accuracy.item(),
        "val_loss": val_loss.item(),
        "val_accuracy": val_accuracy.item()
    }, step=epoch)
    mlflow.log_metrics(
        {f"val_rouge_{m}_{k}": v for m, d in rouge_dict.items() for k, v in d.items()},
        step=epoch
    )
    # Reset accuracy
    accuracy.reset()

print("Finished training loop!")

# Saving the model
torch.save(model.state_dict(), "model.pth")
mlflow.log_artifact("model.pth")

# Running inference on five examples
print("Running inference on five examples...")
for i in range(5):
    example_diff = df.loc[i, "diff_text"]
    example_summary = model.generate(example_diff, beam_size=beam_size_sample, length_penalty=length_penalty)
    print(f"Example Diff: {example_diff}")
    print(f"Example Summary: {example_summary}")
    print("-"*100)
    # Log the example
    mlflow.log_text(f"Example Diff: {example_diff}\nExample Summary: {example_summary}", f"example_{i}.txt")

mlflow.end_run()
print("Training completed!")


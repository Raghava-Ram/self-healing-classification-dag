import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, TaskType
import numpy as np
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {"accuracy": accuracy_score(labels, predictions)}

def fine_tune():
    # Load dataset
    dataset = load_dataset("imdb")
    
    # Initialize tokenizer and model
    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=128
        )
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # Prepare for training
    train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))  # Small subset for demo
    eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(100))
    
    # Initialize model with LoRA
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1}
    )
    
    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=4,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_lin", "v_lin"],
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",  # Changed from evaluation_strategy to eval_strategy
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_strategy="no",
        load_best_model_at_end=False,
        logging_dir='./logs',
        logging_steps=10,
        no_cuda=True,  # Force CPU usage
        report_to="none"
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    print("Starting training...")
    trainer.train()
    
    # Save the model
    output_dir = "./fine_tuned_model"
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    fine_tune()

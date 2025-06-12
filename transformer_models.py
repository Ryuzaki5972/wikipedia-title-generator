import pandas as pd
import numpy as np
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from rouge import Rouge
import nltk
import time
import os
import json
from copy import deepcopy

# Download the NLTK punkt tokenizer
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class TransformerTitleGenerator:
    def __init__(self, model_name="google-t5/t5-small", device=None):
        """
        Initialize the transformer model for title generation
        
        Args:
            model_name (str): Name of the pretrained model to use
            device (str): Device to run the model on (cuda or cpu)
        """
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        # Set up prefix for different models
        if "flan" in model_name:
            # For Flan-T5 models
            self.is_flan = True
        else:
            # For regular T5 models
            self.is_flan = False
            
        self.rouge = Rouge()
    
    def preprocess_data(self, train_df, val_df, test_df, prefix=""):
        """
        Preprocess data for training and evaluation
        
        Args:
            train_df, val_df, test_df (pandas.DataFrame): Input dataframes
            prefix (str): Prefix for the prompt
            
        Returns:
            datasets (dict): Dictionary of train, validation and test datasets
        """
        def preprocess_examples(examples):
            # For T5, we generally format as "prefix: input_text"
            if self.is_flan:
                inputs = [prefix + " " + text for text in examples["text"]]
            else:
                inputs = ["summarize: " + text for text in examples["text"]]
                
            # Truncate inputs to max length
            model_inputs = self.tokenizer(
                inputs, 
                max_length=512,  # T5 max input length
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Tokenize targets
            targets = self.tokenizer(
                examples["title"],
                max_length=64,  # Reasonable max title length
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            model_inputs["labels"] = targets["input_ids"]
            return model_inputs
        
        # Convert dataframes to datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        test_dataset = Dataset.from_pandas(test_df)
        
        # Preprocess datasets
        train_dataset = train_dataset.map(
            preprocess_examples,
            batched=True,
            remove_columns=["text", "title"]
        )
        val_dataset = val_dataset.map(
            preprocess_examples,
            batched=True,
            remove_columns=["text", "title"]
        )
        test_dataset = test_dataset.map(
            preprocess_examples,
            batched=True,
            remove_columns=["text", "title"]
        )
        
        datasets = {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset
        }
        
        return datasets
    
    def train(self, train_dataset, val_dataset, output_dir="./results", hyperparams=None):
        """
        Fine-tune the model on the provided datasets with specified hyperparameters
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            output_dir (str): Directory to save the model
            hyperparams (dict): Dictionary of hyperparameters for training
            
        Returns:
            trainer: The trained model trainer
            scores: Evaluation scores on validation set
        """
        # Use default hyperparameters if none are provided
        if hyperparams is None:
            hyperparams = {
                "learning_rate": 5e-5,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "weight_decay": 0.01,
                "num_train_epochs": 3,
            }
        
        # Define training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=output_dir,
            eval_steps=100,
            save_steps=100,
            eval_strategy="steps",
            save_strategy="steps",
            learning_rate=hyperparams["learning_rate"],
            per_device_train_batch_size=hyperparams["per_device_train_batch_size"],
            per_device_eval_batch_size=hyperparams["per_device_eval_batch_size"],
            weight_decay=hyperparams["weight_decay"],
            save_total_limit=3,
            num_train_epochs=hyperparams["num_train_epochs"],
            predict_with_generate=True,
            load_best_model_at_end=True,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            model=self.model,
            padding=True
        )
        
        # Initialize trainer
        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator
        )
        
        # Train the model
        print(f"Training {self.model_name} model with hyperparameters: {hyperparams}")
        start_time = time.time()
        trainer.train()
        end_time = time.time()
        
        print(f"Training completed in {end_time - start_time:.2f} seconds")
        
        # Evaluate the model on validation set
        eval_results = trainer.evaluate()
        
        # Save the model
        model_output_dir = f"{output_dir}/model_lr{hyperparams['learning_rate']}_bs{hyperparams['per_device_train_batch_size']}_wd{hyperparams['weight_decay']}_ep{hyperparams['num_train_epochs']}"
        os.makedirs(model_output_dir, exist_ok=True)
        self.model.save_pretrained(model_output_dir)
        self.tokenizer.save_pretrained(model_output_dir)
        
        # Save evaluation results
        with open(f"{model_output_dir}/eval_results.json", "w") as f:
            json.dump(eval_results, f)
        
        return trainer, eval_results
    
    def train_with_hyperparameter_search(self, train_dataset, val_dataset, output_dir="./results"):
        """
        Train the model with different hyperparameter configurations
        
        Args:
            train_dataset (Dataset): Training dataset
            val_dataset (Dataset): Validation dataset
            output_dir (str): Directory to save the model
            
        Returns:
            best_config (dict): Best hyperparameter configuration
            best_score (float): Best evaluation score
            all_results (dict): All hyperparameter configurations and their scores
        """
        # Define hyperparameter configurations to try
        hyperparameter_configs = [
            {
                "learning_rate": 1e-4,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "weight_decay": 0.01,
                "num_train_epochs": 3,
            },
            {
                "learning_rate": 5e-5,
                "per_device_train_batch_size": 4,
                "per_device_eval_batch_size": 4,
                "weight_decay": 0.01,
                "num_train_epochs": 3,
            },
            {
                "learning_rate": 3e-5,
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "weight_decay": 0.01,
                "num_train_epochs": 3,
            },
            {
                "learning_rate": 5e-5,
                "per_device_train_batch_size": 8,
                "per_device_eval_batch_size": 8,
                "weight_decay": 0.1,
                "num_train_epochs": 2,
            },
        ]
        
        all_results = {}
        best_score = float('inf')  # Lower is better for evaluation loss
        best_config = None
        best_config_index = None
        
        # Try each hyperparameter configuration
        for i, config in enumerate(hyperparameter_configs):
            print(f"\nTrying hyperparameter configuration {i+1}/{len(hyperparameter_configs)}")
            config_dir = f"{output_dir}/config_{i+1}"
            os.makedirs(config_dir, exist_ok=True)
            
            # We need to reload the model for each configuration to start fresh
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            
            # Train with current configuration
            trainer, eval_results = self.train(
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                output_dir=config_dir,
                hyperparams=config
            )
            
            # Extract eval_loss, ensuring it's a native Python float
            current_score = float(eval_results.get("eval_loss", float('inf')))
            
            # Store results
            config_result = {
                "config": config,
                "eval_results": eval_results,
                "eval_loss": current_score  # Store the loss as a separate field for easier access
            }
            all_results[f"config_{i+1}"] = config_result
            
            # Check if this is the best configuration so far
            if current_score < best_score:
                best_score = current_score
                best_config = config
                best_config_index = i + 1
            
            print(f"Configuration {i+1} eval_loss: {current_score:.4f}")
        
        # Save all results
        with open(f"{output_dir}/hyperparameter_search_results.json", "w") as f:
            # Convert to serializable format
            serializable_results = {}
            for k, v in all_results.items():
                serializable_results[k] = {
                    "config": v["config"],
                    "eval_results": {key: float(val) for key, val in v["eval_results"].items()},
                    "eval_loss": v["eval_loss"]
                }
            json.dump(serializable_results, f, indent=2)
        
        if best_config_index is not None:
            # Load the best model
            best_model_dir = f"{output_dir}/config_{best_config_index}"
            print(f"Loading best model from {best_model_dir}")
            
            try:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(best_model_dir).to(self.device)
                
                # Save the best model with a descriptive name
                best_model_output_dir = f"{output_dir}/best_model"
                os.makedirs(best_model_output_dir, exist_ok=True)
                self.model.save_pretrained(best_model_output_dir)
                self.tokenizer.save_pretrained(best_model_output_dir)
                
                # Save best configuration
                with open(f"{best_model_output_dir}/best_config.json", "w") as f:
                    json.dump(best_config, f, indent=2)
                    
                print(f"Saved best model with configuration index {best_config_index}")
            except Exception as e:
                print(f"Error loading best model: {e}")
                print("Using the current model state instead")
        else:
            print("No best configuration found, using current model state")
        
        return best_config, best_score, all_results
    
    def generate_titles(self, test_df, prefix="", use_beam_search=False, num_beams=5):
        """
        Generate titles for the test set
        
        Args:
            test_df (pandas.DataFrame): Test dataframe
            prefix (str): Prefix for the prompt
            use_beam_search (bool): Whether to use beam search or greedy decoding
            num_beams (int): Number of beams for beam search
            
        Returns:
            generated_titles (list): List of generated titles
            reference_titles (list): List of reference titles
        """
        generated_titles = []
        reference_titles = test_df["title"].tolist()
        
        # Generate titles batch by batch
        batch_size = 16
        for i in range(0, len(test_df), batch_size):
            batch_texts = test_df["text"].iloc[i:i+batch_size].tolist()
            
            # Create inputs
            if self.is_flan:
                inputs = [prefix + " " + text for text in batch_texts]
            else:
                inputs = ["summarize: " + text for text in batch_texts]
            
            # Tokenize inputs
            inputs = self.tokenizer(
                inputs,
                max_length=512,
                truncation=True,
                padding=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate titles
            with torch.no_grad():
                if use_beam_search:
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=64,
                        num_beams=num_beams,
                        early_stopping=True
                    )
                else:
                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=inputs["attention_mask"],
                        max_length=64
                    )
            
            # Decode outputs
            decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            generated_titles.extend(decoded_outputs)
            
            # Print progress
            print(f"Generated {len(generated_titles)}/{len(test_df)} titles")
        
        return generated_titles, reference_titles
    
    def evaluate(self, generated_titles, reference_titles):
        """
        Evaluate generated titles using ROUGE scores
        
        Args:
            generated_titles (list): List of generated titles
            reference_titles (list): List of reference titles
            
        Returns:
            scores (dict): Dictionary of ROUGE scores
        """
        # Prepare references and hypotheses for ROUGE
        hypotheses = generated_titles
        references = reference_titles
        
        # Calculate ROUGE scores
        scores = {}
        for i, (hyp, ref) in enumerate(zip(hypotheses, references)):
            # Skip if empty
            if not hyp or not ref:
                continue
                
            try:
                rouge_scores = self.rouge.get_scores(hyp, ref)[0]
                
                # Extract ROUGE-1, ROUGE-2, and ROUGE-L F1 scores
                for rouge_type in ["rouge-1", "rouge-2", "rouge-l"]:
                    if rouge_type not in scores:
                        scores[rouge_type] = []
                    scores[rouge_type].append(rouge_scores[rouge_type]["f"])
            except Exception as e:
                print(f"Error calculating ROUGE scores for example {i}: {e}")
        
        # Calculate average scores
        avg_scores = {rouge_type: np.mean(values) for rouge_type, values in scores.items()}
        
        return avg_scores
    
    def run_zero_shot_prompting(self, test_df, prompts):
        """
        Run zero-shot prompting with various prompts
        
        Args:
            test_df (pandas.DataFrame): Test dataframe
            prompts (list): List of prompts to try
            
        Returns:
            results (dict): Dictionary of results for each prompt
        """
        results = {}
        
        for prompt in prompts:
            print(f"Running zero-shot prompting with prompt: '{prompt}'")
            generated_titles, reference_titles = self.generate_titles(test_df, prefix=prompt)
            scores = self.evaluate(generated_titles, reference_titles)
            
            # Store results
            results[prompt] = {
                "generated_titles": generated_titles,
                "scores": scores 
            }
            
            # Print scores
            print(f"Results for prompt '{prompt}':")
            for rouge_type, score in scores.items():
                print(f"  {rouge_type}: {score:.4f}")
            print()
        
        return results


def main():
    start_time = time.time()

    # Load datasets
    print("Loading datasets...")
    load_start = time.time()
    train_df = pd.read_csv('train_raw.csv')
    val_df = pd.read_csv('val_raw.csv')
    test_df = pd.read_csv('test_raw.csv')
    load_end = time.time()
    print(f"Loaded datasets. Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Create output directory
    dir_start = time.time()
    os.makedirs("./results", exist_ok=True)
    dir_end = time.time()

    # Task C1: Fine-tuning T5 model with hyperparameter search
    print("\n===== Task C1: Fine-tuning T5 model with hyperparameter search =====")
    t5_generator = TransformerTitleGenerator("google-t5/t5-small")

    # Preprocess data
    t5_datasets = t5_generator.preprocess_data(train_df, val_df, test_df)

    # Train model with hyperparameter search
    train_start = time.time()
    best_config, best_score, all_results = t5_generator.train_with_hyperparameter_search(
        t5_datasets["train"],
        t5_datasets["validation"],
        output_dir="./results/t5_small_hyperparameter_search"
    )
    train_end = time.time()
    
    print("\nBest hyperparameter configuration:")
    for param, value in best_config.items():
        print(f"  {param}: {value}")
    print(f"Best validation loss: {best_score:.4f}")

    # Generate and evaluate titles using greedy search
    print("\nGenerating titles using greedy search...")
    greedy_start = time.time()
    t5_generated_titles_greedy, reference_titles = t5_generator.generate_titles(
        test_df,
        use_beam_search=False
    )
    greedy_end = time.time()

    # Evaluate greedy search
    print("\nEvaluating greedy search results...")
    greedy_eval_start = time.time()
    t5_scores_greedy = t5_generator.evaluate(t5_generated_titles_greedy, reference_titles)
    greedy_eval_end = time.time()
    print("T5 (greedy search) ROUGE scores:")
    for rouge_type, score in t5_scores_greedy.items():
        print(f" {rouge_type}: {score:.4f}")

    # Generate and evaluate titles using beam search
    print("\nGenerating titles using beam search...")
    beam_start = time.time()
    t5_generated_titles_beam, _ = t5_generator.generate_titles(
        test_df,
        use_beam_search=True,
        num_beams=5
    )
    beam_end = time.time()

    # Evaluate beam search
    print("\nEvaluating beam search results...")
    beam_eval_start = time.time()
    t5_scores_beam = t5_generator.evaluate(t5_generated_titles_beam, reference_titles)
    beam_eval_end = time.time()
    print("T5 (beam search) ROUGE scores:")
    for rouge_type, score in t5_scores_beam.items():
        print(f" {rouge_type}: {score:.4f}")

    # Save results
    results = {
        "t5_small": {
            "best_hyperparams": best_config,
            "best_eval_loss": float(best_score),
            "greedy": {
                "titles": t5_generated_titles_greedy,
                "scores": t5_scores_greedy
            },
            "beam": {
                "titles": t5_generated_titles_beam,
                "scores": t5_scores_beam
            }
        }
    }

    # Task C2: Zero-shot prompting with Flan-T5
    print("\n===== Task C2: Zero-shot prompting with Flan-T5 =====")
    
    # Define prompts to try
    prompts = [
        "Generate a title for this Wikipedia article:",
        "Create a concise, informative title for the following text:",
        "Summarize this Wikipedia article with a single title:",
        "Extract the main topic and create a title for this passage:",
    ]

    # Try with Flan-T5-base
    print("\nRunning zero-shot prompting with Flan-T5-base...")
    flan_t5_base = TransformerTitleGenerator("google/flan-t5-base")
    zero_shot_base_start = time.time()
    flan_t5_base_results = flan_t5_base.run_zero_shot_prompting(test_df, prompts)
    zero_shot_base_end = time.time()

    # Try with Flan-T5-large
    print("\nRunning zero-shot prompting with Flan-T5-large...")
    flan_t5_large = TransformerTitleGenerator("google/flan-t5-large")
    zero_shot_large_start = time.time()
    flan_t5_large_results = flan_t5_large.run_zero_shot_prompting(test_df, prompts)
    zero_shot_large_end = time.time()

    # Add results to dictionary
    results["flan_t5_base"] = flan_t5_base_results
    results["flan_t5_large"] = flan_t5_large_results

    # Print comparative results
    print("\n===== Comparative Results =====")
    print("T5-small (fine-tuned with best hyperparameters, greedy):")
    for rouge_type, score in results["t5_small"]["greedy"]["scores"].items():
        print(f" {rouge_type}: {score:.4f}")
    print(f" Best hyperparameters: {best_config}")

    print("\nT5-small (fine-tuned with best hyperparameters, beam search):")
    for rouge_type, score in results["t5_small"]["beam"]["scores"].items():
        print(f" {rouge_type}: {score:.4f}")

    print("\nFlan-T5-base (zero-shot, best prompt):")
    best_prompt_base = max(flan_t5_base_results, key=lambda p: flan_t5_base_results[p]["scores"]["rouge-1"])
    for rouge_type, score in flan_t5_base_results[best_prompt_base]["scores"].items():
        print(f" {rouge_type}: {score:.4f}")
    print(f" Best prompt: '{best_prompt_base}'")

    print("\nFlan-T5-large (zero-shot, best prompt):")
    best_prompt_large = max(flan_t5_large_results, key=lambda p: flan_t5_large_results[p]["scores"]["rouge-1"])
    for rouge_type, score in flan_t5_large_results[best_prompt_large]["scores"].items():
        print(f" {rouge_type}: {score:.4f}")
    print(f" Best prompt: '{best_prompt_large}'")

    end_time = time.time()

    # Report time taken for each part
    print("\nTime taken for each part of the code:")
    print(f"Loading datasets: {load_end - load_start:.2f} seconds")
    print(f"Creating output directory: {dir_end - dir_start:.2f} seconds")
    print(f"Fine-tuning T5 model with hyperparameter search: {train_end - train_start:.2f} seconds")
    print(f"Greedy search generation: {greedy_end - greedy_start:.2f} seconds")
    print(f"Greedy search evaluation: {greedy_eval_end - greedy_eval_start:.2f} seconds")
    print(f"Beam search generation: {beam_end - beam_start:.2f} seconds")
    print(f"Beam search evaluation: {beam_eval_end - beam_eval_start:.2f} seconds")
    print(f"Zero-shot with Flan-T5-base: {zero_shot_base_end - zero_shot_base_start:.2f} seconds")
    print(f"Zero-shot with Flan-T5-large: {zero_shot_large_end - zero_shot_large_start:.2f} seconds")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
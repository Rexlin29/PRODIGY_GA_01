# Fine-Tuning GPT-2 for Text Generation  
This project demonstrates how to fine-tune the GPT-2 model for text generation using the Hugging Face Transformers library. GPT-2 is a state-of-the-art language model that generates coherent and contextually relevant text based on given prompts. This project will guide you through setting up the environment, installing the necessary packages, and running a script to fine-tune GPT-2 on a custom dataset and generate text.

## Introduction
Text generation is a powerful application of deep learning where a model generates text based on given prompts. This project utilizes the GPT-2 model, which is known for its ability to produce fluent and contextually appropriate text. By leveraging the Hugging Face Transformers library, we can easily fine-tune and use this model for specific tasks.

The GPT-2 model has various applications, including:
- Creative writing assistance
- Automated content generation
- Chatbot development
- Text completion and summarization

In this project, we will:
- Set up a Python virtual environment.
- Install the necessary dependencies.
- Prepare a custom dataset for fine-tuning.
- Write a Python script to fine-tune GPT-2 on the dataset.
- Generate text using the fine-tuned model.

## Prerequisite
- Install Python 3.7 or higher from the official website (https://www.python.org/downloads/).

## Setup

### 1. Create a folder and activate a virtual environment inside the folder

First, we need to create a folder for the project and create a Python virtual environment in it to manage our project's dependencies. We'll use `venv` for this purpose:

```sh
# Create folder
mkdir folder_name

# Navigate to folder
cd folder_name

# Create virtual environment
python -m venv myenv

# Activate virtual environment (Windows)
myenv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source myenv/bin/activate
```

### 2. Install the required packages

Next, we'll install the necessary Python packages: torch, transformers, and datasets. These packages are required to fine-tune and run the GPT-2 model:

```sh
pip install torch transformers datasets
```

## Code

Create a Python script named `fine_tune_gpt2.py` in your project folder and add the following code:

```python
import os
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True, max_length=128)

def generate_text(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.8,
        top_k=30,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    dataset = load_dataset("text", data_files={"train": "your_dataset.txt"})
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)  
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=["text"])
    
    def add_labels(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples
    
    tokenized_datasets = tokenized_datasets.map(add_labels, batched=True)
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=2,
        learning_rate=3e-5,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=1000,
        eval_strategy="no",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        data_collator=data_collator,
    )

    trainer.train()

    prompt = "Your text prompt here."
    generated_text = generate_text(model, tokenizer, prompt)
    with open("generated_text.txt", "w", encoding="utf-8") as f:
        f.write(generated_text)

if __name__ == "__main__":
    main()
```

## Running the Project

To fine-tune GPT-2 and generate text, run the script with:

```sh
python fine_tune_gpt2.py
```
This will fine-tune GPT-2 on the provided dataset and generate text based on the given prompt, saving the output to `generated_text.txt`.

## Example
The provided example prompt is:

(type in prompt of the code)
"The sun rises over Verona, and the streets begin to bustle with life."

The generated text will be saved as `generated_text.txt` in the same directory as your script.

## Notes

- Ensure that you have a valid dataset (`your_dataset.txt`) in the same directory as your script.
- You can specify different prompts to generate various text outputs.

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these steps:
- Fork the repository.
- Create a new branch with a descriptive name (`git checkout -b my-branch-name`).
- Make your changes and commit them (`git commit -am 'Add some feature'`).
- Push to the branch (`git push origin my-branch-name`).
- Create a new Pull Request.

*Please make sure your code follows the project's coding standards and includes relevant tests.*

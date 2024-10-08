from transformers import GPTNeoXForCausalLM, GPTNeoXTokenizerFast, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os

def load_texts(data_dir):
    texts = []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:
                    texts.append(text)
    return texts

def main():
    data_dir = 'my_dataset/train'  
    model_name = 'EleutherAI/gpt-neox-20b'
    output_dir = 'gpt-output'
    num_train_epochs = 5
    per_device_train_batch_size = 1  
    per_device_eval_batch_size = 1
    learning_rate = 5e-5
    max_seq_length = 2048

    print("Loading data...")
    texts = load_texts(data_dir)

    if len(texts) == 0:
        raise ValueError(f"No data found in {data_dir}. Please check your dataset.")
    else:
        print(f"Loaded {len(texts)} texts from {data_dir}")

    dataset = Dataset.from_dict({"text": texts})

    print("Initializing tokenizer...")
    tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token

    print("Tokenizing data...")
    def tokenize_function(examples):
        tokens = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=max_seq_length)
        return tokens

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    if len(tokenized_dataset) == 0:
        raise ValueError("Tokenization resulted in an empty dataset. Please check your tokenization function.")
    else:
        print(f"Tokenized dataset has {len(tokenized_dataset)} samples")

    print("Loading pre-trained model...")
    model = GPTNeoXForCausalLM.from_pretrained(model_name)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        learning_rate=learning_rate,
        remove_unused_columns=False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    print("Starting Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == '__main__':
    main()


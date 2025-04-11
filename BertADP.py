import argparse
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def predict_and_save(input_csv_path, model_path, output_csv_path):
    # Load the data
    df = pd.read_csv(input_csv_path)
    prepare_df = df.copy()

    # Add spaces between characters in the sequence
    prepare_df['Sequence'] = prepare_df['Sequence'].apply(lambda seq: ' '.join(list(seq)))

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, use_fast=False)

    # Define the tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['Sequence'],
            padding='max_length',
            max_length=41,
            truncation=True,
            is_split_into_words=False,
            add_special_tokens=True
        )

    # Convert to HuggingFace dataset and tokenize
    tokenized_dataset = Dataset.from_pandas(prepare_df).map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # Load the model
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)
    trainer = Trainer(model=model)

    # Perform model prediction
    result = trainer.predict(tokenized_dataset)
    logit = result.predictions
    prob = torch.nn.functional.softmax(torch.tensor(logit), dim=1).numpy()
    positive_prob = prob[:, 1]
    prediction = np.argmax(prob, axis=1)

    # Save prediction results
    df['Positive_Probability'] = positive_prob
    df['Prediction'] = prediction
    df.to_csv(output_csv_path, index=False)
    print(f"Prediction result saved to {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict from CSV using BertADP.")
    parser.add_argument("input_csv", help="Path to the input CSV file") 
    args = parser.parse_args()

    # Fixed path, can be modified if needed
    model_path = "./BertADP"
    output_csv_path = "./prediction_result.csv"

    predict_and_save(
        input_csv_path=args.input_csv,
        model_path=model_path,
        output_csv_path=output_csv_path
    )
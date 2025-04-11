# 🧬 BertADP: A BERT-based Predictor for Anti-Diabetic Peptides
## **BertADP** is a protein sequence classification tool built by fine-tuning the [ProtBert](https://huggingface.co/Rostlab/prot_bert) language model. It predicts whether a peptide sequence has anti-diabetic activity.

---

## 📁 Project Structure
```
BertADP/ 
├── BertADP.py # Main execution script 
├── BertADP/ # Directory containing the fine-tuned model
│ └── adapter_config.json
│ └── adapter_model.safetensors
├── example/ 
│ └── example.csv # Example input file with 10 sequences (5 positive, 5 negative) 
└── requirements.txt # List of required Python packages
```

---

## ⚡ Quick Start
You can clone this repository and run predictions in a few steps:
```
git clone https://github.com/xqxie007/BertADP.git
cd BertADP
pip install -r requirements.txt --ignore-installed
python BertADP.py example/example.csv
```
The prediction results will be saved to prediction_result.csv.

---

## 📦 Environment dependencies  
Please use Python 3.11 or above, and install the following dependencies (it is recommended to use a virtual environment).
```
pip install -r requirements.txt --ignore-installed
```

---

## 📥 Input Format
The input should be a CSV file with a single column named `Sequence`, containing raw amino acid sequences, like:
```
Sequence
GPPGPA
LLNQELLLNPTHQIYPV
SPTIPFFDPQIPK
...
```

---

## 🚀 How to Use (Detailed)
1. Install dependencies (recommended in a virtual environment):
```
pip install -r requirements.txt --ignore-installed
```
2. Run the prediction script:
```
python BertADP.py example/example.csv
```
- example/example.csv can be replaced with your own file.
3. Output
The script will generate a prediction_result.csv file with the following format:
```
Sequence,Positive_Probability,Prediction
GPPGPA,0.96674114,1
LLNQELLLNPTHQIYPV,0.96733195,1
SPTIPFFDPQIPK,0.9591547,1
...
```
- Positive_Probability: Probability that the sequence is an anti-diabetic peptide.  
- Prediction: Classification result (1 = positive, 0 = negative).

---

## 🧠 Model description
The model used in this project is based on the pre-trained **ProtBert** model (`Rostlab/prot_bert`) from the [Hugging Face Model Hub](https://huggingface.co/Rostlab/prot_bert). It is fine-tuned for binary classification to distinguish anti-diabetic peptides (ADPs) from non-ADPs.  

We use the `transformers` library along with the **PEFT** framework and **DoRA (Dropout as Reparameterization of Attention)** for parameter-efficient fine-tuning. Only the final classification head and selected attention modules are updated during training.

Key training details:
- Evaluation and checkpoint saving are done at the end of each epoch.
- The best model is automatically selected based on validation accuracy.

The fine-tuned model weights are saved in the `BertADP/` directory.

---

## ⚠️ Notes
The tokenizer will be automatically downloaded from HuggingFace the first time you run the script. Please ensure you are connected to the internet.

---

## 📄 License
This project is intended for academic and research use only. Please cite appropriately if used in publications.

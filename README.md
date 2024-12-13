# DLBWE-Cys: A Deep-Learning-Based Tool for Identifying Cysteine S-Carboxyethylated Sites Using Binary-Weight Encoding
In this study, we developed a new deep learning model, DLBWE-Cys, which integrates CNN, BiLSTM, Bahdaba attention mechanisms and a fully connected neural network (FNN), using Binary-Weight encoding specifically designed for the accurate identification of cysteine S-carboxyethylated sites.

## Dataset
The sequences with lengths of 21, 31, 41, 51, and 61 amino acids have been uploaded to the `data`, including the training dataset and the independent test dataset.

## Requirements
* python == 3.7.11
* pytorch == 1.6.0 + cuda == 10.1
* pandas == 1.1.0
* numpy == 1.21.5

## Usage
* test_fasta: a fasta file for test samples (the sequence length must exceed 21 amino acids).
* decay: The decay coefficient a for the positional weights of amino acids in a protein sequence has a reasonable range of 0 to 0.09.
* out_dir: The folder name for output file.

For example:
```bash
python main.py -decay 0.02 -test_fasta test.fasta -out_dir output

# Text Sentence Similarity - TOPSIS Model Selection

## Project Overview
This project applies the TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution) method to select the best pre-trained model for text sentence similarity tasks from HuggingFace.

## Assignment Details
- **Task**: Text Sentence Similarity
- **Roll Numbers**: Ending with 3 or 8
- **Objective**: Find the best pre-trained model using TOPSIS methodology

## Project Structure
```
sentence_similarity_topsis/
├── data/                  # Dataset files
├── models/                # Model evaluation results
├── results/               # TOPSIS results, graphs, tables
├── notebooks/             # Jupyter notebooks for exploration
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### 1. Create Conda Environment
```bash
conda create -n sentence_sim python=3.10 -y
conda activate sentence_sim
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

## Methodology
1. Select candidate models from HuggingFace for sentence similarity
2. Evaluate models on standard datasets
3. Apply TOPSIS to rank models based on multiple criteria
4. Document results with graphs and tables

## Results
Results will be uploaded to GitHub with:
- Performance metrics comparison
- TOPSIS analysis
- Visualizations
- Model recommendations

## Author
[Your Name]
[Roll Number]
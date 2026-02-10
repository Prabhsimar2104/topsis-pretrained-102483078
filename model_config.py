"""
Model Configuration for TOPSIS Analysis
Text Sentence Similarity Task
"""

# List of models to evaluate
MODELS = [
    {
        'name': 'all-MiniLM-L6-v2',
        'model_id': 'sentence-transformers/all-MiniLM-L6-v2',
        'parameters_millions': 22.7,
        'description': 'Fast and efficient, general purpose'
    },
    {
        'name': 'all-mpnet-base-v2',
        'model_id': 'sentence-transformers/all-mpnet-base-v2',
        'parameters_millions': 109,
        'description': 'Best quality, trained on 1B+ pairs'
    },
    {
        'name': 'paraphrase-MiniLM-L6-v2',
        'model_id': 'sentence-transformers/paraphrase-MiniLM-L6-v2',
        'parameters_millions': 22.7,
        'description': 'Optimized for paraphrase detection'
    },
    {
        'name': 'distilbert-base-nli-stsb',
        'model_id': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens',
        'parameters_millions': 66,
        'description': 'DistilBERT-based, balanced performance'
    },
    {
        'name': 'all-distilroberta-v1',
        'model_id': 'sentence-transformers/all-distilroberta-v1',
        'parameters_millions': 82,
        'description': 'RoBERTa-based architecture'
    },
    {
        'name': 'paraphrase-multilingual-MiniLM',
        'model_id': 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
        'parameters_millions': 118,
        'description': 'Supports 50+ languages'
    }
]

# TOPSIS Criteria Configuration
CRITERIA = {
    'spearman_correlation': {
        'weight': 0.30,
        'type': 'benefit',  # higher is better
        'description': 'Spearman rank correlation with ground truth'
    },
    'pearson_correlation': {
        'weight': 0.25,
        'type': 'benefit',  # higher is better
        'description': 'Pearson correlation with ground truth'
    },
    'inference_speed': {
        'weight': 0.20,
        'type': 'benefit',  # higher is better (more sentences/sec)
        'description': 'Sentences processed per second'
    },
    'model_size': {
        'weight': 0.15,
        'type': 'cost',  # lower is better
        'description': 'Number of parameters in millions'
    },
    'memory_usage': {
        'weight': 0.10,
        'type': 'cost',  # lower is better
        'description': 'Memory usage in MB'
    }
}

# Dataset configuration
DATASET_CONFIG = {
    'name': 'STS Benchmark',
    'source': 'stsb_multi_mt',
    'subset': 'en',
    'test_size': None  # Use full test set
}

def get_model_list():
    """Return list of model IDs"""
    return [model['model_id'] for model in MODELS]

def get_model_names():
    """Return list of model names"""
    return [model['name'] for model in MODELS]

def get_criteria_names():
    """Return list of criteria names"""
    return list(CRITERIA.keys())

def get_criteria_weights():
    """Return list of criteria weights"""
    return [CRITERIA[key]['weight'] for key in CRITERIA.keys()]

def get_criteria_types():
    """Return list of criteria types (benefit/cost)"""
    return [CRITERIA[key]['type'] for key in CRITERIA.keys()]

if __name__ == "__main__":
    print("="*60)
    print("MODEL CONFIGURATION")
    print("="*60)
    print(f"\nTotal Models: {len(MODELS)}")
    for i, model in enumerate(MODELS, 1):
        print(f"\n{i}. {model['name']}")
        print(f"   Model ID: {model['model_id']}")
        print(f"   Parameters: {model['parameters_millions']}M")
        print(f"   Description: {model['description']}")
    
    print("\n" + "="*60)
    print("TOPSIS CRITERIA")
    print("="*60)
    for criterion, config in CRITERIA.items():
        print(f"\n{criterion}:")
        print(f"   Weight: {config['weight']*100}%")
        print(f"   Type: {config['type'].upper()}")
        print(f"   Description: {config['description']}")
    
    print(f"\n{'='*60}")
    print(f"Total Weight: {sum(get_criteria_weights())*100}%")
    print("="*60)

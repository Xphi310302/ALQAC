# ALQAC Source Code

This directory contains refactored Python scripts from the original Jupyter notebooks for the ALQAC project.

## Scripts

- **preprocessing.py**: Processes the raw ALQAC law data and generates individual JSON files per topic in the output folder
- **indexing.py**: Sets up embedding models, indexes the processed data into a vector database, and includes evaluation functionality

## Setup

1. Ensure you have all required dependencies installed
2. Make sure the `law_search` module is in your Python path
3. The scripts expect the data folder `ALQAC_2025_data` to be available in the parent directory

## Usage

### Step 1: Preprocessing

```bash
python preprocessing.py
```

This will:
- Load the law data from `../ALQAC_2025_data/alqac25_law.json`
- Process the data by topic
- Save individual JSON files to the `output` directory

### Step 2: Indexing and Evaluation

```bash
python indexing.py
```

This will:
- Set up the embedding model
- Index all processed documents from the `output` directory
- Run evaluation on the test dataset
- Save evaluation results to the `evaluation` directory

## Configuration

Both scripts have configurable paths at the top of each file. You may need to adjust these paths based on your directory structure.

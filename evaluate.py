import os
import json
from typing import Any, Dict, List

# --- Data Structures and Helpers from Reference ---

class Article:
    """Represents a legal article with a specific ID and law ID."""
    pattern = "{}-->{}-->{}"

    def __init__(self, a_id, l_id, content="") -> None:
        self.a_id = a_id
        self.l_id = l_id
        self.content = content

    def __str__(self) -> str:
        return self.pattern.format(self.l_id, self.a_id, self.content)

    @classmethod
    def from_string(cls, str_in):
        info = str_in.split(cls.pattern.format("", "a", "").split("a")[0])
        return cls(info[1], info[0], info[2]) # Corrected order based on pattern

    def get_id(self):
        return self.pattern.format(self.l_id, self.a_id, "")
    
    def __eq__(self, other):
        if not isinstance(other, Article) or self.a_id is None or self.l_id is None:
            return False
        return self.a_id == other.a_id and self.l_id == other.l_id
    
    def __hash__(self):
        return hash((self.a_id, self.l_id))

def f_score(p, r, beta=1):
    """Calculates the F-score."""
    y = (beta * beta * p + r)
    return (1 + beta * beta) * p * r / y if y != 0 else 0.0

def avg_list(arr):
    """Calculates the average of a list."""
    return sum(arr) / len(arr) if arr else 0.0

# --- Evaluation Logic ---

def evaluate_task1(prediction_file, test_data):
    """Evaluates Task 1: Relevant Article Retrieval."""
    print("\nRunning evaluation for Task 1...")
    
    try:
        with open(prediction_file, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {prediction_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {prediction_file}")
        return

    prediction_dict = {item["question_id"]: item for item in prediction_data}
    
    all_results = {'detail': {}}
    
    for q_id, q_info in test_data.items():
        pred_articles_raw = prediction_dict.get(q_id, {}).get('relevant_articles', [])
        a_prediction = {Article(item['article_id'], item['law_id']) for item in pred_articles_raw}
        
        gold_articles_raw = q_info.get('relevant_articles', [])
        a_gold = {Article(item['article_id'], item['law_id']) for item in gold_articles_raw}
        
        count_true = len(a_prediction.intersection(a_gold))
        count_gold = len(a_gold)
        count_prediction = len(a_prediction)
        
        p = count_true / count_prediction if count_prediction > 0 else 0
        r = count_true / count_gold if count_gold > 0 else 0
        f2 = f_score(p, r, beta=2)

        all_results['detail'][q_id] = {
            'gold': [str(art) for art in a_gold],
            'pred': [str(art) for art in a_prediction],
            'p': p,
            'r': r,
            'f2': f2,
        }
        
    all_results['p'] = avg_list([q['p'] for q in all_results['detail'].values()])
    all_results['r'] = avg_list([q['r'] for q in all_results['detail'].values()])
    all_results['f2'] = avg_list([q['f2'] for q in all_results['detail'].values()])

    print(f"Precision: {all_results['p']:.4f}, Recall: {all_results['r']:.4f} => Macro F2-score: {all_results['f2']:.4f}")


def evaluate_task2(prediction_file, test_data):
    """Evaluates Task 2: Yes/No/Other Answer Prediction.""""
    print("\nRunning evaluation for Task 2...")

    try:
        with open(prediction_file, 'r', encoding='utf-8') as f:
            prediction_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Prediction file not found at {prediction_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {prediction_file}")
        return
        
    prediction_dict = {item["question_id"]: item for item in prediction_data}
    
    count_true = 0
    total = 0
    
    for q_id, q_info in test_data.items():
        # Only evaluate on questions that have a 'yes/no' type answer in gold set
        if 'answer' in q_info:
            total += 1
            l_prediction = prediction_dict.get(q_id, {}).get('answer')
            l_gold = q_info['answer']
            if l_prediction is not None and l_gold == l_prediction:
                count_true += 1
                
    if total == 0:
        print("No questions with 'answer' field found in test data for Task 2 evaluation.")
        return

    accuracy = count_true / total
    print(f"Accuracy = {accuracy:.4f} ({count_true}/{total})")


def main():
    """Main function to run the evaluation."""
    
    # --- Load Ground Truth Data ---
    test_data_path = "ALQAC_2025_data/alqac25_train.json" # As used in the notebook
    print(f"Loading test data from: {test_data_path}")
    
    try:
        with open(test_data_path, 'r', encoding='utf-8') as f:
            valid_data = json.load(f)
        test_data = {item["question_id"]: item for item in valid_data}
        print(f"Loaded {len(test_data)} examples from test data.")
    except FileNotFoundError:
        print(f"Error: Test data file not found at {test_data_path}")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {test_data_path}")
        return

    # --- Define Prediction File Paths ---
    # Assuming predictions are saved in an 'output' directory
    task1_preds = "output/result_task_1.json"
    task2_preds = "output/result_task_2.json"

    # --- Run Evaluations ---
    if os.path.isfile(task1_preds):
        evaluate_task1(task1_preds, test_data)
    else:
        print(f"\nSkipping Task 1: Prediction file not found at '{task1_preds}'")

    if os.path.isfile(task2_preds):
        evaluate_task2(task2_preds, test_data)
    else:
        print(f"\nSkipping Task 2: Prediction file not found at '{task2_preds}'")


if __name__ == "__main__":
    main()

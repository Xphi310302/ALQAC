#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Task 2 Baseline Script for ALQAC 2025
Converted from task2_baseline.ipynb
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("task2_baseline.log"), logging.StreamHandler()],
)

logger = logging.getLogger(__name__)

# Load data paths - can be modified via arguments if needed
DATA_DIR = Path("../ALQAC_2025_data")
OUTPUT_DIR = Path("../evaluation_task2")

# Create output directory if it doesn't exist
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Constants for output formatting
OUTPUT_OPTIONS = """Đọc đoạn văn dưới đây và đánh giá xem tuyên bố có đúng hay sai. Hãy suy nghĩ cẩn thận và phân tích kỹ. Câu trả lời của bạn phải là một trong hai lựa chọn: "Đúng" hoặc "Sai".

"""

OUTPUT_ESSAY = """Đọc đoạn văn dưới đây và trả lời câu hỏi. Hãy trả lời ngắn gọn, súc tích, và đi thẳng vào vấn đề.

"""

OUTPUT_CHOICES = """Đọc đoạn văn dưới đây và chọn đáp án đúng cho câu hỏi. Hãy suy nghĩ cẩn thận và phân tích từng lựa chọn. Câu trả lời của bạn phải là một trong các lựa chọn: A, B, C, hoặc D.

"""


# Function to load prompt templates from JSON files
def load_prompt_templates(template_type="default"):
    """
    Load prompt templates from JSON files based on template_type

    Args:
        template_type: Type of templates to load ('default', 'tot', or 'vn')

    Returns:
        Dictionary containing prompt templates for different question types
    """
    # Default templates (hardcoded as fallback)
    default_templates = {
        "truefalse": [
            """{premise}

Tuyên bố: {hypothesis}

Tuyên bố trên đúng hay sai dựa trên đoạn văn bản? Hãy suy nghĩ cẩn thận và phân tích thông tin trong văn bản để đưa ra kết luận chính xác.
"""
        ],
        "essay": [
            """{premise}

Câu hỏi: {hypothesis}

Dựa vào đoạn văn bản, hãy trả lời câu hỏi trên một cách ngắn gọn và chính xác.
"""
        ],
        "options": [
            """{premise}

Câu hỏi: {hypothesis}

A. {choices[A]}
B. {choices[B]}
C. {choices[C]}
D. {choices[D]}

Dựa vào đoạn văn bản, đâu là đáp án đúng cho câu hỏi trên? Hãy suy nghĩ cẩn thận và phân tích từng lựa chọn.
"""
        ],
    }

    if template_type == "default":
        logger.info("Using default prompt templates")
        return default_templates

    try:
        if template_type == "tot":
            file_path = Path("../prompts_TOT.json")
            logger.info(f"Loading TOT prompt templates from {file_path}")
        elif template_type == "vn":
            file_path = Path("../prompts_vn.json")
            logger.info(f"Loading VN prompt templates from {file_path}")
        else:
            logger.warning(
                f"Unknown template type '{template_type}'. Using default templates."
            )
            return default_templates

        with open(file_path, "r", encoding="utf-8") as f:
            templates = json.load(f)
            logger.info(f"Successfully loaded {len(templates)} template types")
            return templates

    except Exception as e:
        logger.error(f"Error loading prompt templates: {e}. Using default templates.")
        return default_templates


# Initialize prompts_template with default value
prompts_template = load_prompt_templates("default")


def load_corpus(file_path=DATA_DIR / "alqac25_corpus.json"):
    """
    Load the corpus of legal texts from the specified file

    Args:
        file_path: Path to the corpus JSON file

    Returns:
        List of law documents with articles
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            corpus = json.load(f)
        return corpus
    except Exception as e:
        logger.error(f"Error loading corpus: {e}")
        return []


def load_questions(file_path=DATA_DIR / "alqac25_train.json", limit=None):
    """
    Load the questions from the specified file

    Args:
        file_path: Path to the questions JSON file
        limit: Optional limit on the number of questions to load

    Returns:
        List of questions
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            questions = json.load(f)

        if limit is not None:
            questions = questions[:limit]

        return questions
    except Exception as e:
        logger.error(f"Error loading questions: {e}")
        return []


def build_prompts(
    example,
    corpus,
    prompts_template=prompts_template,
    prompt_index=0,
    OUTPUT_OPTIONS=OUTPUT_OPTIONS,
    OUTPUT_ESSAY=OUTPUT_ESSAY,
    OUTPUT_CHOICES=OUTPUT_CHOICES,
):
    """
    Build prompts for different question types by combining relevant articles and question text

    Args:
        example: List of question dictionaries
        corpus: The legal corpus to extract articles from
        prompts_template: Dictionary of prompt templates for each question type
        prompt_index: Index of the prompt template to use
        OUTPUT_OPTIONS: Prefix text for multiple choice questions
        OUTPUT_ESSAY: Prefix text for essay questions
        OUTPUT_CHOICES: Prefix text for true/false questions

    Returns:
        List of prompt dictionaries
    """
    prompts = []
    for sample in example:
        qt = sample["question_type"].lower()
        premise = ""
        try:
            for article in sample["relevant_articles"]:
                law_id = article["law_id"]
                article_id = article["article_id"]
                law_corpus = next((law for law in corpus if law["id"] == law_id), None)
                if law_corpus is None:
                    raise KeyError(f"law_id '{law_id}' not found in corpus")
                article_corpus = next(
                    (a for a in law_corpus["articles"] if a["id"] == str(article_id)),
                    None,
                )
                if article_corpus is None:
                    raise KeyError(
                        f"article_id '{article_id}' not found in law_id '{law_id}'"
                    )
                premise += f"\n{article_corpus['text']}"
            if qt == "đúng/sai":
                prompt = prompts_template["truefalse"][prompt_index].format(
                    premise=premise, hypothesis=sample["text"]
                )
                prompt = OUTPUT_OPTIONS + prompt
            elif qt == "tự luận":
                prompt = prompts_template["essay"][prompt_index].format(
                    premise=premise, hypothesis=sample["text"]
                )
                prompt = OUTPUT_ESSAY + prompt
            elif qt == "trắc nghiệm":
                prompt = prompts_template["options"][prompt_index].format(
                    premise=premise,
                    hypothesis=sample["text"],
                    choices={
                        "A": sample["choices"]["A"],
                        "B": sample["choices"]["B"],
                        "C": sample["choices"]["C"],
                        "D": sample["choices"]["D"],
                    },
                )
                prompt = OUTPUT_CHOICES + prompt
            prompts.append(
                {
                    "question_id": sample["question_id"],
                    "prompt": prompt,
                    "question_type": qt,
                    "question": sample["text"],
                }
            )
        except KeyError as e:
            logger.error(
                f"Error processing sample {sample.get('question_id', '')}: {e}"
            )
            logger.debug(f"Problem with prompt: {prompt}")
    return prompts


def extract_output(output, question_type, question, llm=None, max_retries=2):
    """
    Extract and clean answers from the raw output of the language model

    Args:
        output: Raw output text from LLM
        question_type: Type of question (đúng/sai, tự luận, trắc nghiệm)
        question: Question text
        llm: LLM client for potential follow-up processing
        max_retries: Maximum number of retries for extraction

    Returns:
        Cleaned answer string
    """

    def extract_truefalse(text):
        lowered = text.lower().replace(".", "").strip()
        if "đúng" in lowered:
            return "Đúng"
        if "sai" in lowered:
            return "Sai"
        return lowered.strip().replace(".", "")

    def extract_choices(text):
        cleaned = text.strip().replace(".", "")
        for choice in ["A", "B", "C", "D"]:
            if re.match(rf"^\s*{choice}\b", cleaned, re.IGNORECASE):
                return choice
        match = re.search(r"\b([A-D])\b", cleaned, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        return cleaned.strip().replace(".", "")

    def extract_answer(text):
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        match = re.search(
            r"(?:\*\*answer\*\*|```answer)\s*([\s\S]+?)\s*(?:```)?$", text, re.MULTILINE
        )
        answer = match.group(1).strip() if match else text.strip()
        return answer.replace(".", "")

    output = extract_answer(output)
    if question_type.lower() == "đúng/sai":
        return extract_truefalse(output)
    if question_type.lower() == "trắc nghiệm":
        return extract_choices(output)
    if question_type.lower() == "tự luận":
        return output
    return output


def acc_eval_from_list(preds, golds, ignore_case=False):
    """
    Calculate accuracy by comparing predicted answers to gold answers

    Args:
        preds: List of predicted answers
        golds: List of gold/correct answers
        ignore_case: Whether to ignore case when comparing

    Returns:
        Accuracy score as a float
    """
    if len(preds) != len(golds):
        logger.warning(f"Length mismatch: preds {len(preds)} vs golds {len(golds)}")
        return 0.0

    correct = 0
    for p, g in zip(preds, golds):
        if ignore_case:
            if p.lower() == g.lower():
                correct += 1
        else:
            if p == g:
                correct += 1

    return correct / len(preds)


def setup_llm_client():
    """
    Set up the LLM client with error handling

    Returns:
        LLM client or None if error occurs
    """
    try:
        from infer import Client

        llm = Client(endpoint="https://gemma-3-27b-it:lh6OmLRoZP@infer.windsurf.ai/v1")
        return llm
    except Exception as e:
        logger.error(f"Error initializing LLM client: {e}")
        return None


def main(
    examples_path,
    output_path=None,
    limit=None,
    model_name="Qwen2.5-7B-Instruct",
    template_type="default",
):
    """
    Main execution function

    Args:
        examples_path: Path to examples JSON file
        output_path: Path to save output files (default: OUTPUT_DIR)
        limit: Maximum number of examples to process
        model_name: Name of the model to use for inference
        template_type: Type of prompt template to use (default, tot, vn)
    """
    # Set output path to default if not specified
    if output_path is None:
        output_path = OUTPUT_DIR
    else:
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True, parents=True)

    # Set up timestamp for outputs
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load examples, corpus, and templates
    logger.info(f"Loading examples from {examples_path}")
    questions = load_questions(examples_path, limit)
    corpus = load_corpus()

    # Load prompt templates based on specified type
    global prompts_template
    prompts_template = load_prompt_templates(template_type)
    logger.info(f"Using prompt template type: {template_type}")

    # Build prompts
    logger.info("Building prompts...")
    prompts = build_prompts(questions, corpus)
    logger.info(f"Built {len(prompts)} prompts")

    # Set up LLM client
    logger.info("Setting up LLM client...")
    llm = setup_llm_client()
    if llm is None:
        logger.error("Failed to initialize LLM client. Exiting.")
        return

    # Process prompts and get answers
    logger.info("Processing prompts...")
    predictions = []
    model_outputs = []

    for prompt in tqdm(prompts, desc="Processing prompts"):
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                logger.info(
                    f"Processing question {prompt['question_id']} (attempt {attempt + 1})"
                )
                response = llm.complete(
                    prompt=prompt["prompt"], model=model_name, max_tokens=1024
                )

                # Extract answer from model output
                answer = extract_output(
                    response, prompt["question_type"], prompt["question"], llm
                )

                # Store prediction and model output
                predictions.append(
                    {"question_id": prompt["question_id"], "prediction": answer}
                )
                model_outputs.append(
                    {
                        "question_id": prompt["question_id"],
                        "model_output": response,
                        "extracted_answer": answer,
                    }
                )

                logger.info(f"Question: {prompt['question']} - Predicted: {answer}")
                break
            except Exception as e:
                logger.error(
                    f"Error on attempt {attempt + 1} for question {prompt['question_id']}: {e}"
                )
                if attempt == max_attempts - 1:
                    logger.error(
                        f"Failed all {max_attempts} attempts for question {prompt['question_id']}"
                    )
                    predictions.append(
                        {"question_id": prompt["question_id"], "prediction": "ERROR"}
                    )
                    model_outputs.append(
                        {
                            "question_id": prompt["question_id"],
                            "model_output": f"ERROR: {str(e)}",
                            "extracted_answer": "ERROR",
                        }
                    )

    # Calculate accuracy
    logger.info("Calculating accuracy...")
    gold_answers = []
    pred_answers = []

    for q in questions:
        gold_answers.append(q["answer"])

        # Find corresponding prediction
        pred = next(
            (
                p["prediction"]
                for p in predictions
                if p["question_id"] == q["question_id"]
            ),
            "ERROR",
        )
        pred_answers.append(pred)

    accuracy = acc_eval_from_list(pred_answers, gold_answers)
    logger.info(f"Accuracy: {accuracy:.4f}")

    # Save results
    logger.info("Saving results...")

    # Save predictions
    predictions_file = output_path / f"predictions_{timestamp}.json"
    with open(predictions_file, "w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    # Save model outputs
    outputs_file = output_path / f"outputs_{timestamp}.json"
    with open(outputs_file, "w", encoding="utf-8") as f:
        json.dump(model_outputs, f, ensure_ascii=False, indent=2)

    # Save evaluation results
    eval_results = {
        "accuracy": accuracy,
        "timestamp": timestamp,
        "num_questions": len(questions),
        "num_predictions": len(predictions),
    }

    eval_file = output_path / f"eval_{timestamp}.json"
    with open(eval_file, "w", encoding="utf-8") as f:
        json.dump(eval_results, f, ensure_ascii=False, indent=2)

    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the Task 2 baseline model")
    parser.add_argument(
        "--examples",
        type=str,
        default=str(DATA_DIR / "alqac25_train.json"),
        help="Path to the examples JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to process",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemma-3-27b-it",
        help="Model name to use for inference",
    )
    parser.add_argument(
        "--template-type",
        type=str,
        choices=["default", "tot", "vn"],
        default="default",
        help="Type of prompt template to use (default, tot, or vn)",
    )

    args = parser.parse_args()
    main(args.examples, args.output, args.limit, args.model, args.template_type)

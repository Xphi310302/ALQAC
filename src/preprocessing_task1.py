#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing script for ALQAC data.
Processes JSON law data and splits it by topic into individual files.
"""

import json
import os


def load_law_data(file_path):
    """
    Load the law data from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing law data

    Returns:
        dict: Loaded JSON data
    """
    with open(file_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    return json_data


def process_data(json_data):
    """
    Process the loaded JSON data into a structured format.

    Args:
        json_data (dict): Loaded law data

    Returns:
        dict: Processed data organized by topics
    """
    data = dict()
    for topic_json in json_data:
        topic = topic_json["id"]
        articles = topic_json["articles"]
        data[topic] = [
            {idx + 1: article["text"]} for idx, article in enumerate(articles)
        ]
    return data


def save_processed_data(data, output_folder="output"):
    """
    Save the processed data as separate JSON files by topic.

    Args:
        data (dict): Processed data organized by topics
        output_folder (str): Folder to save the output files
    """
    os.makedirs(output_folder, exist_ok=True)

    for key in data.keys():
        output_path = os.path.join(output_folder, f"{key}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data[key], f, ensure_ascii=False, indent=2)


def main():
    """Main execution function."""
    # Define input path
    input_path = "../ALQAC_2025_data/alqac25_law.json"

    # Load data
    json_data = load_law_data(input_path)

    # Process data
    data = process_data(json_data)

    # Print all topics/documents
    print("Available topics/documents:")
    for topic in sorted(data.keys()):
        print(f"- {topic}")

    # Save processed data
    save_processed_data(data)
    print(f"Processed data saved to 'output' directory")


if __name__ == "__main__":
    main()

"""
This script reads and retrieves all text from a root data folder containing
many text files in different subdirs, concatenating their contents into a consolidated output text file.
It then cleans the file to remove leading whitespace.
"""

import os
from typing import List


# Recursively walk through `data_folder`, reading text and writing to the `output_file`
# With options for specifying subdirs of certain literary form(s) and/or author(s)
# By default, all forms and authors are included in the output file
def retrieve_text(
    data_folder: str,
    output_file: str,
    categories: List[str] = ['poems', 'prose', 'publicism'],
    specific_authors: List[str] = None
) -> None:
    # Walk through the data folder and concatenate all .txt files into an output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(data_folder):
            # Modify dirs in place so that only text from selected categories are read
            if root == data_folder:
                dirs[:] = [d for d in dirs if d in categories]
            elif specific_authors:
                # If the argument was given, only include works from the specified authors
                dirs[:] = [d for d in dirs if d in specific_authors]
            for file in files:
                if file.endswith('txt'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        print(f"Reading {file_path}")
                        text = infile.read()
                        outfile.write(text)
                        outfile.write('\n')


# Remove leading whitespaces from all lines in input text file
def clean_text(data_file: str, output_file: str) -> None:
    with open(data_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    cleaned_lines = [line.lstrip() if line.strip() else line for line in lines]

    with open(output_file, 'w') as outfile:
        outfile.writelines(cleaned_lines)


if __name__ == '__main__':
    data_folder = 'data/tiny-russian-lit'
    retrieve_text(data_folder, 'data/tiny-russian-lit/tiny_russian_lit.txt')
    clean_text('data/tiny-russian-lit/tiny_russian_lit.txt',
               'data/tiny-russian-lit/cleaned_tiny_russian_lit.txt')

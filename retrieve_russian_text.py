"""
This script reads and retrieves all text from a root data folder containing
many text files in different subdirs, concatenating their contents into a consolidated output text file.
It then cleans the file to remove leading whitespace.
"""

import os
import re
from typing import List

# Remove all instances of these characters/patterns when cleaning data file
EXCLUDE = r'_|`|\||~|§|\*|\d+|#||	||	|«|»|½|Ќ|€|№|¾|=|\(|\)|\[|\]|\{|\}|°|<|>|“|”|„|"|…|%|\''
# Exclude lines with these non-Russian characters in the cleaned data file
FRENCH = set(
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghjklmnopqrstuvwxyzїÇÉÊÔÜßàáâäçèéêëíîïòóôöùúûüýœ')
GREEK = set('ΕΘΚΠΣάέήίαβγδεηικλμνοπρςστυφψωόύώϑѣἀἁἃἄἈἐἔἡἴἷἹὁὄὐὑὰὴὶὸᾶῆ῎ῖῦῶῷ')


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


# Remove leading whitespaces from all lines in input text file and exclude garbage characters
# Remove lines with french or greek characters if russ_only is True
# Keep blank lines if preserve_blank is True
def clean_text(data_file: str, output_file: str, russ_only: bool = True, preserve_blank: bool = False) -> None:
    with open(data_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
    cleaned_lines = []
    for line in lines:
        if preserve_blank and not line.strip():
            cleaned_lines.append(line)
            continue
        # Remove leading whitespace and leading "- "
        line = line.lstrip()
        if line.startswith("- ") or line.startswith("– ") or line.startswith("— "):
            line = line[2:]
        elif line.startswith("-- "):
            line = line[3:]
        # Don't add the line if it contains non-Russian
        if russ_only and not_russ(line):
            continue
        # Remove all instances of patterns in EXCLUDE
        clean_line = re.sub(EXCLUDE, '', line)
        # Replace '//' with '. '
        clean_line = re.sub(r'(?<!/)//(?!/)', '. ', clean_line)
        # Replace '/' with ''
        clean_line = re.sub(r'(?<!/)/(?!/)', '', clean_line)

        cleaned_lines.append(clean_line)

    with open(output_file, 'w') as outfile:
        outfile.writelines(cleaned_lines)


# Check whether the line of text contains any French or Greek
def not_russ(line):
    for c in line:
        if c in FRENCH or c in GREEK:
            return True
    return False


if __name__ == '__main__':
    data_folder = 'data/tiny-russian-lit'
    # retrieve_text(data_folder, 'data/tiny-russian-lit/tiny_russian_lit.txt')
    clean_text('data/tiny-russian-lit/tiny_russian_lit.txt',
               'data/tiny-russian-lit/very_clean_tiny_russian_lit.txt')

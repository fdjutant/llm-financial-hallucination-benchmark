import os
import glob
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from lxml import etree
import pandas as pd
from bs4 import BeautifulSoup

def save_blocks_to_file(html_filepath, tags, output_file="all_blocks.xml"):
    """
    Extracts specified tags from an HTML file and saves them into a single file.

    Args:
        html_filepath (str or Path): Path to the HTML file.
        tags (list): List of tags to extract.
        output_file (str): Path to the output file where blocks will be saved.

    Example:
        TAGS = ["ix:header", "ix:references", "link:schemaref", "ix:resources", "xbrli:unit"]
        save_blocks_to_file(html_filepaths[0], TAGS)
    """
    html = open(html_filepath, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml-xml")  # IMPORTANT: XML PARSER PRESERVES NAMESPACES

    blocks = {}
    for tag in tags:
        found = soup.find_all(tag)
        blocks[tag] = [str(x) for x in found]  # keep tag + its content

    with open(output_file, "w", encoding="utf-8") as f:
        for tag, content in blocks.items():
            for i, block in enumerate(content):
                f.write(f"<!-- {tag.upper()} {i} -->\n")
                f.write(block + "\n\n")

def extract_text_from_tags(html_filepath, tags):
    """
    Extracts text content from specified tags in an HTML file.

    Args:
        html_filepath (str or Path): Path to the HTML file.
        tags (list): List of tags to extract text from.

    Returns:
        dict: A dictionary where keys are tag names and values are lists of extracted text.

    Example:
        html_filepaths = [projectRoot / "data" / "raw" / "FCA" / "GSK" / "GSK_2024.html"]
        tags = ["ix:header", "ix:references", "ix:resources", "xbrli:unit", "link:schemaref"]
        extracted_text = extract_text_from_tags(html_filepaths[0], tags)
        print(extracted_text["ix:header"])
    """
    html = open(html_filepath, encoding="utf-8", errors="ignore").read()
    soup = BeautifulSoup(html, "lxml-xml")

    extracted_text = {}
    for tag in tags:
        nodes = soup.find_all(tag)
        extracted_text[tag] = [n.get_text(" ", strip=True) for n in nodes]

    return extracted_text

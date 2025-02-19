import json
import random
import xml.etree.ElementTree as ET
from cleantext import clean
from transformers import AutoTokenizer

random.seed(42)

POST_MAX_LEN = 2048
MAX_ENTRIES_PER_CHUNK = 100000
MODEL = "/mnt/gpu-fastdata/eliseo/dsm5/models/Llama-3.1-8B-Instruct-abliterated/"


tokenizer = AutoTokenizer.from_pretrained(MODEL)


def custom_clean(text):
    """
    Cleans the input text based on specified rules for handling URLs, emails,
    phone numbers, and other specific elements, using the 'clean' function.

    Parameters:
    text (str): The text to be cleaned.

    Returns:
    str: The cleaned text.
    """
    return clean(
        text,
        fix_unicode=True,
        to_ascii=False,
        lower=False,
        no_line_breaks=False,
        no_urls=True,
        no_emails=True,
        no_phone_numbers=True,
        no_numbers=False,
        no_digits=False,
        no_currency_symbols=False,
        no_punct=False,
        replace_with_punct="",
        replace_with_url="<URL>",
        replace_with_email="<EMAIL>",
        replace_with_phone_number="<PHONE>",
        replace_with_number="<NUMBER>",
        replace_with_digit="0",
        replace_with_currency_symbol="<CUR>",
        lang="en",
    )


def load_golden_data(filepath):
    """
    Loads the 'golden truth' data from a text file.

    Parameters:
    filepath (str): The path to the 'golden truth' text file.

    Returns:
    list: A list of dictionaries, each containing an 'id' and 'depressed' status.
    """
    golden = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            golden.append({"id": parts[0], "depressed": bool(int(parts[1]))})
    return golden


def trim_text(text, max_len=POST_MAX_LEN):
    """
    Trims the input text to ensure it fits within a specified token limit.

    This function encodes the input text into tokens using a specified tokenizer and
    then checks if the token count exceeds the defined maximum length (`max_len`).
    If the tokenized text exceeds `max_len`, it truncates the token list to this limit.
    Finally, the truncated tokens are decoded back into text, ensuring the output
    remains within the token limit.

    Parameters:
    text (str): The input text to be trimmed.
    max_len (int): The maximum number of tokens allowed for the text. Defaults to POST_MAX_LEN.

    Returns:
    str: The trimmed text, decoded back from the truncated tokens.
    """
    # Encode the input text into tokens
    tokens = tokenizer.encode(text, return_tensors="pt")[0]

    # Truncate the tokens if they exceed max_len
    tokens = tokens[:max_len] if tokens.shape[0] > max_len else tokens

    # Decode and return the truncated token sequence back into text
    return tokenizer.decode(tokens, skip_special_tokens=True)


def parse_user_data(user_id):
    """
    Parses and cleans the XML data for a given user, extracting and formatting each writing entry.

    Parameters:
    user_id (str): The ID of the user.

    Returns:
    dict: A dictionary containing the user's ID, depression status, and posts.
    """
    tree = ET.parse(f"data/raw/erisk_t2_2022/test/datos/{user_id}.xml")
    root = tree.getroot()
    assert root.find("ID").text == user_id  # Ensure the ID in XML matches the user ID

    user_data = []

    for writing in root.findall("WRITING"):
        title = (
            custom_clean(writing.find("TITLE").text)
            if writing.find("TITLE") is not None
            else None
        )
        # date = (
        #     custom_clean(writing.find("DATE").text)
        #     if writing.find("DATE") is not None
        #     else None
        # )
        text = (
            custom_clean(writing.find("TEXT").text)
            if writing.find("TEXT") is not None
            else None
        )

        # Format each post
        post = (
            # f"Date: {date}\n"
            f""
            + (f'Title:\n\n"{trim_text(title, POST_MAX_LEN)}"\n' if title else "")
            + (f'Text:\n\n"{trim_text(text, POST_MAX_LEN)}"\n' if text else "")
        )

        user_data.append(
            {
                "problem": post.strip(),
                "subject": user_id,
            }
        )

    return user_data


def save_to_jsonl_chunked(data, output_dir, max_entries_per_file):
    """
    Saves the processed user data into multiple JSONL files with a maximum of N entries each.

    Parameters:
    data (list): List of dictionaries containing processed user data.
    output_dir (str): Directory to save the JSONL files.
    max_entries_per_file (int): Maximum number of entries per JSON file.
    """
    for i in range(0, len(data), max_entries_per_file):
        chunk = data[i : i + max_entries_per_file]
        output_filepath = (
            f"{output_dir}/erisk_t2_2022_chunk_{i // max_entries_per_file + 1}.jsonl"
        )
        with open(output_filepath, "w") as f:
            for entry in chunk:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"Saved chunk {i // max_entries_per_file + 1} to {output_filepath}")


def main():
    """
    Main function that orchestrates loading golden data, parsing user XML data,
    and saving the processed output as multiple JSON files with a maximum of N entries each.
    """
    golden = load_golden_data("data/raw/erisk_t2_2022/test/risk_golden_truth.txt")

    random.shuffle(golden)

    user_data = []
    output_dir = "data/processed/erisk_t2_2022_chunked"

    for i, subject in enumerate(golden):
        print(f"{i+1}/{len(golden)}")
        user_data.extend(parse_user_data(subject["id"]))

    save_to_jsonl_chunked(user_data, output_dir, MAX_ENTRIES_PER_CHUNK)


if __name__ == "__main__":
    main()

import re
import os
import numpy as np

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def is_competition_downloaded(dir_path, competition_name, is_wsm):
    """ Checks if a competition file is downloaded.
        Args:
            dir_path (str): The directory path where the competition file is located (results or events directory)
            competition_name (str): The name of the competition.
            is_wsm (bool): Indicates if it is a World's Strongest Man competition.
    """
    path = f'{dir_path}/{competition_name}.csv'
    if is_wsm:
        path = f'{dir_path}/finals/{competition_name}.csv'

    return os.path.isfile(path)


def get_floats(txt):
    return [float(x) for x in txt.split(' ') if is_float(x)]


def get_ints(txt):
    return [int(x) for x in re.findall(r"\d+", txt)]


def remove_numbers(txt):
    """Remove numbers and excess spaces from string"""
    return re.sub(r'[0-9]+', '', txt).strip()


def remove_punctuation(txt):
    try:
        return re.sub(r'[^\w\s]', '', txt)
    except TypeError:
        return ''


def remove_braces(txt):
    """Remove brackets, braces without removing text inside."""
    try:
        return re.sub(r"[\([{})\]]", "", txt)
    except TypeError:
        return ''


def remove_text_inside_braces(txt):
    """Remove brackets, braces with text inside."""
    try:
        return re.sub(r"[\(\[].*?[\)\]]", "", txt)
    except TypeError:
        return ''


def get_text_inside_braces(txt):
    """Get text inside single braces only"""
    if not isinstance(txt, str):
        return ''
    end_index = txt.find(")") if txt.find(")") != -1 else len(txt)
    return txt[txt.find("(") + 1: end_index]


def get_float_inside_braces(txt):
    """Get float inside single braces only"""
    return to_float(get_text_inside_braces(txt))


def is_nan(x):
    return x != x


def is_float(num):
    try:
        float(num)
        return True
    except (ValueError, TypeError):
        return False


def is_int(num):
    try:
        int(num)
        return True
    except (ValueError, TypeError):
        return False


def to_int(s, default=np.nan):
    """Convert string to int safely."""
    return float(s) if is_int(s) else default


def to_float(s, default=np.nan):
    """Convert string to float safely."""
    return float(s) if is_float(s) else default


def filter_str(s, word_list):
    """Remove words from string that are in the word list and keep everything else."""
    return ' '.join([word for word in s.split(' ') if word.lower() not in word_list]).strip()


def reverse_filter_str(s, word_list):
    "Only keep the words in string that are in the word list and remove everything else."
    return ' '.join([word for word in remove_punctuation(s).split(' ') if word.lower() in word_list]).strip()


def flatten_dict(map: dict) -> dict:
    """Flattens a dictionary by swapping keys and values.

    Args:
        map (dict): A dictionary to be flattened.
    Returns:
        dict: A new dictionary with keys and values swapped.
    Examples:
        >>> flatten_dict({'a': [1, 2], 'b': [3, 4]})
        {1: 'a', 2: 'a', 3: 'b', 4: 'b'}
    """

    flattened_dict = {}
    for key, values in map.items():
        for value in values:
            flattened_dict[value] = key
    return flattened_dict

def get_text_after_word(s, word):
    """First split by the word and then get the first string after the word that we split on."""
    if word not in s:
        return ''
    return split_s[0].strip() if (split_s := s.split(word)[1].split()) else ''


def get_float_before_word(s, word):
    """First split by the word and then get the first string before the word that we split on. """
    word_adjusted = f' {word}'
    if word_adjusted not in s:
        return np.nan
    return to_float(split_s[-1].strip(), default=np.nan) if (split_s := s.split(word_adjusted)[0].split()) else np.nan


def get_int_before_word(s, word):
    """First split by the word and then get the first word before the word that we split on. [-2] is there because split() adds an empty string at the end"""
    word_adjusted = f' {word}'
    if word_adjusted not in s:
        return np.nan
    return to_int(split_s[-1].strip(), default=np.nan) if (split_s := s.split(word_adjusted)[0].split()) else np.nan


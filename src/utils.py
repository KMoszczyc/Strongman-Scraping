import re
import os

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
    return re.sub(r'[^\w\s]', '', txt) if isinstance(txt, str) else ''


def remove_braces(txt):
    return re.sub(r"[(\[].*?[)\]]", "", txt) if isinstance(txt, str) else ''


def remove_text_inside_braces(txt):
    return re.sub(r"[\(\[].*?[\)\]]", "", txt) if isinstance(txt, str) else ''


def get_text_inside_braces(txt):
    return txt[txt.find("(") + 1:txt.find(")")]


def get_float_inside_braces(txt):
    return to_float(txt[txt.find("(") + 1:txt.find(")")])


def is_nan(x):
    return x != x


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def is_int(num):
    try:
        int(num)
        return True
    except ValueError:
        return False


def to_int(s, default=''):
    """Convert string to int safely."""
    return float(s) if is_int(s) else default


def to_float(s, default=''):
    """Convert string to float safely."""

    return float(s) if is_float(s) else default


def filter_str(s, word_list, exclude=False):
    """If exclude is True remove words in string that are in the word list
        If exclude is False then only keep the words in string that are in the word list"""
    if exclude:
        return ' '.join([word for word in s.split(' ') if word.lower() not in word_list]).strip()
    else:
        return ' '.join([word for word in s.split(' ') if word.lower() in word_list]).strip()

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

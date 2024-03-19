import pytest
import src.utils as utils
import os
from pathlib import Path
import numpy as np

TEST_DIR = "test_dir/"
COMPETITION_NAME="2004 Arnold's Strongest Man"
WSM_COMPETITION_NAME="1995 World's Strongest Man"

event_info1="563 kg / 11 meters / 30 second time limit / Straps allowed"
event_info2="30 meters / 75 second time limit / 450 kg yoke (15 m), 127 kg keg (15 m) "
event_info3="75 second time limit / 2x 125 kg boxes + 455 kg yoke carry (10 metres each)"

@pytest.mark.parametrize("test_id, path", [("HP_01", f"{TEST_DIR}new_dir/"), ("HP_02", f"{TEST_DIR}new_dir/sub_dir/"), ("EC_01", TEST_DIR)])
def test_create_dir(test_id, path):
    utils.create_dir(path)
    assert os.path.isdir(path), f"Test {test_id}: Directory {path} was not created."

@pytest.mark.parametrize("test_dir, competition_name, is_wsm, expected", [
    # Happy path tests
    pytest.param(TEST_DIR, COMPETITION_NAME, False, True, id="non_wsm_competition"),
    pytest.param(TEST_DIR, WSM_COMPETITION_NAME, True, True, id="wsm_competition"),

    # Edge cases
    pytest.param(TEST_DIR, "non_existing_competition", False, False, id="edge_non_existing_file"),
    pytest.param(TEST_DIR, "non_existing_competition", True, False, id="edge_wsm_non_existing_file"),
    pytest.param("", COMPETITION_NAME, False, False, id="edge_empty_dir_path"),
    pytest.param(TEST_DIR, "", False, False, id="edge_empty_competition_name"),
])
def test_is_competition_downloaded(test_dir, competition_name, is_wsm, expected):
    result = utils.is_competition_downloaded(test_dir, competition_name, is_wsm)
    assert result == expected


@pytest.mark.parametrize("test_input, expected", [
    ("1.0 2.0 3.0", [1.0, 2.0, 3.0]),
    ("0.0 -1.0 1e3", [0.0, -1.0, 1000.0]),
    ("4.5 .76 78.", [4.5, 0.76, 78.0]),
    ("1.0 2.0 three", [1.0, 2.0]),
    ("", []),
])
def test_get_floats(test_input, expected):
    result = utils.get_floats(test_input)
    assert result == expected

@pytest.mark.parametrize("input_text, expected", [
    ("abc123xyz", [123]),
    ("123 456 789", [123, 456, 789]),
    ("00123", [123]),
    ("123abc456", [123, 456]),
    ("1.23", [1, 23]),
    ("123-456", [123, 456]),
    ("", []),
    ("123, 456, 789", [123, 456, 789]),
    ("2147483647", [2147483647]),
    ("-123", [123]),
    ("123_456", [123, 456]),
    (event_info1, [563, 11, 30]),
    (event_info2, [30, 75, 450, 15, 127, 15]),
    (event_info3, [75, 2, 125, 455, 10]),
])
def test_get_ints(input_text, expected):
    result = utils.get_ints(input_text)
    assert result == expected

@pytest.mark.parametrize("input_text, expected_output", [
    ("test123", "test"),
    ("123test", "test"),
    ("test123test", "testtest"),
    ("test", "test"),
    ("123 456 789", ""),
    ("   spaces 123 ", "spaces"),
    ("", ""),
])
def test_remove_numbers(input_text, expected_output):
    result = utils.remove_numbers(input_text)
    assert result == expected_output

@pytest.mark.parametrize("input_text, expected_output", [
    ("Hello, World!", "Hello World"),
    ("No punctuations here", "No punctuations here"),
    ("Special characters !@#$%^&*()+,. are removed", "Special characters  are removed"),
    ("1234567890", "1234567890"),
    ("Tabs\tand\nnewlines\nare\nok", "Tabs\tand\nnewlines\nare\nok"),
    (np.nan, "")
])
def test_remove_punctuation(input_text, expected_output):
    result = utils.remove_punctuation(input_text)
    assert result == expected_output

@pytest.mark.parametrize("input_text, expected_output, test_id", [
    ("no braces", "no braces", "test_id_no_braces"),
    ("text with (parentheses)", "text with ", "test_id_with_parentheses"),
    ("text with [brackets]", "text with ", "test_id_with_brackets"),
    ("(braces) at the start", " at the start", "test_id_start_braces"),
    ("at the end (braces)", "at the end ", "test_id_end_braces"),
    ("(multiple) (sets) of (braces)", "  of ", "test_id_multiple_sets"),
    ("nested (parentheses (like) this)", "nested ", "test_id_nested_parentheses"),
    ("empty () braces", "empty  braces", "test_id_empty_braces"),
    (None, '', "test_id_none_input"),
    (123, '', "test_id_numeric_input"),
    ({"key": "(value)"}, '', "test_id_dict_input"),
    ([1, 2, 3], '', "test_id_list_input"),
    (True, '', "test_id_bool_input"),
     (np.nan, '', "test_id_bool_input")
        ])
def test_remove_braces(input_text, expected_output, test_id):
    result = utils.remove_braces(input_text)
    assert result == expected_output
import pytest
import src.utils as utils
import os
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
    pytest.param("abc123xyz", [123], id="ID-1"),
    pytest.param("123 456 789", [123, 456, 789], id="ID-2"),
    pytest.param("00123", [123], id="ID-3"),
    pytest.param("123abc456", [123, 456], id="ID-4"),
    pytest.param("1.23", [1, 23], id="ID-5"),
    pytest.param("123-456", [123, 456], id="ID-6"),
    pytest.param("", [], id="ID-7"),
    pytest.param("123, 456, 789", [123, 456, 789], id="ID-8"),
    pytest.param("2147483647", [2147483647], id="ID-9"),
    pytest.param("-123", [123], id="ID-10"),
    pytest.param("123_456", [123, 456], id="ID-11"),
    pytest.param(event_info1, [563, 11, 30], id="ID-12"),
    pytest.param(event_info2, [30, 75, 450, 15, 127, 15], id="ID-13"),
    pytest.param(event_info3, [75, 2, 125, 455, 10], id="ID-14"),
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
    pytest.param("Hello, World!", "Hello World", id="ID-1"),
    pytest.param("No punctuations here", "No punctuations here", id="ID-2"),
    pytest.param("Special characters !@#$%^&*()+,. are removed", "Special characters  are removed", id="ID-3"),
    pytest.param("1234567890", "1234567890", id="ID-4"),
    pytest.param("Tabs\tand\nnewlines\nare\nok", "Tabs\tand\nnewlines\nare\nok", id="ID-5"),
    pytest.param(np.nan, "", id="ID-6"),
])
def test_remove_punctuation(input_text, expected_output):
    result = utils.remove_punctuation(input_text)
    assert result == expected_output


@pytest.mark.parametrize("input_text, expected_output", [
    # Happy path tests
    pytest.param("no_braces", "no_braces", id="happy_no_braces"),
    pytest.param("text_with_parentheses(like_this)", "text_with_parentheseslike_this", id="happy_parentheses"),
    pytest.param("text_with_brackets[like_this]", "text_with_bracketslike_this", id="happy_brackets"),
    pytest.param("text_with_braces{like_this}", "text_with_braceslike_this", id="happy_braces"),
    # Edge cases
    pytest.param("(text", "text", id="edge_unclosed_parenthesis"),
    pytest.param("[text", "text", id="edge_unclosed_bracket"),
    pytest.param("{text", "text", id="edge_unclosed_brace"),
    pytest.param("text)", "text", id="edge_unopened_parenthesis"),
    pytest.param("text]", "text", id="edge_unopened_bracket"),
    pytest.param("text}", "text", id="edge_unopened_brace"),
    # Error cases
    pytest.param(None, '', id="error_none_input"),
    pytest.param(123, '', id="error_non_string_input"),
])
def test_remove_braces(input_text, expected_output):
    result = utils.remove_braces(input_text)
    assert result == expected_output


@pytest.mark.parametrize("input_text, expected_output", [
    # Happy path tests
    pytest.param("no braces", "no braces", id="happy_no_braces"),
    pytest.param("text with (parentheses)", "text with ", id="happy_with_parentheses"),
    pytest.param("text with [brackets]", "text with ", id="happy_with_brackets"),
    pytest.param("", "", id="edge_empty_string"),
    pytest.param(None, "", id="error_none"),
    pytest.param(123, "", id="error_integer")
])
def test_remove_text_inside_braces(input_text, expected_output):
    result = utils.remove_text_inside_braces(input_text)
    assert result == expected_output

@pytest.mark.parametrize("input_text, expected_output", [
    # Happy path tests
    pytest.param("(hello)", "hello", id="happy_simple"),
    pytest.param("before(hello)after", "hello", id="happy_with_text_around"),
    pytest.param("(hello)(world)", "hello", id="happy_multiple_braces"),
    # Edge cases
    pytest.param("()", "", id="edge_empty_braces"),
    pytest.param("(a)", "a", id="edge_single_char"),
    pytest.param("( )", " ", id="edge_space_inside"),
    pytest.param("(hello world", "hello world", id="edge_missing_closing_brace"),
    pytest.param("hello world)", "hello world", id="edge_missing_opening_brace"),
    # Error cases
    pytest.param(None, "", id="error_none_input"),
    pytest.param(123, "", id="error_integer"),
    pytest.param("", "", id="error_empty_string"),
])
def test_get_text_inside_braces_happy_path(input_text, expected_output):
    result = utils.get_text_inside_braces(input_text)
    assert result == expected_output

@pytest.mark.parametrize("input_value, expected", [
    (float('nan'), True),    # Happy path test
    ('2137', False),            # Edge case test
    (1.0, False),            # Additional happy path test
    (-1.0, False),           # Additional happy path test
    (float('inf'), False),   # Additional happy path test
    (1e+1000, False),        # Additional edge case test
    ("NaN", False),          # Error case test
    (None, False)            # Additional error case test
])
def test_is_nan(input_value, expected):
    result = utils.is_nan(input_value)
    assert result == expected, f"Expected {expected} for input {input_value}"


@pytest.mark.parametrize("input_value, expected_result", [
    pytest.param("2137", True, id="happy_integer_string"),
    pytest.param("123.456", True, id="happy_decimal_string"),
    pytest.param("-123.456", True, id="happy_negative_decimal_string"),
    pytest.param("0.0", True, id="happy_zero_decimal_string"),
    pytest.param("4e7", True, id="happy_scientific_notation_positive"),
    pytest.param("-1.4e3", True, id="happy_scientific_notation_negative"),
    pytest.param("+123", True, id="happy_positive_sign_integer_string"),
    pytest.param(".5", True, id="happy_leading_dot_decimal_string"),
    pytest.param("5.", True, id="happy_trailing_dot_decimal_string"),
    pytest.param("", False, id="edge_empty_string"),
    pytest.param(" ", False, id="edge_space_string"),
    pytest.param("NaN", True, id="edge_nan_string"),
    pytest.param("1e", False, id="edge_incomplete_scientific_notation"),
    pytest.param(".", False, id="edge_single_dot_string"),
    pytest.param("..1", False, id="edge_double_dot_string"),
    pytest.param(None, False, id="error_none_value"),
    pytest.param([123.456], False, id="error_list_with_float"),
    pytest.param({"number": 123.456}, False, id="error_dict_with_float"),
    pytest.param(complex(1, 1), False, id="error_complex_number"),
])
def test_is_float(input_value, expected_result):
    result = utils.is_float(input_value)
    assert result == expected_result


@pytest.mark.parametrize("input_value, expected_result", [
    pytest.param("2137", True, id="happy_integer_string"),
    pytest.param("-42", True, id="happy_negative_integer_string"),
    pytest.param("0", True, id="happy_zero_string"),
    pytest.param(42, True, id="happy_integer"),
    pytest.param(-42, True, id="happy_negative_integer"),
    pytest.param(0, True, id="happy_zero"),
    pytest.param("3.14", False, id="invalid_float_string"),
    pytest.param("abc", False, id="invalid_alphabetic_string"),
    pytest.param("", False, id="invalid_empty_string"),
    pytest.param(None, False, id="invalid_none"),
    pytest.param([1, 2, 3], False, id="invalid_list"),
    pytest.param({"number": 42}, False, id="invalid_dict"),
    pytest.param(complex(1, 2), False, id="invalid_complex_number"),
])
def test_is_int(input_value, expected_result):
    result = utils.is_int(input_value)
    assert result == expected_result

@pytest.mark.parametrize("input_value, expected_result", [
    pytest.param("2137", 2137, id="happy_integer_string"),
    pytest.param("-42", -42, id="happy_negative_integer_string"),
    pytest.param("0", 0, id="happy_zero_string"),
])
def test_to_int(input_value, expected_result):
    result = utils.to_int(input_value)
    assert result == expected_result

@pytest.mark.parametrize("input_value, expected_result", [
    pytest.param("lol 3.14", np.nan, id="invalid_string_and_float"),
    pytest.param("3.14", np.nan, id="invalid_float_string"),
    pytest.param("abc", np.nan, id="invalid_alphabetic_string"),
    pytest.param("", np.nan, id="invalid_empty_string"),
    pytest.param(None, np.nan, id="invalid_none"),
    pytest.param([1, 2, 3], np.nan, id="invalid_list"),
    pytest.param({"number": 42}, np.nan, id="invalid_dict"),
    pytest.param(complex(1, 2), np.nan, id="invalid_complex_number"),
])
def test_to_int_nans(input_value, expected_result):
    result = utils.to_int(input_value)
    assert np.isnan(result) and np.isnan(expected_result)


@pytest.mark.parametrize("input_value, expected_result", [
    pytest.param("2137", 2137, id="happy_integer_string"),
    pytest.param("-42", -42, id="happy_negative_integer_string"),
    pytest.param("0", 0, id="happy_zero_string"),
    pytest.param("3.14", 3.14, id="happy_float_string"),
    pytest.param("-2137.14", -2137.14, id="happy_negative_float_string"),
])
def test_to_float(input_value, expected_result):
    result = utils.to_float(input_value)
    assert result == expected_result

@pytest.mark.parametrize("input_value, expected_result", [
    pytest.param("abc", np.nan, id="invalid_alphabetic_string"),
    pytest.param("", np.nan, id="invalid_empty_string"),
    pytest.param(None, np.nan, id="invalid_none"),
    pytest.param([1, 2, 3], np.nan, id="invalid_list"),
    pytest.param({"number": 42}, np.nan, id="invalid_dict"),
    pytest.param(complex(1, 2), np.nan, id="invalid_complex_number"),
])
def test_to_float_nans(input_value, expected_result):
    result = utils.to_float(input_value)
    assert np.isnan(result) and np.isnan(expected_result)

@pytest.mark.parametrize("test_input, word_list, expected_output, test_id", [
    # Happy path tests with various realistic test values
    pytest.param("Hello world", ["hello"], "Hello", "happy_case_single_word", id="happy_case_single_word"),
    pytest.param("Hello world", ["hello", "world"], "Hello world", "happy_case_multiple_words", id="happy_case_multiple_words"),
    pytest.param("The quick brown fox", ["the", "fox"], "The fox", "happy_case_mixed_case", id="happy_case_mixed_case"),
    pytest.param("Python is great", ["python", "great"], "Python great", "happy_case_non_contiguous_words", id="happy_case_non_contiguous_words"),
    pytest.param("", [], "", "happy_case_empty_string", id="happy_case_empty_string"),
    pytest.param("One Two Three", ["one", "three"], "One Three", "happy_case_filter_out_middle_word", id="happy_case_filter_out_middle_word"),

    # Edge cases
    pytest.param("    ", [], "", "edge_case_spaces_only", id="edge_case_spaces_only"),
    pytest.param("Hello world!!!", ["hello", "world"], "Hello world", "edge_case_punctuation", id="edge_case_punctuation"),
    pytest.param("UPPER lower", ["upper", "lower"], "UPPER lower", "edge_case_mixed_case_input", id="edge_case_mixed_case_input"),
    pytest.param("Word1 Word2", ["word1", "word2"], "Word1 Word2", "edge_case_numerics_in_string", id="edge_case_numerics_in_string"),
    pytest.param("", ["irrelevant"], "", "edge_case_empty_string_with_non_empty_list", id="edge_case_empty_string_with_non_empty_list"),
    pytest.param("Non-matching string", ["none"], "", "edge_case_no_matches", id="edge_case_no_matches"),
])
def test_reverse_filter_str(test_input, word_list, expected_output, test_id):
    result = utils.reverse_filter_str(test_input, word_list)
    assert result == expected_output, f"Failed {test_id}"

@pytest.mark.parametrize("input_dict, expected_output", [
    # Happy path tests with various realistic test values
    pytest.param({"a": [1, 2], "b": [3, 4]}, {1: 'a', 2: 'a', 3: 'b', 4: 'b'}, id="happy_case1"),
    pytest.param({"x": [10], "y": [20, 30]}, {10: 'x', 20: 'y', 30: 'y'}, id="happy_case2"),
    pytest.param({"key1": ["value1"], "key2": ["value2"]}, {"value1": 'key1', "value2": 'key2'}, id="happy_case3"),
    pytest.param({"animal": ["dog", "cat"], "color": ["blue", "green"]}, {"dog": 'animal', "cat": 'animal', "blue": 'color', "green": 'color'}, id="happy_case4"),

    # Edge cases
    pytest.param({}, {}, id="edge_case1"),  # Empty dictionary
    pytest.param({"a": []}, {}, id="edge_case2"),  # Empty list as value
    pytest.param({"a": [1], "b": [1]}, {1: 'b'}, id="edge_case3"),  # Overlapping values
    pytest.param({"a": [None], "b": [""]}, {None: 'a', "": 'b'}, id="edge_case4"),  # None and empty string as values
])
def test_flatten_dict(input_dict, expected_output):
    result = utils.flatten_dict(input_dict)
    assert result == expected_output

@pytest.mark.parametrize("s, word, expected", [
    # Happy path tests with various realistic test values
    pytest.param("test word python", "word", "python", id="happy_simple"),
    pytest.param("word is at the start word another", "word", "is", id="happy_start"),
    pytest.param("ends with the word word ", "word", "", id="happy_end_with_space"),
    pytest.param("word repeated word multiple word times", "word", "repeated", id="happy_repeated_word"),
    pytest.param("special characters !@# word after", "word", "after", id="happy_special_chars"),
    pytest.param("word\nnewline word with", "word", "newline", id="happy_newline"),
    pytest.param("tabs\tword\tand spaces", "word", "and", id="happy_tabs_and_spaces"),

    # Edge cases
    pytest.param("", "word", "", id="edge_empty_string"),
    pytest.param("word", "word", "", id="edge_word_is_only_string"),
    pytest.param("word word", "word", "", id="edge_word_repeated_no_space"),
    pytest.param("   word   spaced   ", "word", "spaced", id="edge_spaces_around_word"),
    pytest.param("wordwordword", "word", "", id="edge_no_delimiter"),
])

def test_get_text_after_word(s, word, expected):
    result = utils.get_text_after_word(s, word)
    assert result == expected


@pytest.mark.parametrize("s, word, expected", [
    # Happy path tests with various realistic test values
    pytest.param("The price is 19.99 dollars", "dollars", 19.99, id="HP-1"),
    pytest.param("100 cats", "cats", 100, id="HP-2"),
    pytest.param("Value: 0.56 units", "units", 0.56, id="HP-3"),
    pytest.param("4.5 s something", "s", 4.5, id="HP-4"),
    pytest.param("Temperature is -40.00 C", "C", -40.00, id="HP-5"),

    # Edge cases
    pytest.param("19.99 dollars", "", 19.99, id="EC-1"),
    pytest.param("19.99 19.99 dollars", "dollars", 19.99, id="EC-2"),
    pytest.param("19.99 dollars 20.01 dollars", "dollars", 19.99, id="EC-3"),
    pytest.param("19.99 dollars 19.99", "dollars", 19.99, id="EC-4"),
])
def test_get_float_before_word(s, word, expected):
    result = utils.get_float_before_word(s, word)
    assert result == expected

@pytest.mark.parametrize("s, word, expected", [
    pytest.param("", "dollars", np.nan, id="EC-1"),
    pytest.param("dollars 19.99", "dollars", np.nan, id="EC-2"),
    pytest.param("The price is nineteen dollars", "dollars", np.nan, id="EC-3"),
    pytest.param("19.99dollars", "dollars", np.nan, id="EC-4"),
    pytest.param("The price is dollars", "dollars", np.nan, id="EC-5"),
])
def test_get_float_before_word_nans(s, word, expected):
    result = utils.get_float_before_word(s, word)
    assert np.isnan(result) and np.isnan(expected)


@pytest.mark.parametrize("s, word, expected", [
    # Happy path tests with various realistic test values
    pytest.param("The price is 19 dollars", "dollars", 19, id="HP-1"),
    pytest.param("100 cats", "cats", 100, id="HP-2"),
    pytest.param("Value: 0 units", "units", 0, id="HP-3"),
    pytest.param("4 s something", "s", 4, id="HP-4"),
    pytest.param("Temperature is -40 C", "C", -40, id="HP-5"),

    # Edge cases
    pytest.param("19 dollars", "", 19, id="EC-2"),
    pytest.param("19 19 dollars", "dollars", 19, id="EC-5"),
    pytest.param("19 dollars 20 dollars", "dollars", 19, id="EC-7"),
    pytest.param("19 dollars 19", "dollars", 19, id="EC-8"),
])
def test_get_int_before_word(s, word, expected):
    result = utils.get_int_before_word(s, word)
    assert result == expected

@pytest.mark.parametrize("s, word, expected", [
    pytest.param("", "dollars", np.nan, id="EC-1"),
    pytest.param("dollars 19", "dollars", np.nan, id="EC-2"),
    pytest.param("The price is nineteen dollars", "dollars", np.nan, id="EC-3"),
    pytest.param("19dollars", "dollars", np.nan, id="EC-4"),
    pytest.param("The price is dollars", "dollars", np.nan, id="EC-5"),
])
def test_get_int_before_word_nans(s, word, expected):
    result = utils.get_int_before_word(s, word)
    assert np.isnan(result) and np.isnan(expected)

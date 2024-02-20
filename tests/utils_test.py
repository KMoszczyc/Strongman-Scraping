import pytest
from src.utils import remap_list
def test_remap_list():
    values = [45.93, 4.0, 40.01, 4.0, 3.0, 0.0]
    points = [5, 3, 6, 4, 2, 0]
    # expected_values =

    remapped_values = remap_list(values_list=values, points_list=points)
    # assert
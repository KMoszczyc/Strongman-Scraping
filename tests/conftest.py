import pytest
from pathlib import Path
import os

TEST_DIR = "test_dir/"
COMPETITION_NAME="2004 Arnold's Strongest Man"
WSM_COMPETITION_NAME="1995 World's Strongest Man"

def pytest_sessionstart(session):
    comp_path = os.path.join(TEST_DIR, f"{COMPETITION_NAME}.csv")
    wsm_comp_path = os.path.join(TEST_DIR, "finals", f"{WSM_COMPETITION_NAME}.csv")
    wsm_dir = os.path.join(TEST_DIR, "finals")

    if not os.path.isdir(wsm_dir):
        os.makedirs(wsm_dir)
    if not os.path.isfile(comp_path):
        Path(comp_path).touch()
    if not os.path.isfile(wsm_comp_path):
        Path(wsm_comp_path).touch()

# def pytest_sessionfinish(session):

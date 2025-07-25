import os
import sys
from pathlib import Path

# Add project root to Python path
project_root = str(Path(__file__).parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import common test utilities
import pytest

@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return project_root

@pytest.fixture(scope="session")
def test_data_path():
    """Return the test data directory path."""
    return os.path.join(project_root, "tests", "data") 
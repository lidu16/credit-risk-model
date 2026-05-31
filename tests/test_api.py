import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import():
    from src.api.main import app
    assert app.title == "Credit Risk Model API"

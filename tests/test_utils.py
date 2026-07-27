"""
Unit tests for utils module.
"""
import pytest
import pandas as pd
import numpy as np
import json
import tempfile
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from src.utils import (
    ensure_directory,
    save_json,
    load_json,
    save_model,
    load_model,
    save_dataframe,
    load_dataframe
)


def test_ensure_directory():
    """Test that directory is created when it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test" / "nested" / "dir"
        ensure_directory(str(test_dir))
        assert test_dir.exists()
        assert test_dir.is_dir()


def test_save_and_load_json():
    """Test JSON save and load functionality."""
    data = {"name": "test", "value": 42, "nested": {"key": "value"}}
    
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        filepath = f.name
    
    save_json(data, filepath)
    loaded = load_json(filepath)
    assert loaded == data


def test_save_and_load_model():
    """Test model save and load functionality."""
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit([[1, 2], [3, 4]], [0, 1])
    
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
        filepath = f.name
    
    save_model(model, filepath)
    loaded = load_model(filepath)
    
    assert loaded is not None
    assert hasattr(loaded, 'predict')
    assert loaded.n_estimators == 10


def test_save_and_load_dataframe():
    """Test DataFrame save and load functionality."""
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': ['a', 'b', 'c'],
        'col3': [1.5, 2.5, 3.5]
    })
    
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        filepath = f.name
    
    save_dataframe(df, filepath)
    loaded = load_dataframe(filepath)
    
    pd.testing.assert_frame_equal(df, loaded)
from unittest.mock import MagicMock, patch, call
import numpy as np
import pytest
import torch
from fid import (
    calculate_fid,
    compute_fid_score,
    extract_features,
    get_inception_model,
    preprocess_for_inception,
)


@pytest.fixture(scope="module")
# Fetching InceptionV3 is fairly expensive so we cache and reuse it
def inception():
    return get_inception_model()

def test_get_inception_model_returns_module(inception):    
    assert isinstance(inception, torch.nn.Module)

def test_get_inception_model_fc_is_identity(inception):
    assert isinstance(inception.fc, torch.nn.Identity)

def test_get_inception_model_is_eval_mode(inception):
    assert not inception.training

def test_preprocess_for_inception_resizes_image_to_299_by_299():
    # Arrange
    channels = 3
    height = 32
    width = 32
    num_images = 2
    images = [torch.rand(channels, height, width) for _ in range(num_images)]

    # Act
    result = preprocess_for_inception(images)

    # Assert
    assert result.shape == (num_images, channels, 299, 299)


def test_preprocess_for_inception_output_is_normalized():
    # Arrange
    channels = 3
    height = 32
    width = 32
    images = [torch.zeros(channels, height, width) for _ in range(2)]

    # Act
    result = preprocess_for_inception(images)

    # Assert
    # result = (0 - mean) / std = -mean / std so all values should always be negative
    assert (result < 0).all()


def test_extract_features_returns_2D_array(inception):
    # Arrange
    channels = 3
    height = 32
    width = 32
    num_images = 8

    images = [torch.rand(channels, height, width) for _ in range(num_images)]

    # Act 
    result = extract_features(images, inception, batch_size=4)

    # Assert
    assert result.shape[0] == num_images
    assert result.shape[1] == 2048
    assert result.ndim == 2

def test_extract_features_returns_numpy_ndarray(inception):
    # Arrange
    channels = 3
    height = 32
    width = 32
    num_images = 2
    images = [torch.rand(channels, height, width) for _ in range(num_images)]

    # Act 
    result = extract_features(images, inception)

    # Assert   
    assert isinstance(result, np.ndarray)

def test_extract_features_num_images_indivisible_by_batch_size_returns_all_rows(inception):
    # Arrange
    channels = 3
    height = 32
    width = 32
    num_images = 5
    images = [torch.rand(channels, height, width) for _ in range(num_images)]

    # Act
    result = extract_features(images, inception, batch_size=3)
    assert result.shape[0] == num_images


def _make_features(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((64, 2048)).astype(np.float32)

def test_calculate_fid_returns_float():
    # Arrange
    a = _make_features(seed=1) 
    b = _make_features(seed=2)

    # Act
    result = calculate_fid(a, b)

    # Assert
    assert isinstance(result, float)

def test_calculate_fid_non_negative():
    # Arrange
    a = _make_features(seed=3) 
    b = _make_features(seed=4)

    # Act
    result = calculate_fid(a, b)

    # Assert
    assert result >= 0


def test_calculate_fid_real_and_generated_features_identical_returns_score_near_zero():
    feats = _make_features()

    # Act
    result = calculate_fid(feats, feats)

    # Assert
    assert result < 0.0001


def test_calculate_fid_larger_when_distributions_differ():
    # Arrange
    feats = _make_features(seed=5)
    shifted = _make_features(seed=5) + 10  # large mean shift

    # Act
    score_same = calculate_fid(feats, feats)
    score_diff = calculate_fid(feats, shifted)
    assert score_diff > score_same
    assert score_diff > 1

def test_calculate_fid_complex_covmean_returns_real_part_only():
    # Arrange
    rng = np.random.default_rng(99)
    samples = 5
    dimensions = 5
    feats = rng.standard_normal((samples, dimensions)).astype(np.float32)

    # Act
    result = calculate_fid(feats, feats + 0.01)

    # Assert
    assert np.isfinite(result)

@patch("fid.calculate_fid", return_value=5.0)
@patch("fid.extract_features")
@patch("fid.get_inception_model")
def test_compute_fid_score_orchestration(mock_model, mock_extract, mock_fid):
    # Arrange
    channels = 3
    height = 32
    width = 32
    real_feats = np.ones((4, 2048))
    gen_feats  = np.ones((4, 2048)) * 2
    mock_extract.side_effect = [real_feats, gen_feats]

    real_images = [torch.rand(channels, height, width) for _ in range(4)]
    gen_images  = [torch.rand(channels, height, width) for _ in range(4)]

    # Act
    result = compute_fid_score(real_images, gen_images, device="cpu", batch_size=16)

    # Assert
    mock_model.assert_called_once_with("cpu")
    # extract_features is called once per image set
    assert mock_extract.call_count == 2
    mock_fid.assert_called_once_with(real_feats, gen_feats)
    assert result == 5.0


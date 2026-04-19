import torch
from unittest.mock import patch
from kid import compute_kid, polynomial_kernel, _mmd2_on_subset, compute_mmd_squared


def test_polynomial_kernel_output_shape_grid_of_X_and_Y():
    # Arrange
    X = torch.rand(10, 64)
    Y = torch.rand(8, 64)

    # Act
    K = polynomial_kernel(X, Y)

    # Assert
    assert K.shape == (X.shape[0], Y.shape[0])

def test_polynomial_kernel_square_output_is_symmetric():    
    # Arrange
    INCEPTION_FEATURE_DIM = 2048 # large d to test for floating point accumulation
    X = torch.rand(10, INCEPTION_FEATURE_DIM)

    # Act
    K = polynomial_kernel(X, X)

    # Assert
    # K[i,j] == K[j,i]
    # This test would fail if we have floating point accumulation, 
    # non-commutative dtypes, or GPU non-determinism (if test ran on GPU)
    assert torch.allclose(K, K.T, atol=1e-5)

def test_polynomial_kernel_formula():
    X = torch.rand(4, 8)
    Y = torch.rand(4, 8)
    
    # Act
    result = polynomial_kernel(X, Y)
    
    # Assert
    expected = ((1.0 / X.shape[1]) * X @ Y.T + 1) ** 3
    assert torch.allclose(result, expected, atol=1e-5)


def _make_features(n=100, d=128, seed=0):
    torch.manual_seed(seed)
    return torch.randn(n, d).double()

def test_mmd2_on_subset_same_distribution_lower_than_different():
    # Arrange
    real = _make_features(seed=0)
    same_dist = _make_features(seed=1)
    diff_dist = _make_features(seed=2) + 10.0

    # Act
    result_same = _mmd2_on_subset(real, same_dist)
    result_diff = _mmd2_on_subset(real, diff_dist)
        
    # Assert
    # Independent draws from same distribution should score lower than different distributions
    assert result_same < result_diff

def test_mmd2_on_subset_different_distributions_positive():
    # Arrange
    real = _make_features(seed=0)
    gen = _make_features(seed=1) + 10.0

    # Act
    result = _mmd2_on_subset(real, gen)

    # Assert
    assert result > 0

def test_mmd2_on_subset_returns_float():
    # Arrange
    real = _make_features(seed=0)
    gen = _make_features(seed=1)

    # Act
    result = _mmd2_on_subset(real, gen)

    # Assert
    assert isinstance(result, float)


def test_compute_mmd_squared_identical_distributions_near_zero():
    
    # Arrange
    real = _make_features(n=100, seed=0).numpy()
    same = _make_features(n=100, seed=1).numpy()

    # Act
    mean, _ = compute_mmd_squared(real, same, seed=3)

    # Assert
    # Two independent samples from the same distribution. Averaged over 100 subsets, mean should be near 0
    assert abs(mean) < 0.05

def test_compute_mmd_squared_different_distributions_positive_mean():
    # Arrange
    real = _make_features(n=100, seed=0).numpy()
    gen = (_make_features(n=100, seed=1) + 10.0).numpy()
    
    # Act    
    mean, _ = compute_mmd_squared(real, gen, seed=3)

    assert mean > 0

def test_compute_mmd_squared_returns_nonnegative_std():
    # Arrange
    real = _make_features(n=100, seed=0).numpy()
    gen = _make_features(n=100, seed=1).numpy()

    # Act
    _, std = compute_mmd_squared(real, gen, seed=3)

    # Assert
    assert std >= 0

def test_compute_mmd_squared_seed_reproducible_with_same_seed():
    # Arrange
    real = _make_features(n=100, seed=0).numpy()
    gen = _make_features(n=100, seed=1).numpy()

    # Act
    mean1, std1 = compute_mmd_squared(real, gen, seed=3)
    mean2, std2 = compute_mmd_squared(real, gen, seed=3)

    # Assert
    assert mean1 == mean2 
    assert std1 == std2

def test_compute_mmd_squared_caps_subset_size(capsys):
    # Arrange
    real = _make_features(n=3, seed=0).numpy()
    gen = _make_features(n=3, seed=1).numpy()

    # Act
    mean, _ = compute_mmd_squared(real, gen, seed=1)
    captured = capsys.readouterr()

    # Assert
    # With fewer images than subset_size=50, should warn and not crash
    assert "Warning" in captured.out
    assert isinstance(mean, float)

def test_compute_mmd_squared_images_equal_subset_size_no_warning(capsys):
    # Arrange
    real = _make_features(n=50, seed=0).numpy()
    gen = _make_features(n=50, seed=1).numpy()

    # Act
    mean, _ = compute_mmd_squared(real, gen, seed=1)
    captured = capsys.readouterr()

    # Assert
    # With subset_size=50, should not warn
    assert not "Warning" in captured.out
    assert captured.out == ""
    assert isinstance(mean, float)

# Inception feature extraction is mocked so tests stay fast and CPU-only.

def _fake_extract_features(images, *_, **__):
    torch.manual_seed(id(images))
    return torch.randn(len(images), 2048).numpy()

def _make_images(n=100, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    channels = 3
    height = 64
    width = 64    
    return torch.rand(n, channels, height, width)

@patch("kid.extract_features", side_effect=_fake_extract_features)
@patch("kid.get_inception_model", return_value=None)
def test_compute_kid_returns_float_tuple(*_):
    # Act
    mean, std = compute_kid(_make_images(), _make_images())

    # Assert
    assert isinstance(mean, float)
    assert isinstance(std, float)

@patch("kid.extract_features", side_effect=_fake_extract_features)
@patch("kid.get_inception_model", return_value=None)
def test_compute_kid_same_distribution_near_zero(*_):
    # Arrange
    real_images = _make_images(seed=0)
    generated_images = _make_images(seed=1)

    # Act
    mean, _ = compute_kid(real_images, generated_images)

    # Assert
    # independent same-distribution features
    assert abs(mean) < 0.05

@patch("kid.get_inception_model", return_value=None)
def test_compute_kid_different_distributions_positive(*_):
    # Arrange
    # Two feature sets with the sname shae but different means
    real_features = torch.randn(100, 2048).numpy()  # centered at 0
    gen_features = (torch.randn(100, 2048) + 10.0).numpy()  # shifted +10

    real_images = _make_images(seed=0)
    gen_images = _make_images(seed=1)

    def fake_extract(images, *_, **__):
        # return the correct feature tensor by identity-checking the input tensor's memory pointer 
        return real_features if images.data_ptr() == real_images.data_ptr() else gen_features

    with patch("kid.extract_features", side_effect=fake_extract):
        # Act
        mean, _ = compute_kid(real_images, gen_images)
    # Assert    
    assert mean > 0

@patch("kid.extract_features", side_effect=_fake_extract_features)
@patch("kid.get_inception_model", return_value=None)
def test_compute_kid_std_nonnegative(*_):
    # Act
    _, std = compute_kid(_make_images(), _make_images())
    # Assert
    assert std >= 0


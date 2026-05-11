import pytest
import torch
from noise_schedule import NoiseSchedule, cosine_beta_schedule

def test_noise_schedule_initialization_sets_num_timesteps():
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"

    # Act
    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Assert
    assert schedule.num_timesteps == num_timesteps
    assert schedule.betas.shape[0] == num_timesteps
    assert schedule.alphas.shape[0] == num_timesteps
    assert schedule.alphas_cumprod.shape[0] == num_timesteps
    assert schedule.sqrt_alphas_cumprod.shape[0] == num_timesteps
    assert schedule.sqrt_one_minus_alphas_cumprod.shape[0] == num_timesteps
    assert schedule.sqrt_recip_alphas.shape[0] == num_timesteps
    assert schedule.posterior_variance.shape[0] == num_timesteps    
    
def test_linear_noise_schedule_initialization_sets_betas():
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"

    # Act
    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Assert
    assert torch.allclose(schedule.betas[0], torch.tensor(beta_start))
    assert torch.allclose(schedule.betas[-1], torch.tensor(beta_end))

def test_cosine_noise_schedule_initialization_sets_betas():
    # Arrange
    num_timesteps = 100
    device = "cpu"

    # Act
    schedule = NoiseSchedule(num_timesteps=num_timesteps, schedule="cosine", device=device)

    # Assert
    expected_betas = cosine_beta_schedule(num_timesteps)
    assert torch.allclose(schedule.betas[0], expected_betas[0])
    assert torch.allclose(schedule.betas[-1], expected_betas[-1])

def test_linear_noise_schedule_initialization_computes_alphas():
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"

    # Act
    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Assert
    expected_alphas = 1.0 - torch.linspace(beta_start, beta_end, num_timesteps)
    assert torch.allclose(schedule.alphas, expected_alphas)

def test_cosine_noise_schedule_initialization_computes_alphas():
    # Arrange
    num_timesteps = 100
    device = "cpu"

    # Act
    schedule = NoiseSchedule(num_timesteps=num_timesteps, schedule="cosine", device=device)

    # Assert
    expected_alphas = 1.0 - cosine_beta_schedule(num_timesteps)
    assert torch.allclose(schedule.alphas, expected_alphas)

def test_linear_noise_schedule_initialization_computes_alphas_cumprod():
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"

    # Act
    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Assert
    expected_alphas_cumprod = torch.cumprod(1.0 - torch.linspace(beta_start, beta_end, num_timesteps), dim=0)
    assert torch.allclose(schedule.alphas_cumprod, expected_alphas_cumprod)

def test_cosine_noise_schedule_initialization_computes_alphas_cumprod():
    # Arrange
    num_timesteps = 100
    device = "cpu"

    # Act
    schedule = NoiseSchedule(num_timesteps=num_timesteps, schedule="cosine", device=device)

    # Assert
    expected_alphas_cumprod = torch.cumprod(1.0 - cosine_beta_schedule(num_timesteps), dim=0)
    assert torch.allclose(schedule.alphas_cumprod, expected_alphas_cumprod)

def test_linear_noise_schedule_to_returns_self():
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

    self = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Act
    result = self.to(new_device)

    # Assert
    assert result is self

def test_cosine_noise_schedule_to_returns_self():
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

    self = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="cosine", device=device)

    # Act
    result = self.to(new_device)

    # Assert
    assert result is self

def test_linear_noise_schedule_to_moves_tensors_to_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        return
    
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cuda"

    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Act
    schedule.to(new_device)

    # Assert
    assert schedule.device == new_device
    assert schedule.betas.device.type == new_device
    assert schedule.alphas.device.type == new_device
    assert schedule.alphas_cumprod.device.type == new_device
    assert schedule.sqrt_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_one_minus_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_recip_alphas.device.type == new_device
    assert schedule.posterior_variance.device.type == new_device

def test_cosine_noise_schedule_to_moves_tensors_to_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        return
    
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cuda"

    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="cosine", device=device)

    # Act
    schedule.to(new_device)

    # Assert
    assert schedule.device == new_device
    assert schedule.betas.device.type == new_device
    assert schedule.alphas.device.type == new_device
    assert schedule.alphas_cumprod.device.type == new_device
    assert schedule.sqrt_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_one_minus_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_recip_alphas.device.type == new_device
    assert schedule.posterior_variance.device.type == new_device

def test_linear_noise_schedule_to_moves_tensors_to_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
        return
    
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Act
    schedule.to(new_device)

    # Assert
    assert schedule.device == new_device
    assert schedule.betas.device.type == new_device
    assert schedule.alphas.device.type == new_device
    assert schedule.alphas_cumprod.device.type == new_device
    assert schedule.sqrt_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_one_minus_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_recip_alphas.device.type == new_device
    assert schedule.posterior_variance.device.type == new_device

def test_cosine_noise_schedule_to_moves_tensors_to_mps():
    if not torch.backends.mps.is_available():
        pytest.skip("MPS not available")
        return
    
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu")

    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="cosine", device=device)

    # Act
    schedule.to(new_device)

    # Assert
    assert schedule.device == new_device
    assert schedule.betas.device.type == new_device
    assert schedule.alphas.device.type == new_device
    assert schedule.alphas_cumprod.device.type == new_device
    assert schedule.sqrt_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_one_minus_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_recip_alphas.device.type == new_device
    assert schedule.posterior_variance.device.type == new_device

def test_linear_noise_schedule_to_moves_tensors_to_cpu():
    
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cpu"

    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="linear", device=device)

    # Act
    schedule.to(new_device)

    # Assert
    assert schedule.device == new_device
    assert schedule.betas.device.type == new_device
    assert schedule.alphas.device.type == new_device
    assert schedule.alphas_cumprod.device.type == new_device
    assert schedule.sqrt_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_one_minus_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_recip_alphas.device.type == new_device
    assert schedule.posterior_variance.device.type == new_device    

def test_cosine_noise_schedule_to_moves_tensors_to_cpu():
    
    # Arrange
    num_timesteps = 100
    beta_start = 0.01
    beta_end = 0.1
    device = "cpu"
    new_device = "cpu"

    schedule = NoiseSchedule(num_timesteps=num_timesteps, beta_start=beta_start, beta_end=beta_end, schedule="cosine", device=device)

    # Act
    schedule.to(new_device)

    # Assert
    assert schedule.device == new_device
    assert schedule.betas.device.type == new_device
    assert schedule.alphas.device.type == new_device
    assert schedule.alphas_cumprod.device.type == new_device
    assert schedule.sqrt_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_one_minus_alphas_cumprod.device.type == new_device
    assert schedule.sqrt_recip_alphas.device.type == new_device
    assert schedule.posterior_variance.device.type == new_device    
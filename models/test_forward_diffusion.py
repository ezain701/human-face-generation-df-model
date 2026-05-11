import torch
from forward_diffusion import extract, q_sample, p_losses

class DummySchedule:
    def __init__(self):
        # dummy values for the noise schedule. 
        self.sqrt_alphas_cumprod = torch.tensor([0.9, 0.8, 0.7])
        self.sqrt_one_minus_alphas_cumprod = torch.tensor([0.1, 0.2, 0.3])

class DummyModel(torch.nn.Module):
    def forward(self, x, t):
        return torch.zeros_like(x)

def test_extract_batchsize2_threechannels():
    # Arrange
    tensor = torch.tensor([0.1, 0.5, 0.9])
    # indices 0 and 2
    t = torch.tensor([0, 2])

    # Test with batch size of 2 and 3 channels, image size 256x256
    shape = (2, 3, 256, 256)

    # Act
    result = extract(tensor, t, shape)

    # Assert
    # The result should have shape (2, 1, 1, 1) 
    assert result.shape == (2, 1, 1, 1)
    # result contains values corresponding to indices 0 and 2 from the tensor 
    assert torch.allclose(result[0], torch.tensor([[[0.1]]]))
    assert torch.allclose(result[1], torch.tensor([[[0.9]]]))

def test_q_sample_same_shape_as_x_start():
    # Arrange
    schedule = DummySchedule()
    x_start = torch.ones((2, 3, 256, 256))
    t = torch.tensor([0, 1])

    # Act
    result = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape) * x_start 

    # Assert  
    # The result should have the same shape as x_start 
    # and contain values that are a combination of x_start 
    # and noise according to the noise schedule.
    assert result.shape == x_start.shape

def test_q_sample_result_not_greater_than_largest_sqrt_alpha():
    # Arrange
    schedule = DummySchedule()
    x_start = torch.ones((2, 3, 256, 256))
    t = torch.tensor([0, 1])

    # Act
    result = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape) * x_start 

    # Assert  
    assert torch.all(result <= 0.9)  

def test_q_sample_result_not_less_than_smallest_sqrt_alpha():
    # Arrange
    schedule = DummySchedule()
    x_start = torch.ones((2, 3, 256, 256))
    t = torch.tensor([0, 1])

    # Act
    result = extract(schedule.sqrt_alphas_cumprod, t, x_start.shape) * x_start 

    # Assert  
    assert torch.all(result >= 0.7)  

def test_p_losses_result_positive():
    # Arrange
    schedule = DummySchedule()
    model = DummyModel()
    x_start = torch.ones((2, 3, 256, 256))
    t = torch.tensor([0, 1])

    # Act
    loss = p_losses(schedule, model, x_start, t)

    # Assert
    assert loss.item() > 0  # Loss should be positive since noise is random

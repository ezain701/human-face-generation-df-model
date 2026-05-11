import torch
from dataset import get_dataloader


def test_get_dataloader_sets_batch_size():

    # Arrange
    batch_size = 16

    # Act
    result = get_dataloader([1], batch_size=batch_size, shuffle=True, num_workers=2)

    # Assert
    assert result.batch_size == batch_size

def test_get_dataloader_sets_num_workers():

    # Arrange
    num_workers = 2

    # Act
    result = get_dataloader([1], batch_size=16, shuffle=True, num_workers=num_workers)

    # Assert
    assert result.num_workers == num_workers

def test_get_dataloader_sets_dataset():

    # Arrange
    dataset = [1]

    # Act
    result = get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2)

    # Assert
    assert result.dataset == dataset

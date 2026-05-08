import json

import torch
from dataset import build_image_transform, get_dataloader, load_caption_map


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


def test_load_caption_map_from_json(tmp_path):
    caption_file = tmp_path / "captions.json"
    caption_file.write_text(json.dumps({"0001.jpg": "studio portrait"}))

    result = load_caption_map(str(caption_file))

    assert result["0001.jpg"] == "studio portrait"


def test_build_image_transform_supports_face_safe_level():
    transform = build_image_transform(64, augment_level="face_safe", train=True)

    assert transform is not None


def test_build_image_transform_test_split_is_deterministic_resize():
    transform = build_image_transform(64, augment_level="strong", train=False)

    names = [type(step).__name__ for step in transform.transforms]
    assert "RandomHorizontalFlip" not in names
    assert "RandomResizedCrop" not in names

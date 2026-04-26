import copy
import json
import torch
import torch.nn as nn
import pytest
from unittest.mock import MagicMock

from training.trainer import Trainer


class _SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.register_buffer("running_stat", torch.zeros(4))


def _make_trainer(tmp_path, *, ema=False, decay=0.9, scheduler=None):
    model = _SimpleModel()
    ema_model = copy.deepcopy(model) if ema else None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return Trainer(
        model=model,
        schedule=MagicMock(device="cpu", num_timesteps=10),
        dataloader=MagicMock(),
        optimizer=optimizer,
        ema_model=ema_model,
        ema_decay=decay,
        device="cpu",
        checkpoint_dir=str(tmp_path / "checkpoints"),
        log_dir=str(tmp_path / "logs"),
        scheduler=scheduler,
    )

def test_update_ema_noop_when_no_ema_model(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, ema=False)

    # Act 
    try:
        trainer._update_ema()
    except:
        pytest.fail("Exception raised in _update_ema")

def test_update_ema_decay_zero_sets_ema_to_model(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, ema=True, decay=0.0)
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.fill_(2.0)
        for p in trainer.ema_model.parameters():
            p.fill_(5.0)

    # Act
    trainer._update_ema()

    # Assert
    for p in trainer.ema_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 2.0))


def test_update_ema_decay_one_freezes_ema(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, ema=True, decay=1.0)
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.fill_(7.0)
        for p in trainer.ema_model.parameters():
            p.fill_(3.0)

    # Act
    trainer._update_ema()

    # Assert
    for p in trainer.ema_model.parameters():
        assert torch.allclose(p, torch.full_like(p, 3.0))


def test_update_ema_blends_with_correct_formula(tmp_path):
    # Arrange
    decay = 0.9
    trainer = _make_trainer(tmp_path, ema=True, decay=decay)
    with torch.no_grad():
        for p in trainer.model.parameters():
            p.fill_(10.0)
        for p in trainer.ema_model.parameters():
            p.fill_(0.0)
    expected = decay * 0.0 + (1 - decay) * 10.0  # 1.0

    for p in trainer.ema_model.parameters():
        assert len(p) == 4
        assert not torch.allclose(p, torch.full_like(p, expected))

    # Act
    trainer._update_ema()

    # Assert
    for p in trainer.ema_model.parameters():
        assert len(p) == 4
        assert torch.allclose(p, torch.full_like(p, expected))


def test_update_ema_copies_buffers_exactly(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, ema=True, decay=0.9)
    with torch.no_grad():
        trainer.model.running_stat.fill_(42.0)
        trainer.ema_model.running_stat.fill_(0.0)

    # Act
    trainer._update_ema()

    # Assert
    assert torch.allclose(trainer.ema_model.running_stat, torch.full((4,), 42.0))


def test_save_checkpoint_creates_file(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path)

    # Act
    trainer._save_checkpoint(epoch=5)

    # Assert
    assert (tmp_path / "checkpoints" / "model_epoch_5.pt").exists()


def test_save_checkpoint_contains_required_keys(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path)
    trainer._save_checkpoint(epoch=3)

    # Act
    result = torch.load(tmp_path / "checkpoints" / "model_epoch_3.pt")

    # Assert
    assert result["epoch"] == 3
    assert "model_state_dict" in result
    assert "optimizer_state_dict" in result


def test_save_checkpoint_with_ema_includes_ema_state(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, ema=True)
    trainer._save_checkpoint(epoch=1)

    # Act
    result = torch.load(tmp_path / "checkpoints" / "model_epoch_1.pt")

    # Assert
    assert "ema_model_state_dict" in result


def test_save_checkpoint_without_ema_excludes_ema_state(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path, ema=False)
    trainer._save_checkpoint(epoch=1)

    # Act
    result = torch.load(tmp_path / "checkpoints" / "model_epoch_1.pt")

    # Assert
    assert "ema_model_state_dict" not in result


def test_save_checkpoint_with_scheduler_includes_scheduler_state(tmp_path):
    # Arrange
    scheduler = MagicMock()
    scheduler.state_dict.return_value = {"last_epoch": 0}
    trainer = _make_trainer(tmp_path, scheduler=scheduler)
    trainer._save_checkpoint(epoch=2)

    # Act
    result = torch.load(tmp_path / "checkpoints" / "model_epoch_2.pt")

    # Assert
    assert "scheduler_state_dict" in result
    assert result["scheduler_state_dict"] == {"last_epoch": 0}


def test_save_log_creates_file(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path)
    trainer.training_log = [{"epoch": 1, "avg_loss": 0.5}]

    # Act
    trainer._save_log()

    # Assert
    assert (tmp_path / "logs" / "training_log.json").exists()


def test_save_log_content_matches_training_log(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path)
    log = [{"epoch": 1, "avg_loss": 0.42}, {"epoch": 2, "avg_loss": 0.31}]
    trainer.training_log = log

    # Act
    trainer._save_log()

    # Assert
    with open(tmp_path / "logs" / "training_log.json") as f:
        assert json.load(f) == log


def test_save_log_empty_log_writes_empty_array(tmp_path):
    # Arrange
    trainer = _make_trainer(tmp_path)
    trainer.training_log = []

    # Act
    trainer._save_log()

    # Assert
    with open(tmp_path / "logs" / "training_log.json") as f:
        assert json.load(f) == []

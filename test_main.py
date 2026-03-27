import pytest
from main import parse_args


@pytest.fixture(autouse=True)
def clear_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["main.py"])
    def _extend(*args):
        monkeypatch.setattr("sys.argv", ["main.py", *args])
    return _extend


def test_parse_args_defaults():
    args = parse_args()
    assert args.dataset == "celeba"
    assert args.image_size == 256
    assert args.epochs == 100
    assert args.lr == pytest.approx(2e-4)
    assert args.resume is None

def test_parse_args_custom_values(clear_argv):
    clear_argv("--dataset", "butterfly", "--epochs", "50", "--batch_size", "8")
    args = parse_args()
    assert args.dataset == "butterfly"
    assert args.epochs == 50
    assert args.batch_size == 8

def test_parse_args_invalid_dataset_raises(clear_argv):
    clear_argv("--dataset", "invalid")
    with pytest.raises(SystemExit):
        parse_args()

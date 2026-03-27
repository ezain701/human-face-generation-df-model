import pytest
from generate import parse_args


@pytest.fixture
def set_argv(monkeypatch):
    def _set(args):
        monkeypatch.setattr("sys.argv", args)
    return _set


def test_generate_parse_args_requires_checkpoint(set_argv):
    set_argv(["generate.py"])
    with pytest.raises(SystemExit):
        parse_args()

def test_generate_parse_args_with_checkpoint(set_argv):
    set_argv(["generate.py", "--checkpoint", "checkpoints/model.pt"])
    args = parse_args()
    assert args.checkpoint == "checkpoints/model.pt"
    assert args.num_images == 300
    assert args.compute_fid is False

def test_generate_parse_args_compute_fid_flag(set_argv):
    set_argv(["generate.py", "--checkpoint", "x.pt", "--compute_fid"])
    args = parse_args()
    assert args.compute_fid is True

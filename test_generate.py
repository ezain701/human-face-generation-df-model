import pytest
from generate import parse_args


@pytest.fixture(autouse=True)
def clear_argv(monkeypatch):
    monkeypatch.setattr("sys.argv", ["generate.py"])
    def _extend(*args):
        monkeypatch.setattr("sys.argv", ["generate.py", *args])
    return _extend

def test_generate_parse_args_requires_checkpoint():
    with pytest.raises(SystemExit):
        parse_args()

def test_generate_parse_args_with_checkpoint(clear_argv):
    clear_argv("--checkpoint", "checkpoints/model.pt")
    args = parse_args()
    assert args.checkpoint == "checkpoints/model.pt"
    assert args.num_images == 300
    assert args.compute_fid is False

def test_generate_parse_args_compute_fid_flag(clear_argv):
    clear_argv("--checkpoint", "x.pt", "--compute_fid")
    args = parse_args()
    assert args.compute_fid is True


def test_generate_parse_args_prompt_derives_metadata(clear_argv):
    clear_argv("--checkpoint", "x.pt", "--prompt", "studio portrait")
    args = parse_args()
    assert args.prompt == "studio portrait"
    assert args.seed is None


def test_generate_parse_args_seed_override(clear_argv):
    clear_argv("--checkpoint", "x.pt", "--seed", "12345")
    args = parse_args()
    assert args.seed == 12345


def test_generate_parse_args_text_conditioning(clear_argv):
    clear_argv("--checkpoint", "x.pt", "--text_conditioning", "--guidance_scale", "5")
    args = parse_args()
    assert args.text_conditioning is True
    assert args.guidance_scale == 5


def test_generate_parse_args_clip_text_encoder(clear_argv):
    clear_argv("--checkpoint", "x.pt", "--text_conditioning", "--text_encoder_type", "clip")
    args = parse_args()
    assert args.text_encoder_type == "clip"

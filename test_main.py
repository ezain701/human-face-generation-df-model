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
    assert args.image_size == 192
    assert args.train_size == 27000
    assert args.epochs == 100
    assert args.lr == pytest.approx(2e-4)
    assert args.resume is None
    assert args.augment_level == "face_safe"

def test_parse_args_custom_values(clear_argv):
    clear_argv("--dataset", "butterfly", "--epochs", "50", "--batch_size", "8", "--train_size", "1200")
    args = parse_args()
    assert args.dataset == "butterfly"
    assert args.epochs == 50
    assert args.batch_size == 8
    assert args.train_size == 1200

def test_parse_args_text_conditioning_values(clear_argv):
    clear_argv("--text_conditioning", "--caption_file", "captions.csv", "--caption_dropout", "0.2")
    args = parse_args()
    assert args.text_conditioning is True
    assert args.caption_file == "captions.csv"
    assert args.caption_dropout == pytest.approx(0.2)
    assert args.text_encoder_type == "simple"


def test_parse_args_clip_text_encoder(clear_argv):
    clear_argv("--text_conditioning", "--text_encoder_type", "clip", "--clip_model_name", "openai/clip-vit-base-patch16")
    args = parse_args()
    assert args.text_encoder_type == "clip"
    assert args.clip_model_name == "openai/clip-vit-base-patch16"


def test_parse_args_use_amp(clear_argv):
    clear_argv("--use_amp")
    args = parse_args()
    assert args.use_amp is True


def test_parse_args_sample_prompts(clear_argv):
    clear_argv("--sample_prompts", "smiling face", "person wearing a hat")
    args = parse_args()
    assert args.sample_prompts == ["smiling face", "person wearing a hat"]

def test_parse_args_augment_level(clear_argv):
    clear_argv("--augment_level", "strong")
    args = parse_args()
    assert args.augment_level == "strong"

def test_parse_args_invalid_dataset_raises(clear_argv):
    clear_argv("--dataset", "invalid")
    with pytest.raises(SystemExit):
        parse_args()

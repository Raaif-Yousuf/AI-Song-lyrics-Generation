from __future__ import annotations

from pathlib import Path

import pytest

from lyricgen.config import (
    ConfigError,
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    TokenizerConfig,
    load_config,
)

MINIMAL_YAML = """
name: test-experiment
data:
  dir: data/processed
tokenizer:
  type: char
model:
  kind: lstm
train:
  steps: 10
  batch_size: 4
  block_size: 16
  lr: 0.001
out_dir: runs/test-experiment
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def remove_block(text: str, key: str) -> str:
    """Remove a top-level `key:` line and every indented line under it."""
    lines = text.splitlines()
    kept = []
    skipping = False
    for line in lines:
        if line.startswith(f"{key}:"):
            skipping = True
            continue
        if skipping and (line.startswith(" ") or line.startswith("\t")):
            continue
        skipping = False
        kept.append(line)
    return "\n".join(kept)


def test_load_minimal_config_applies_defaults(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, MINIMAL_YAML))

    assert isinstance(config, ExperimentConfig)
    assert config.name == "test-experiment"
    assert config.data == DataConfig(dir="data/processed")
    assert config.tokenizer == TokenizerConfig(type="char", vocab_size=None)
    assert config.model == ModelConfig(kind="lstm", hyperparameters={})
    assert config.train.steps == 10
    assert config.train.batch_size == 4
    assert config.train.block_size == 16
    assert config.train.lr == pytest.approx(0.001)
    # Defaults.
    assert config.train.weight_decay == 0.0
    assert config.train.seed == 1337
    assert config.train.device == "cpu"
    assert config.out_dir == "runs/test-experiment"


def test_data_config_defaults_when_omitted(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace("data:\n  dir: data/processed\n", "")
    config = load_config(write(tmp_path, text))
    assert config.data.dir == "data/processed"
    assert config.data.train_path == str(Path("data/processed") / "train.jsonl")
    assert config.data.val_path == str(Path("data/processed") / "val.jsonl")
    assert config.data.test_path == str(Path("data/processed") / "test.jsonl")


def test_model_hyperparameters_pass_through(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace(
        "model:\n  kind: lstm\n",
        "model:\n  kind: transformer\n  n_layer: 2\n  n_head: 2\n  d_model: 64\n",
    )
    config = load_config(write(tmp_path, text))
    assert config.model.kind == "transformer"
    assert config.model.hyperparameters == {"n_layer": 2, "n_head": 2, "d_model": 64}


def test_bpe_tokenizer_requires_vocab_size(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace("tokenizer:\n  type: char\n", "tokenizer:\n  type: bpe\n")
    with pytest.raises(ConfigError, match="vocab_size"):
        load_config(write(tmp_path, text))


def test_bpe_tokenizer_with_vocab_size(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace(
        "tokenizer:\n  type: char\n", "tokenizer:\n  type: bpe\n  vocab_size: 2000\n"
    )
    config = load_config(write(tmp_path, text))
    assert config.tokenizer == TokenizerConfig(type="bpe", vocab_size=2000)


def test_invalid_tokenizer_type_rejected(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace("tokenizer:\n  type: char\n", "tokenizer:\n  type: word\n")
    with pytest.raises(ConfigError, match="tokenizer type"):
        load_config(write(tmp_path, text))


def test_invalid_model_kind_rejected(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace("model:\n  kind: lstm\n", "model:\n  kind: rnn\n")
    with pytest.raises(ConfigError, match="model kind"):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("key", ["name", "tokenizer", "model", "train", "out_dir"])
def test_missing_top_level_key_raises(tmp_path: Path, key: str) -> None:
    text = remove_block(MINIMAL_YAML, key)
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, text))


@pytest.mark.parametrize("key", ["steps", "batch_size", "block_size", "lr"])
def test_missing_required_train_key_raises(tmp_path: Path, key: str) -> None:
    lines = [line for line in MINIMAL_YAML.splitlines() if not line.strip().startswith(f"{key}:")]
    text = "\n".join(lines)
    with pytest.raises(ConfigError, match=key):
        load_config(write(tmp_path, text))


def test_unknown_top_level_key_raises(tmp_path: Path) -> None:
    text = MINIMAL_YAML + "\nextra_key: 1\n"
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write(tmp_path, text))


def test_unknown_train_key_raises(tmp_path: Path) -> None:
    text = MINIMAL_YAML.replace("  lr: 0.001\n", "  lr: 0.001\n  bogus: 1\n")
    with pytest.raises(ConfigError, match="unknown"):
        load_config(write(tmp_path, text))


def test_non_mapping_config_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="mapping"):
        load_config(write(tmp_path, "- just\n- a\n- list\n"))


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="could not read"):
        load_config(tmp_path / "does-not-exist.yaml")


def test_invalid_yaml_raises(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(write(tmp_path, "name: [unclosed\n"))


def test_to_dict_round_trips_shape(tmp_path: Path) -> None:
    config = load_config(write(tmp_path, MINIMAL_YAML))
    d = config.to_dict()
    assert d["name"] == "test-experiment"
    assert d["data"] == {"dir": "data/processed"}
    assert d["tokenizer"] == {"type": "char"}
    assert d["model"] == {"kind": "lstm"}
    assert d["train"]["steps"] == 10
    assert d["out_dir"] == "runs/test-experiment"


def test_shipped_configs_load(tmp_path: Path) -> None:
    configs_dir = Path(__file__).resolve().parent.parent / "configs"
    names = ["tiny", "lstm", "gru", "transformer", "transformer_bpe"]
    loaded = {name: load_config(configs_dir / f"{name}.yaml") for name in names}

    shared_train_keys = ("seed", "batch_size", "block_size")
    experiment_names = ["lstm", "gru", "transformer", "transformer_bpe"]
    reference = loaded["lstm"].train
    for name in experiment_names[1:]:
        other = loaded[name].train
        for key in shared_train_keys:
            assert getattr(other, key) == getattr(reference, key), (name, key)
        assert other.steps == reference.steps
        assert other.block_size == 256

    assert loaded["transformer_bpe"].tokenizer.type == "bpe"
    assert loaded["transformer_bpe"].tokenizer.vocab_size == pytest.approx(2000, rel=0.1)
    for name in ["lstm", "gru", "transformer"]:
        assert loaded[name].tokenizer.type == "char"

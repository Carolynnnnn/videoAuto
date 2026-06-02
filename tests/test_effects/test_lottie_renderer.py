from __future__ import annotations

import json
import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

lottie_renderer = importlib.import_module("src.effects.lottie_renderer")


class _FakeAnimation:
    def __init__(self) -> None:
        self.fr = 12
        self.ip = 0
        self.op = 23
        self.w = 200
        self.h = 100
        self.render_calls = 0

    def load_file(self, _path: str) -> None:
        return None

    def render(self, _frame_idx: int) -> np.ndarray:
        self.render_calls += 1
        frame = np.zeros((self.h, self.w, 4), dtype=np.uint8)
        frame[:, :, 3] = 255
        return frame


def _write_min_lottie(path: Path) -> None:
    payload = {
        "v": "5.7.6",
        "fr": 12,
        "ip": 0,
        "op": 23,
        "w": 200,
        "h": 100,
        "layers": [],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_convert_lottie_to_frames_uses_cache(monkeypatch, tmp_path: Path) -> None:
    lottie_path = tmp_path / "sample.json"
    _write_min_lottie(lottie_path)

    instances: list[_FakeAnimation] = []

    def _factory() -> _FakeAnimation:
        instance = _FakeAnimation()
        instances.append(instance)
        return instance

    monkeypatch.setattr(lottie_renderer, "Animation", _factory)

    first = lottie_renderer.convert_lottie_to_frames(
        lottie_path=str(lottie_path),
        cache_root=str(tmp_path / "cache"),
        target_fps=24,
    )
    first_calls = instances[0].render_calls

    second = lottie_renderer.convert_lottie_to_frames(
        lottie_path=str(lottie_path),
        cache_root=str(tmp_path / "cache"),
        target_fps=24,
    )

    assert first["used_cache"] is False
    assert second["used_cache"] is True
    assert first["frame_count"] == 48
    assert first["native_fps"] == 12.0
    assert first["frame_rate"] == 24
    assert Path(first["frame_pattern"]).name == "frame_%04d.png"
    assert Path(first["frames_dir"]).exists()
    assert (Path(first["frames_dir"]) / "frame_0001.png").exists()
    assert instances[0].render_calls == first_calls


def test_generate_lottie_overlay_builds_ffmpeg_image2_and_filter() -> None:
    effect = SimpleNamespace(
        position="top-right",
        start_time=1.0,
        duration=2.0,
        scale=0.5,
        transparency=0.8,
    )
    sequence = {
        "frame_rate": 30,
        "frame_pattern": "assets/lottie_cache/abc/frame_%04d.png",
        "width": 200,
        "height": 100,
    }

    result = lottie_renderer.generate_lottie_overlay(
        effect=effect,
        sequence_info=sequence,
        video_width=1080,
        video_height=1920,
        image_input_index=1,
        base_stream="[0:v]",
        output_stream="[vout]",
    )

    assert "-framerate 30 -i assets/lottie_cache/abc/frame_%04d.png" in result["input_args"]
    assert "scale=iw*0.5:ih*0.5" in result["filter"]
    assert "colorchannelmixer=aa=0.8" in result["filter"]
    assert "overlay=980:0" in result["filter"]
    assert "enable='between(t,1.0,3.0)'" in result["filter"]


def test_convert_lottie_to_frames_errors_without_animation(tmp_path: Path, monkeypatch) -> None:
    lottie_path = tmp_path / "sample.json"
    _write_min_lottie(lottie_path)
    monkeypatch.setattr(lottie_renderer, "Animation", None)

    try:
        lottie_renderer.convert_lottie_to_frames(str(lottie_path), cache_root=str(tmp_path))
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "pylottie" in str(exc).lower()

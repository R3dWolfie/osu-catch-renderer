import lzma
import tempfile
from pathlib import Path

from osu_catch_renderer.security import (
    bounded_lzma_decompress,
    ffmpeg_file_input_args,
    safe_related_file,
    safe_skin_file,
)


def test_bounded_lzma_decompress_rejects_oversized_output():
    compressed = lzma.compress(b"x" * (2 * 1024 * 1024))
    try:
        bounded_lzma_decompress(compressed, max_output=1024 * 1024)
    except ValueError as exc:
        assert "output limit" in str(exc)
    else:
        raise AssertionError("oversized LZMA stream was accepted")


def test_safe_related_file_confines_untrusted_beatmap_names():
    with tempfile.TemporaryDirectory() as directory:
        tmp_path = Path(directory)
        nested = tmp_path / "assets"
        nested.mkdir()
        target = nested / "background.png"
        target.write_bytes(b"x")
        assert safe_related_file(tmp_path, "background.png") == target
        assert safe_related_file(tmp_path, "assets/background.png") == target
        assert safe_related_file(tmp_path, "ASSETS\\BACKGROUND.PNG") == target
        assert safe_related_file(tmp_path, "../background.png") is None
        assert safe_related_file(tmp_path, "..\\background.png") is None
        assert safe_related_file(tmp_path, "/etc/passwd") is None
        assert safe_related_file(tmp_path, "C:\\Windows\\win.ini") is None
        assert safe_related_file(tmp_path, "..") is None
        assert safe_related_file(tmp_path, "") is None


def test_safe_skin_file_rejects_traversal_and_accepts_casefolded_subdirs():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        tmp_path = root / "skin"
        tmp_path.mkdir()
        fonts = tmp_path / "Fonts"
        fonts.mkdir()
        target = fonts / "Score-0.png"
        target.write_bytes(b"x")
        assert safe_skin_file(tmp_path, "fonts/score-0.png") == target
        assert safe_skin_file(tmp_path, "../Score-0.png") is None
        assert safe_skin_file(tmp_path, "/etc/passwd") is None
        assert safe_skin_file(tmp_path, "C:\\Windows\\win.ini") is None
        outside = root / "outside-skin.png"
        outside.write_bytes(b"x")
        (tmp_path / "escape").symlink_to(outside)
        assert safe_skin_file(tmp_path, "escape") is None


def test_ffmpeg_file_input_args_forces_file_protocol():
    with tempfile.TemporaryDirectory() as directory:
        args = ffmpeg_file_input_args(Path(directory) / "hit.wav")
        assert args[:4] == ["-protocol_whitelist", "file", "-f", "wav"]

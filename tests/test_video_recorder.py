import os
from unittest.mock import patch

import numpy as np
import pytest

from visualization.video_recorder import VideoRecorder


@pytest.fixture
def tmp_video_dir(tmp_path):
    return str(tmp_path / "videos")


@pytest.fixture
def recorder(tmp_video_dir):
    with patch("visualization.video_recorder.play_video"):
        rec = VideoRecorder(output_dir=tmp_video_dir)
    yield rec
    if rec.recording:
        rec.stop_recording()


def _make_frame(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


class TestInit:
    def test_creates_output_dir(self, tmp_video_dir):
        assert not os.path.exists(tmp_video_dir)
        VideoRecorder(output_dir=tmp_video_dir)
        assert os.path.isdir(tmp_video_dir)

    def test_existing_dir_no_error(self, tmp_video_dir):
        os.makedirs(tmp_video_dir)
        VideoRecorder(output_dir=tmp_video_dir)

    def test_initial_state(self, recorder):
        assert recorder.recording is False
        assert recorder.video_writer is None
        assert recorder.frame_size is None
        assert recorder.output_filename is None


class TestStartRecording:
    def test_starts_recording(self, recorder):
        frame = _make_frame()
        recorder.start_recording(frame)
        assert recorder.recording is True
        assert recorder.video_writer is not None
        assert recorder.frame_size == (640, 480)
        assert recorder.output_filename is not None

    def test_creates_output_file(self, recorder):
        recorder.start_recording(_make_frame())
        assert os.path.exists(recorder.output_filename)

    def test_no_double_start(self, recorder):
        recorder.start_recording(_make_frame())
        first_filename = recorder.output_filename
        recorder.start_recording(_make_frame())
        assert recorder.output_filename == first_filename


class TestRecordFrame:
    def test_writes_frame(self, recorder):
        frame = _make_frame()
        recorder.start_recording(frame)
        # Should not raise
        recorder.record_frame(frame)

    def test_no_write_when_not_recording(self, recorder):
        # Should not raise even if not recording
        recorder.record_frame(_make_frame())


class TestStopRecording:
    def test_stops_recording(self, recorder):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        assert recorder.recording is False

    def test_stop_when_not_recording(self, recorder):
        # Should not raise
        recorder.stop_recording()


class TestSaveVideoWithNewName:
    def test_rename(self, recorder):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        old_path = recorder.output_filename
        with patch("builtins.input", return_value="renamed.mp4"):
            recorder.save_video_with_new_name()
        assert not os.path.exists(old_path)
        assert os.path.exists(recorder.output_filename)
        assert recorder.output_filename.endswith("renamed.mp4")

    def test_keep_name(self, recorder, capsys):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        old_path = recorder.output_filename
        with patch("builtins.input", return_value=""):
            recorder.save_video_with_new_name()
        assert recorder.output_filename == old_path
        assert "saved as" in capsys.readouterr().out


class TestPostRecordingOptions:
    def test_rewatch_then_exit(self, recorder, capsys):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        with patch("visualization.video_recorder.play_video") as mock_play, \
             patch("builtins.input", side_effect=["1", "3", "n"]):
            recorder.post_recording_options()
        mock_play.assert_called_once_with(recorder.output_filename)
        assert "playback completed" in capsys.readouterr().out

    def test_save_option(self, recorder):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        with patch("builtins.input", side_effect=["2", "new.mp4"]):
            recorder.post_recording_options()
        assert recorder.output_filename.endswith("new.mp4")

    def test_exit_and_delete(self, recorder):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        path = recorder.output_filename
        with patch("builtins.input", side_effect=["3", "y"]):
            recorder.post_recording_options()
        assert not os.path.exists(path)

    def test_exit_keep(self, recorder):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        path = recorder.output_filename
        with patch("builtins.input", side_effect=["3", "n"]):
            recorder.post_recording_options()
        assert os.path.exists(path)

    def test_invalid_then_exit(self, recorder, capsys):
        recorder.start_recording(_make_frame())
        recorder.stop_recording()
        with patch("builtins.input", side_effect=["bad", "3", "n"]):
            recorder.post_recording_options()
        assert "Invalid choice" in capsys.readouterr().out

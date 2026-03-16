from unittest.mock import patch, MagicMock



class TestVideoPlayer:
    @patch("visualization.video_player.cv2")
    @patch("visualization.video_player.QTimer")
    @patch("visualization.video_player.QSlider")
    @patch("visualization.video_player.QLabel")
    @patch("visualization.video_player.QPushButton")
    @patch("visualization.video_player.QComboBox")
    def test_init(self, mock_combo, mock_btn, mock_label, mock_slider,
                  mock_timer, mock_cv2):
        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            3: 640,   # CAP_PROP_FRAME_WIDTH
            4: 480,   # CAP_PROP_FRAME_HEIGHT
            5: 30.0,  # CAP_PROP_FPS
            7: 100,   # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        mock_cap.read.return_value = (False, None)
        mock_cv2.VideoCapture.return_value = mock_cap

        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.video_path = "test.mp4"
            player.cap = mock_cap
            player.frame_count = 100
            player.fps = 30.0
            player.current_frame = 0
            player.playing = False
            player.playback_speed = 1.0

        assert player.frame_count == 100
        assert player.fps == 30.0
        assert player.playing is False

    def test_change_playback_speed(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.playing = False
            player.fps = 30.0
            player.playback_speed = 1.0
            player.timer = MagicMock()

            player.change_playback_speed("0.5x")
            assert player.playback_speed == 0.5

            player.change_playback_speed("2x")
            assert player.playback_speed == 2.0

    def test_change_speed_while_playing_updates_timer(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.playing = True
            player.fps = 30.0
            player.playback_speed = 1.0
            player.timer = MagicMock()

            player.change_playback_speed("2x")
            player.timer.setInterval.assert_called_once()

    def test_play_pause_toggle(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.playing = False
            player.fps = 30.0
            player.playback_speed = 1.0
            player.timer = MagicMock()
            player.play_pause_button = MagicMock()

            player.play_pause()
            assert player.playing is True
            player.timer.start.assert_called_once()
            player.play_pause_button.setText.assert_called_with("Pause")

            player.play_pause()
            assert player.playing is False
            player.timer.stop.assert_called_once()
            player.play_pause_button.setText.assert_called_with("Play")

    def test_next_frame_increments(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.current_frame = 5
            player.frame_count = 100
            player.cap = MagicMock()
            player.cap.read.return_value = (False, None)
            player.video_label = MagicMock()
            player.slider = MagicMock()

            player.next_frame()
            assert player.current_frame == 6

    def test_next_frame_wraps_around(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.current_frame = 99
            player.frame_count = 100
            player.cap = MagicMock()
            player.cap.read.return_value = (False, None)
            player.video_label = MagicMock()
            player.slider = MagicMock()

            player.next_frame()
            assert player.current_frame == 0
            player.cap.set.assert_called()

    def test_rewind(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.current_frame = 200
            player.fps = 30.0
            player.cap = MagicMock()
            player.cap.read.return_value = (False, None)
            player.video_label = MagicMock()
            player.slider = MagicMock()

            player.rewind()
            # Rewinds 5 seconds: 200 - int(30 * 5) = 200 - 150 = 50
            assert player.current_frame == 50

    def test_rewind_clamps_to_zero(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.current_frame = 10
            player.fps = 30.0
            player.cap = MagicMock()
            player.cap.read.return_value = (False, None)
            player.video_label = MagicMock()
            player.slider = MagicMock()

            player.rewind()
            assert player.current_frame == 0

    def test_slider_moved(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.cap = MagicMock()
            player.cap.read.return_value = (False, None)
            player.video_label = MagicMock()
            player.slider = MagicMock()

            player.slider_moved(42)
            assert player.current_frame == 42

    def test_close_event_releases_capture(self):
        from visualization.video_player import VideoPlayer
        with patch.object(VideoPlayer, "__init__", lambda self, path: None):
            player = VideoPlayer.__new__(VideoPlayer)
            player.cap = MagicMock()
            event = MagicMock()

            player.closeEvent(event)
            player.cap.release.assert_called_once()
            event.accept.assert_called_once()


class TestPlayVideo:
    @patch("visualization.video_player.QApplication")
    @patch("visualization.video_player.VideoPlayer")
    def test_play_video_creates_app_if_needed(self, mock_player_cls, mock_qapp):
        mock_qapp.instance.return_value = None
        mock_app = MagicMock()
        mock_qapp.return_value = mock_app

        from visualization.video_player import play_video
        play_video("test.mp4")

        mock_qapp.assert_called()
        mock_player_cls.assert_called_with("test.mp4")
        mock_app.exec_.assert_called_once()

    @patch("visualization.video_player.QApplication")
    @patch("visualization.video_player.VideoPlayer")
    def test_play_video_reuses_existing_app(self, mock_player_cls, mock_qapp):
        mock_app = MagicMock()
        mock_qapp.instance.return_value = mock_app

        from visualization.video_player import play_video
        play_video("test.mp4")

        # Should not create a new QApplication
        mock_qapp.assert_not_called()
        mock_app.exec_.assert_called_once()

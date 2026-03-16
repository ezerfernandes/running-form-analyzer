import time
from unittest.mock import patch, MagicMock

from feedback.audio_feedback import AudioFeedbackProvider


@patch("feedback.audio_feedback.pyttsx3")
def test_init(mock_pyttsx3):
    mock_engine = MagicMock()
    mock_pyttsx3.init.return_value = mock_engine
    provider = AudioFeedbackProvider()
    mock_pyttsx3.init.assert_called_once()
    assert provider.engine is mock_engine
    assert provider.feedback_queue == []
    assert provider.last_feedback_time == {}
    assert provider.feedback_thread.daemon is True
    assert provider.feedback_thread.is_alive()


@patch("feedback.audio_feedback.pyttsx3")
def test_feedback_loop_processes_queue(mock_pyttsx3):
    mock_engine = MagicMock()
    mock_pyttsx3.init.return_value = mock_engine
    provider = AudioFeedbackProvider()
    provider.feedback_queue.append("Test message")
    # Give the daemon thread time to process
    time.sleep(0.3)
    mock_engine.say.assert_called_with("Test message")
    mock_engine.runAndWait.assert_called()
    assert len(provider.feedback_queue) == 0


@patch("feedback.audio_feedback.pyttsx3")
def test_add_feedback_first_message(mock_pyttsx3):
    mock_pyttsx3.init.return_value = MagicMock()
    provider = AudioFeedbackProvider()
    provider.add_feedback("Fix your posture")
    assert "Fix your posture" in provider.feedback_queue
    assert "Fix your posture" in provider.last_feedback_time


@patch("feedback.audio_feedback.pyttsx3")
def test_add_feedback_cooldown_blocks_duplicate(mock_pyttsx3):
    mock_pyttsx3.init.return_value = MagicMock()
    provider = AudioFeedbackProvider()
    # Drain queue so the thread doesn't consume our messages
    provider.feedback_queue.clear()
    provider.add_feedback("Fix your posture", cooldown=10)
    provider.add_feedback("Fix your posture", cooldown=10)
    # Second call within cooldown should not add again
    assert provider.feedback_queue.count("Fix your posture") == 1


@patch("feedback.audio_feedback.pyttsx3")
def test_add_feedback_cooldown_allows_after_expiry(mock_pyttsx3):
    mock_pyttsx3.init.return_value = MagicMock()
    provider = AudioFeedbackProvider()
    provider.add_feedback("Fix your posture", cooldown=0)
    provider.feedback_queue.clear()
    # Cooldown of 0 means any time > 0 allows re-adding
    time.sleep(0.01)
    provider.add_feedback("Fix your posture", cooldown=0)
    assert "Fix your posture" in provider.feedback_queue


@patch("feedback.audio_feedback.pyttsx3")
def test_add_feedback_different_messages(mock_pyttsx3):
    mock_pyttsx3.init.return_value = MagicMock()
    provider = AudioFeedbackProvider()
    provider.add_feedback("Message A")
    provider.add_feedback("Message B")
    assert "Message A" in provider.feedback_queue
    assert "Message B" in provider.feedback_queue

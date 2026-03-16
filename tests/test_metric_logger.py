import csv
import os
from unittest.mock import patch

import pytest

from visualization.metric_logger import MetricsLogger


@pytest.fixture
def tmp_log_dir(tmp_path):
    return str(tmp_path / "logs")


@pytest.fixture
def logger(tmp_log_dir):
    lg = MetricsLogger(log_dir=tmp_log_dir)
    yield lg
    lg.close()


class TestInit:
    def test_creates_log_dir(self, tmp_log_dir):
        assert not os.path.exists(tmp_log_dir)
        MetricsLogger(log_dir=tmp_log_dir)
        assert os.path.isdir(tmp_log_dir)

    def test_existing_dir_no_error(self, tmp_log_dir):
        os.makedirs(tmp_log_dir)
        MetricsLogger(log_dir=tmp_log_dir)

    def test_log_file_path_set(self, logger):
        assert logger.log_file.startswith(logger.log_dir)
        assert logger.log_file.endswith(".csv")

    def test_initial_state(self, logger):
        assert logger.csv_file is None
        assert logger.csv_writer is None
        assert logger.metrics == []


class TestInitializeLogging:
    def test_creates_csv_with_headers(self, logger):
        logger.initialize_logging({"head_angle": 0.0, "trunk_angle": 0.0})
        logger.close()

        with open(logger.log_file) as f:
            reader = csv.reader(f)
            headers = next(reader)
        assert headers == ["timestamp", "head_angle", "trunk_angle"]

    def test_sets_metrics_list(self, logger):
        logger.initialize_logging({"a": 1, "b": 2, "c": 3})
        assert logger.metrics == ["a", "b", "c"]

    def test_csv_writer_ready(self, logger):
        logger.initialize_logging({"x": 0})
        assert logger.csv_file is not None
        assert logger.csv_writer is not None


class TestLogMetrics:
    def test_writes_row(self, logger):
        logger.initialize_logging({"head_angle": 0.0, "trunk_angle": 0.0})
        logger.log_metrics(1.0, {"head_angle": 90.5, "trunk_angle": 10.2})
        logger.close()

        with open(logger.log_file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            row = next(reader)
        assert row == ["1.0", "90.5", "10.2"]

    def test_multiple_rows(self, logger):
        logger.initialize_logging({"val": 0.0})
        logger.log_metrics(0.0, {"val": 1.0})
        logger.log_metrics(0.5, {"val": 2.0})
        logger.log_metrics(1.0, {"val": 3.0})
        logger.close()

        with open(logger.log_file) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 4  # header + 3 data rows

    def test_missing_metric_uses_empty_string(self, logger):
        logger.initialize_logging({"a": 0, "b": 0})
        logger.log_metrics(1.0, {"a": 42})
        logger.close()

        with open(logger.log_file) as f:
            reader = csv.reader(f)
            next(reader)
            row = next(reader)
        assert row == ["1.0", "42", ""]

    def test_without_initialize_prints_error(self, logger, capsys):
        logger.log_metrics(1.0, {"x": 1})
        captured = capsys.readouterr()
        assert "Error" in captured.out


class TestClose:
    def test_closes_file(self, logger):
        logger.initialize_logging({"x": 0})
        assert not logger.csv_file.closed
        logger.close()
        assert logger.csv_file.closed

    def test_close_without_init_no_error(self, tmp_log_dir):
        lg = MetricsLogger(log_dir=tmp_log_dir)
        lg.close()  # should not raise


class TestViewLogSummary:
    def test_no_metrics_prints_message(self, logger, capsys):
        logger.view_log_summary()
        assert "No metrics were logged" in capsys.readouterr().out

    def test_empty_data_prints_message(self, logger, capsys):
        logger.initialize_logging({"val": 0.0})
        logger.close()
        logger.view_log_summary()
        assert "No data recorded" in capsys.readouterr().out

    def test_summary_with_data(self, logger, capsys):
        logger.initialize_logging({"val": 0.0})
        logger.log_metrics(0.0, {"val": 10.0})
        logger.log_metrics(1.0, {"val": 20.0})
        logger.close()
        logger.view_log_summary()
        output = capsys.readouterr().out
        assert "Total records: 2" in output
        assert "Average val: 15.00" in output

    def test_summary_with_non_numeric(self, logger, capsys):
        logger.initialize_logging({"status": ""})
        logger.log_metrics(0.0, {"status": "Good"})
        logger.close()
        logger.view_log_summary()
        output = capsys.readouterr().out
        assert "Non-numeric data" in output


class TestRenameLogFile:
    def test_rename(self, logger):
        logger.initialize_logging({"x": 0})
        logger.close()
        old_path = logger.log_file
        with patch("builtins.input", return_value="renamed.csv"):
            logger.rename_log_file()
        assert not os.path.exists(old_path)
        assert os.path.exists(logger.log_file)
        assert logger.log_file.endswith("renamed.csv")

    def test_keep_current_name(self, logger, capsys):
        logger.initialize_logging({"x": 0})
        logger.close()
        old_path = logger.log_file
        with patch("builtins.input", return_value=""):
            logger.rename_log_file()
        assert logger.log_file == old_path
        assert "saved as" in capsys.readouterr().out


class TestDeleteLogFile:
    def test_delete_confirmed(self, logger):
        logger.initialize_logging({"x": 0})
        logger.close()
        assert os.path.exists(logger.log_file)
        with patch("builtins.input", return_value="y"):
            logger.delete_log_file()
        assert not os.path.exists(logger.log_file)

    def test_delete_declined(self, logger):
        logger.initialize_logging({"x": 0})
        logger.close()
        with patch("builtins.input", return_value="n"):
            logger.delete_log_file()
        assert os.path.exists(logger.log_file)


class TestPostLoggingOptions:
    def test_view_then_exit(self, logger, capsys):
        logger.initialize_logging({"val": 0.0})
        logger.log_metrics(0.0, {"val": 5.0})
        logger.close()
        with patch("builtins.input", side_effect=["1", "3", "n"]):
            logger.post_logging_options()
        output = capsys.readouterr().out
        assert "Total records: 1" in output

    def test_rename_option(self, logger):
        logger.initialize_logging({"x": 0})
        logger.close()
        with patch("builtins.input", side_effect=["2", "new_name.csv"]):
            logger.post_logging_options()
        assert logger.log_file.endswith("new_name.csv")

    def test_exit_option(self, logger):
        logger.initialize_logging({"x": 0})
        logger.close()
        with patch("builtins.input", side_effect=["3", "n"]):
            logger.post_logging_options()

    def test_invalid_then_exit(self, logger, capsys):
        logger.initialize_logging({"x": 0})
        logger.close()
        with patch("builtins.input", side_effect=["bad", "3", "n"]):
            logger.post_logging_options()
        assert "Invalid choice" in capsys.readouterr().out

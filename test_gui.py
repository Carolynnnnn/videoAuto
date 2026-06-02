# pyright: reportMissingImports=false
import os
import sys
from pathlib import Path

import pytest
from PyQt5.QtTest import QSignalSpy
from PyQt5.QtWidgets import QApplication, QTabWidget

APP_DIR = Path(__file__).parent
sys.path.insert(0, str(APP_DIR))

from src.gui.app import VideoAutomationApp

TARGET_WIDTH = 1024
TARGET_HEIGHT = 700
MIN_BUTTON_WIDTH = 60
MIN_BUTTON_HEIGHT = 28


def _get_or_create_app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _process_events(app) -> None:
    for _ in range(3):
        app.processEvents()


def _disconnect_all_click_handlers(button) -> None:
    while True:
        try:
            button.clicked.disconnect()
        except TypeError:
            break


def _assert_control_bounds(widget, name: str) -> None:
    geometry = widget.geometry()
    assert geometry.width() >= MIN_BUTTON_WIDTH, f"{name} width too small: {geometry.width()}"
    assert geometry.height() >= MIN_BUTTON_HEIGHT, f"{name} height too small: {geometry.height()}"


@pytest.fixture
def gui_window():
    app = _get_or_create_app()
    window = VideoAutomationApp()
    window.resize(TARGET_WIDTH, TARGET_HEIGHT)
    window.show()
    _process_events(app)

    yield app, window

    window.close()
    _process_events(app)


def test_geometry_1024x700(gui_window):
    app, window = gui_window

    actual_size = window.size()
    assert actual_size.width() == TARGET_WIDTH
    assert actual_size.height() == TARGET_HEIGHT
    assert window.minimumWidth() == TARGET_WIDTH
    assert window.minimumHeight() == TARGET_HEIGHT

    assert window._btn_start.isVisible()
    _assert_control_bounds(window._btn_start, "_btn_start")

    assert window._btn_stop.isVisible()
    _assert_control_bounds(window._btn_stop, "_btn_stop")

    tab_widgets = window.findChildren(QTabWidget)
    assert tab_widgets, "QTabWidget not found"
    tab_widgets[0].setCurrentIndex(1)
    _process_events(app)

    assert window._btn_inc_start.isVisible()
    _assert_control_bounds(window._btn_inc_start, "_btn_inc_start")


def test_critical_controls_clickable(gui_window):
    app, window = gui_window

    _disconnect_all_click_handlers(window._btn_start)
    start_spy = QSignalSpy(window._btn_start.clicked)
    assert window._btn_start.isEnabled()
    window._btn_start.click()
    _process_events(app)
    assert len(start_spy) == 1

    _disconnect_all_click_handlers(window._btn_stop)
    stop_spy = QSignalSpy(window._btn_stop.clicked)
    assert not window._btn_stop.isEnabled()
    window._btn_stop.click()
    _process_events(app)
    assert len(stop_spy) == 0

    tab_widgets = window.findChildren(QTabWidget)
    assert tab_widgets, "QTabWidget not found"
    tab_widgets[0].setCurrentIndex(1)
    _process_events(app)

    _disconnect_all_click_handlers(window._btn_inc_start)
    inc_spy = QSignalSpy(window._btn_inc_start.clicked)
    assert window._btn_inc_start.isEnabled()
    window._btn_inc_start.click()
    _process_events(app)
    assert len(inc_spy) == 1

def test_material_mode_selector_options(gui_window):
    app, window = gui_window
    
    # Check full build tab
    combo = window._material_mode_combo
    assert combo.count() == 3
    assert combo.itemText(0) == "auto (自动混合)"
    assert combo.itemText(1) == "ai_preferred (AI优先)"
    assert combo.itemText(2) == "ai_only (纯AI生成)"

    # Check incremental tab
    inc_combo = window._inc_material_mode_combo
    assert inc_combo.count() == 3
    assert inc_combo.itemText(0) == "auto (自动混合)"
    assert inc_combo.itemText(1) == "ai_preferred (AI优先)"
    assert inc_combo.itemText(2) == "ai_only (纯AI生成)"


def test_material_mode_forwarded_full_build(gui_window, monkeypatch, tmp_path):
    app, window = gui_window
    
    captured_args = []
    
    class MockWorkerThread:
        def __init__(self, task_fn, *args, **kwargs):
            captured_args.append((task_fn, args, kwargs))
            self.progress_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.step_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.finished_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.percent_signal = type('Signal', (), {'connect': lambda self, x: None})()
            
        def start(self):
            pass
            
    monkeypatch.setattr("src.gui.app.WorkerThread", MockWorkerThread)
    
    # Setup required fields for full build
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    window._pdf_path_edit.setText(str(fake_pdf))
    window._project_name_edit.setText("test_proj")
    window._deepseek_key_edit.setText("fake_key")
    window._minimax_key_edit.setText("fake_key")
    
    # Test ai_preferred
    window._material_mode_combo.setCurrentText("ai_preferred (AI优先)")
    window._btn_start.click()
    _process_events(app)
    
    assert len(captured_args) == 1
    task_fn, args, kwargs = captured_args[0]
    assert task_fn == window._run_full_build
    # args: pdf_path, project_name, aspect_ratio, tts_voice, whisper_model, pexels_key, enable_pexels_video, enable_pexels_photo, enable_ai_image, material_mode
    assert args[-1] == "ai_preferred"


def test_material_mode_forwarded_incremental(gui_window, monkeypatch, tmp_path):
    app, window = gui_window
    
    captured_args = []
    
    class MockWorkerThread:
        def __init__(self, task_fn, *args, **kwargs):
            captured_args.append((task_fn, args, kwargs))
            self.progress_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.step_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.finished_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.percent_signal = type('Signal', (), {'connect': lambda self, x: None})()
            
        def start(self):
            pass
            
    monkeypatch.setattr("src.gui.app.WorkerThread", MockWorkerThread)
    
    # Setup required fields for incremental
    fake_proj = tmp_path / "fake_project"
    fake_proj.mkdir()
    window._inc_project_edit.setText(str(fake_proj))
    
    # Test ai_only
    window._inc_material_mode_combo.setCurrentText("ai_only (纯AI生成)")
    window._btn_inc_start.click()
    _process_events(app)
    
    assert len(captured_args) == 1
    task_fn, args, kwargs = captured_args[0]
    assert task_fn == window._run_incremental
    # args: project_path, new_srt, dry_run, full_rebuild, material_mode
    assert args[-1] == "ai_only"


def test_material_mode_invalid_injection(gui_window, monkeypatch, tmp_path):
    app, window = gui_window
    
    captured_args = []
    
    class MockWorkerThread:
        def __init__(self, task_fn, *args, **kwargs):
            captured_args.append((task_fn, args, kwargs))
            self.progress_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.step_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.finished_signal = type('Signal', (), {'connect': lambda self, x: None})()
            self.percent_signal = type('Signal', (), {'connect': lambda self, x: None})()
            
        def start(self):
            pass
            
    monkeypatch.setattr("src.gui.app.WorkerThread", MockWorkerThread)
    
    # Setup required fields for full build
    fake_pdf = tmp_path / "fake.pdf"
    fake_pdf.touch()
    window._pdf_path_edit.setText(str(fake_pdf))
    window._project_name_edit.setText("test_proj")
    window._deepseek_key_edit.setText("fake_key")
    window._minimax_key_edit.setText("fake_key")
    
    # Inject invalid mode
    window._material_mode_combo.setCurrentText("invalid_mode_hacked")
    window._btn_start.click()
    _process_events(app)
    
    assert len(captured_args) == 1
    task_fn, args, kwargs = captured_args[0]
    # Should default to "auto"
    assert args[-1] == "auto"

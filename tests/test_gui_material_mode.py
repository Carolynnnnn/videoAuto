import pytest
from unittest.mock import patch, MagicMock
from PyQt5.QtWidgets import QApplication
from src.gui.app import VideoAutomationApp

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

@pytest.fixture
def main_window(qapp):
    window = VideoAutomationApp()
    yield window
    window.close()

def test_gui_material_mode_selector_exists(main_window):
    """Test that the material mode selector exists and has correct options."""
    combo = main_window._material_mode_combo
    assert combo is not None
    
    items = [combo.itemText(i) for i in range(combo.count())]
    assert "auto (自动混合)" in items
    assert "ai_preferred (AI优先)" in items
    assert "ai_only (纯AI生成)" in items
    
    # Default should be auto
    assert combo.currentText() == "auto (自动混合)"

def test_gui_inc_material_mode_selector_exists(main_window):
    """Test that the incremental material mode selector exists and has correct options."""
    combo = main_window._inc_material_mode_combo
    assert combo is not None
    
    items = [combo.itemText(i) for i in range(combo.count())]
    assert "auto (自动混合)" in items
    assert "ai_preferred (AI优先)" in items
    assert "ai_only (纯AI生成)" in items
    
    # Default should be auto
    assert combo.currentText() == "auto (自动混合)"

@patch("src.gui.app.WorkerThread")
def test_gui_full_build_passes_material_mode(mock_worker_thread, main_window):
    """Test that the selected material mode is passed to the full build worker."""
    # Setup required fields to pass validation
    main_window._pdf_path_edit.setText("dummy.pdf")
    with patch("pathlib.Path.exists", return_value=True):
        main_window._deepseek_key_edit.setText("dummy_key")
        main_window._minimax_key_edit.setText("dummy_key")
        
        # Select ai_only mode
        idx = main_window._material_mode_combo.findText("ai_only (纯AI生成)")
        main_window._material_mode_combo.setCurrentIndex(idx)
        
        # Trigger build
        main_window._start_build()
        
        # Verify WorkerThread was called with ai_only
        assert mock_worker_thread.called
        args, kwargs = mock_worker_thread.call_args
        
        # The material_mode is the 11th argument (index 10) in WorkerThread call
        # _run_full_build, pdf_path, project_name, aspect_ratio, tts_voice, whisper_model,
        # pexels_key, enable_pexels_video, enable_pexels_photo, enable_ai_image, material_mode
        assert args[10] == "ai_only"

@patch("src.gui.app.WorkerThread")
def test_gui_incremental_build_passes_material_mode(mock_worker_thread, main_window):
    """Test that the selected material mode is passed to the incremental build worker."""
    # Setup required fields to pass validation
    main_window._inc_project_edit.setText("dummy_project")
    with patch("pathlib.Path.exists", return_value=True):
        # Select ai_preferred mode
        idx = main_window._inc_material_mode_combo.findText("ai_preferred (AI优先)")
        main_window._inc_material_mode_combo.setCurrentIndex(idx)
        
        # Trigger build
        main_window._start_incremental()
        
        # Verify WorkerThread was called with ai_preferred
        assert mock_worker_thread.called
        args, kwargs = mock_worker_thread.call_args
        
        # The material_mode is the 6th argument (index 5) in WorkerThread call
        # _run_incremental, project_path, new_srt, dry_run, full_rebuild, material_mode
        assert args[5] == "ai_preferred"

@patch("src.gui.app.WorkerThread")
def test_gui_invalid_material_mode_injection(mock_worker_thread, main_window):
    """Test that injecting an unsupported material mode safely defaults to 'auto'."""
    # Setup required fields to pass validation for full build
    main_window._pdf_path_edit.setText("dummy.pdf")
    with patch("pathlib.Path.exists", return_value=True):
        main_window._deepseek_key_edit.setText("dummy_key")
        main_window._minimax_key_edit.setText("dummy_key")
        
        # Inject an invalid mode by mocking currentText
        with patch.object(main_window._material_mode_combo, 'currentText', return_value="invalid_hacked_mode"):
            # Trigger build
            main_window._start_build()
            
            # Verify WorkerThread was called with the safe default 'auto'
            assert mock_worker_thread.called
            args, kwargs = mock_worker_thread.call_args
            
            # The material_mode is the 11th argument (index 10) in WorkerThread call
            assert args[10] == "auto"

    # Setup required fields to pass validation for incremental build
    main_window._inc_project_edit.setText("dummy_project")
    with patch("pathlib.Path.exists", return_value=True):
        # Inject an invalid mode by mocking currentText
        with patch.object(main_window._inc_material_mode_combo, 'currentText', return_value="another_invalid_mode"):
            # Trigger build
            main_window._start_incremental()
            
            # Verify WorkerThread was called with the safe default 'auto'
            assert mock_worker_thread.called
            args, kwargs = mock_worker_thread.call_args
            
            # The material_mode is the 6th argument (index 5) in WorkerThread call
            assert args[5] == "auto"

@patch("src.gui.app.WorkerThread")
def test_gui_full_duration_selector(mock_worker_thread, main_window):
    """Test that the selected duration limit is passed to the full build worker."""
    main_window._pdf_path_edit.setText("dummy.pdf")
    with patch("pathlib.Path.exists", return_value=True):
        main_window._deepseek_key_edit.setText("dummy_key")
        main_window._minimax_key_edit.setText("dummy_key")
        
        # Select 2分钟
        idx = main_window._duration_combo.findText("2分钟")
        main_window._duration_combo.setCurrentIndex(idx)
        
        main_window._start_build()
        
        assert mock_worker_thread.called
        args, kwargs = mock_worker_thread.call_args
        
        # _run_full_build, pdf_path, project_name, aspect_ratio, tts_voice, whisper_model,
        # pexels_key, enable_pexels_video, enable_pexels_photo, enable_ai_image, material_mode, target_duration
        assert kwargs.get("target_duration") == 2

@patch("src.gui.app.WorkerThread")
def test_gui_full_duration_invalid_injection(mock_worker_thread, main_window):
    """Test that injecting an unsupported duration safely defaults to 1."""
    main_window._pdf_path_edit.setText("dummy.pdf")
    with patch("pathlib.Path.exists", return_value=True):
        main_window._deepseek_key_edit.setText("dummy_key")
        main_window._minimax_key_edit.setText("dummy_key")
        
        with patch.object(main_window._duration_combo, 'currentText', return_value="invalid_duration"):
            main_window._start_build()
            
            assert mock_worker_thread.called
            args, kwargs = mock_worker_thread.call_args
            
            assert kwargs.get("target_duration") == 1

@patch("src.gui.app.WorkerThread")
def test_gui_incremental_duration_propagation(mock_worker_thread, main_window):
    """Test that the selected duration limit is passed to the incremental build worker."""
    main_window._inc_project_edit.setText("dummy_project")
    with patch("pathlib.Path.exists", return_value=True):
        # Select 2 minutes
        idx = main_window._inc_duration_combo.findText("2 分钟")
        main_window._inc_duration_combo.setCurrentIndex(idx)
        
        main_window._start_incremental()
        
        assert mock_worker_thread.called
        args, kwargs = mock_worker_thread.call_args
        
        # _run_incremental, project_path, new_srt, dry_run, full_rebuild, material_mode, duration_limit
        assert kwargs.get("duration_limit") == 2

@patch("src.gui.app.WorkerThread")
def test_gui_incremental_duration_invalid_injection(mock_worker_thread, main_window):
    """Test that injecting an unsupported duration safely defaults to 1."""
    main_window._inc_project_edit.setText("dummy_project")
    with patch("pathlib.Path.exists", return_value=True):
        with patch.object(main_window._inc_duration_combo, 'currentText', return_value="invalid_duration"):
            main_window._start_incremental()
            
            assert mock_worker_thread.called
            args, kwargs = mock_worker_thread.call_args
            
            assert kwargs.get("duration_limit") == 1


def test_gui_full_duration_selector_default_value(main_window):
    """Test that the full build duration selector defaults to 1 minute."""
    combo = main_window._duration_combo
    assert combo is not None
    
    # Default text should indicate 1 minute
    default_text = combo.currentText()
    assert "1" in default_text and "分钟" in default_text, f"Default duration should be 1 minute, got: {default_text}"
    
    # Default selection index should be 0 (which is 1 minute)
    assert combo.currentIndex() == 0, f"Default index should be 0, got: {combo.currentIndex()}"


def test_gui_incremental_duration_selector_default_value(main_window):
    """Test that the incremental build duration selector defaults to 1 minute."""
    combo = main_window._inc_duration_combo
    assert combo is not None
    
    # Default text should indicate 1 minute
    default_text = combo.currentText()
    assert "1" in default_text and "分钟" in default_text, f"Default duration should be 1 minute, got: {default_text}"
    
    # Default selection index should be 0 (which is 1 minute)
    assert combo.currentIndex() == 0, f"Default index should be 0, got: {combo.currentIndex()}"


def test_gui_full_duration_selector_options(main_window):
    """Test that the full build duration selector has all valid options."""
    combo = main_window._duration_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    
    # Should have options for 1, 2, 3 minutes exactly
    assert len(items) == 3, f"Expected 3 options, got {len(items)}"
    assert "1" in items[0] and "分钟" in items[0], "First option should be 1 minute"
    assert "2" in items[1] and "分钟" in items[1], "Second option should be 2 minutes"
    assert "3" in items[2] and "分钟" in items[2], "Third option should be 3 minutes"


def test_gui_incremental_duration_selector_options(main_window):
    """Test that the incremental build duration selector has all valid options."""
    combo = main_window._inc_duration_combo
    items = [combo.itemText(i) for i in range(combo.count())]
    
    # Should have options for 1, 2, 3 minutes exactly
    assert len(items) == 3, f"Expected 3 options, got {len(items)}"
    assert "1" in items[0] and "分钟" in items[0], "First option should be 1 minute"
    assert "2" in items[1] and "分钟" in items[1], "Second option should be 2 minutes"
    assert "3" in items[2] and "分钟" in items[2], "Third option should be 3 minutes"

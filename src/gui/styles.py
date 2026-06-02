COLORS = {
    "bg_dark":      "#0d1117",
    "bg_card":      "#161b22",
    "bg_input":     "#21262d",
    "border":       "#30363d",
    "accent":       "#58a6ff",
    "accent_hover": "#79c0ff",
    "success":      "#3fb950",
    "warning":      "#d29922",
    "error":        "#f85149",
    "text_primary": "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted":   "#66707c",
    "purple":       "#bc8cff",
    "orange":       "#ffa657",
    "gradient_start": "#1a1f35",
    "gradient_end":   "#0d1117",
}

STYLE_SHEET = f"""
QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}
QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Microsoft YaHei", sans-serif;
    font-size: 13px;
}}
QGroupBox {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-size: 13px;
    font-weight: bold;
    color: {COLORS['text_secondary']};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {COLORS['accent']};
    font-size: 13px;
}}
QPushButton {{
    background-color: {COLORS['accent']};
    color: #0d1117;
    border: none;
    border-radius: 6px;
    padding: 8px 18px;
    font-weight: bold;
    font-size: 13px;
    min-height: 32px;
}}
QPushButton:hover {{
    background-color: {COLORS['accent_hover']};
}}
QPushButton:pressed {{
    background-color: #1f6feb;
}}
QPushButton:disabled {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_muted']};
}}
QPushButton#btn_secondary {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
}}
QPushButton#btn_secondary:hover {{
    background-color: {COLORS['border']};
}}
QPushButton#btn_success {{
    background-color: {COLORS['success']};
    color: #0d1117;
}}
QPushButton#btn_success:hover {{
    background-color: #56d364;
}}
QPushButton#btn_danger {{
    background-color: {COLORS['error']};
    color: white;
}}
QPushButton#btn_large {{
    font-size: 15px;
    padding: 12px 28px;
    min-height: 44px;
    border-radius: 8px;
}}
QLineEdit, QTextEdit, QComboBox {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 6px 10px;
    color: {COLORS['text_primary']};
    font-size: 13px;
    selection-background-color: {COLORS['accent']};
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {COLORS['accent']};
}}
QComboBox::drop-down {{
    border: none;
    width: 24px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {COLORS['text_secondary']};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent']};
    selection-color: #0d1117;
    color: {COLORS['text_primary']};
    outline: none;
}}
QListView, QTreeView, QTableView {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent']};
    selection-color: #0d1117;
}}
QToolTip {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    padding: 4px;
}}
QLineEdit:disabled, QTextEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_muted']};
    border: 1px solid {COLORS['border']};
}}
QProgressBar {{
    background-color: {COLORS['bg_input']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
    color: transparent;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {COLORS['accent']}, stop:1 {COLORS['purple']});
    border-radius: 4px;
}}
QScrollBar:vertical {{
    background: {COLORS['bg_dark']};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {COLORS['border']};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {COLORS['text_muted']};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    background: {COLORS['bg_dark']};
    height: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:horizontal {{
    background: {COLORS['border']};
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {COLORS['text_muted']};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    background-color: {COLORS['bg_card']};
}}
QTabBar::tab {{
    background-color: {COLORS['bg_input']};
    color: {COLORS['text_secondary']};
    padding: 8px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
    font-size: 13px;
}}
QTabBar::tab:selected {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border-bottom: 2px solid {COLORS['accent']};
}}
QTabBar::tab:hover {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
}}
QLabel#title {{
    font-size: 22px;
    font-weight: bold;
    color: {COLORS['text_primary']};
}}
QLabel#subtitle {{
    font-size: 13px;
    color: {COLORS['text_secondary']};
}}
QLabel#step_label {{
    font-size: 12px;
    color: {COLORS['text_secondary']};
    padding: 2px 0;
}}
QLabel#step_done {{
    font-size: 12px;
    color: {COLORS['success']};
    padding: 2px 0;
}}
QLabel#step_active {{
    font-size: 12px;
    color: {COLORS['accent']};
    font-weight: bold;
    padding: 2px 0;
}}
QLabel#step_error {{
    font-size: 12px;
    color: {COLORS['error']};
    padding: 2px 0;
}}
QFrame#separator {{
    background-color: {COLORS['border']};
    max-height: 1px;
}}
QCheckBox {{
    color: {COLORS['text_primary']};
    spacing: 8px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS['border']};
    border-radius: 3px;
    background-color: {COLORS['bg_input']};
}}
QCheckBox::indicator:checked {{
    background-color: {COLORS['accent']};
    border-color: {COLORS['accent']};
}}
QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS['bg_input']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 4px 8px;
    color: {COLORS['text_primary']};
}}
QStatusBar {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_secondary']};
    border-top: 1px solid {COLORS['border']};
    font-size: 12px;
}}
"""

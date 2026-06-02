"""
视频自动化工作流 - 桌面 GUI 应用
基于 PyQt5，支持完整的点击操作流程
"""
import sys
import os
import json
import shutil
import time
import threading
from pathlib import Path
from typing import Optional

# 确保项目路径在 sys.path 中
APP_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(APP_DIR))

# API Keys should be set via environment variables
# os.environ["DEEPSEEK_API_KEY"] should be configured externally
# os.environ["MINIMAX_API_KEY"] should be configured externally
# os.environ["PEXELS_API_KEY"] should be configured externally

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QTextEdit, QProgressBar,
    QFileDialog, QComboBox, QGroupBox, QSplitter, QFrame,
    QScrollArea, QTabWidget, QCheckBox, QSpinBox, QDoubleSpinBox,
    QMessageBox, QStatusBar, QToolBar, QAction, QSizePolicy,
    QGridLayout, QSlider,
)
from PyQt5.QtCore import (
    Qt, QThread, pyqtSignal, QTimer, QSize, QPropertyAnimation,
    QEasingCurve, pyqtProperty,
)

from src.gui.styles import COLORS, STYLE_SHEET
from src.gui.widgets import StepIndicator
from src.gui.worker import WorkerThread

from PyQt5.QtGui import (
    QFont, QColor, QPalette, QIcon, QPixmap, QPainter,
    QLinearGradient, QBrush, QPen, QFontDatabase,
)


# ─────────────────────────────────────────────
# 颜色主题（深色科技风）
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
# 工作线程
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 步骤状态指示器
# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
# 主窗口
# ─────────────────────────────────────────────
class VideoAutomationApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频自动化工作流  |  AI Video Pipeline")
        self.setMinimumSize(1024, 700)
        self.resize(1200, 820)
        self.setStyleSheet(STYLE_SHEET)

        self._worker = None
        self._result_path = None
        self._project_root = None

        self._init_ui()
        self._init_status_bar()

    # ── UI 初始化 ──
    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # 顶部 Header
        header = self._make_header()
        root_layout.addWidget(header)

        # 主体内容（左右分栏）
        body = QSplitter(Qt.Horizontal)
        body.setHandleWidth(1)
        body.setStyleSheet(f"QSplitter::handle {{ background: {COLORS['border']}; }}")

        # 左侧：配置面板
        left_panel = self._make_left_panel()
        body.addWidget(left_panel)

        # 右侧：日志 + 结果
        right_panel = self._make_right_panel()
        body.addWidget(right_panel)

        body.setSizes([480, 720])
        root_layout.addWidget(body, 1)

    def _make_header(self):
        header = QWidget()
        header.setFixedHeight(72)
        header.setStyleSheet(f"""
            QWidget {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #1a1f35, stop:0.5 #161b22, stop:1 #1a1f35);
                border-bottom: 1px solid {COLORS['border']};
            }}
        """)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(24, 0, 24, 0)

        # Logo + 标题
        title_layout = QVBoxLayout()
        title_layout.setSpacing(2)
        title = QLabel("🎬  AI Video Pipeline")
        title.setObjectName("title")
        title.setStyleSheet(f"font-size: 20px; font-weight: bold; color: {COLORS['text_primary']}; background: transparent;")
        subtitle = QLabel("PDF → DeepSeek 脚本 → Minimax TTS → 自动渲染视频")
        subtitle.setObjectName("subtitle")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLORS['text_secondary']}; background: transparent;")
        title_layout.addWidget(title)
        title_layout.addWidget(subtitle)
        layout.addLayout(title_layout)

        layout.addStretch()

        # API 状态指示
        api_layout = QHBoxLayout()
        api_layout.setSpacing(12)
        self._deepseek_badge = self._make_badge("DeepSeek", COLORS['success'])
        self._minimax_badge = self._make_badge("Minimax", COLORS['success'])
        api_layout.addWidget(self._deepseek_badge)
        api_layout.addWidget(self._minimax_badge)
        layout.addLayout(api_layout)

        return header

    def _make_badge(self, text: str, color: str) -> QLabel:
        badge = QLabel(f"● {text}")
        badge.setStyleSheet(f"""
            QLabel {{
                background-color: transparent;
                color: {color};
                font-size: 12px;
                padding: 4px 10px;
                border: 1px solid {color}40;
                border-radius: 10px;
            }}
        """)
        return badge

    def _make_left_panel(self):
        panel = QWidget()
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(480)
        panel.setStyleSheet(f"background-color: {COLORS['bg_card']}; border-right: 1px solid {COLORS['border']};")

        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # Tab 切换：新建项目 / 增量更新
        tabs = QTabWidget()
        tabs.setStyleSheet(f"QTabWidget::pane {{ border: none; background: transparent; }}")

        # Tab 1: 新建项目
        new_tab = self._make_new_project_tab()
        tabs.addTab(new_tab, "  ✦ 新建项目  ")

        # Tab 2: 增量更新
        inc_tab = self._make_incremental_tab()
        tabs.addTab(inc_tab, "  ⟳ 增量更新  ")

        # Tab 3: 工作台
        workbench_tab = self._make_workbench_tab()
        tabs.addTab(workbench_tab, "  🛠 工作台  ")

        layout.addWidget(tabs, 1)

        # 步骤进度指示
        steps_group = QGroupBox("构建步骤")
        steps_layout = QVBoxLayout(steps_group)
        steps_layout.setSpacing(6)
        self._step_indicator = StepIndicator([
            "P1  抽取 PDF 文本",
            "P2  DeepSeek 生成脚本",
            "P3  Minimax TTS 语音",
            "S1  Whisper 字幕对齐",
            "S2  生成 Manifest",
            "S3  Visual Plan 规划",
            "S4  素材处理",
            "S5  分段视频渲染",
            "S6  拼接合成输出",
        ])
        steps_layout.addWidget(self._step_indicator)
        layout.addWidget(steps_group)

        return panel

    def _make_new_project_tab(self):
        # Wrap content in scroll area for small window support
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
        """)

        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(10)

        # ── PDF 输入 ──
        pdf_group = QGroupBox("输入文件")
        pdf_layout = QVBoxLayout(pdf_group)

        pdf_row = QHBoxLayout()
        self._pdf_path_edit = QLineEdit()
        self._pdf_path_edit.setPlaceholderText("选择 PDF 文件...")
        self._pdf_path_edit.setReadOnly(True)
        btn_browse = QPushButton("浏览")
        btn_browse.setObjectName("btn_secondary")
        btn_browse.setFixedWidth(64)
        btn_browse.clicked.connect(self._browse_pdf)
        pdf_row.addWidget(self._pdf_path_edit)
        pdf_row.addWidget(btn_browse)
        pdf_layout.addLayout(pdf_row)

        # 项目名称
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("项目名称:"))
        self._project_name_edit = QLineEdit("my_video")
        self._project_name_edit.setPlaceholderText("项目名称（英文）")
        name_row.addWidget(self._project_name_edit)
        pdf_layout.addLayout(name_row)

        layout.addWidget(pdf_group)

        # ── API 配置 ──
        api_group = QGroupBox("API 配置")
        api_layout = QGridLayout(api_group)
        api_layout.setColumnStretch(1, 1)

        api_layout.addWidget(QLabel("DeepSeek Key:"), 0, 0)
        self._deepseek_key_edit = QLineEdit(os.environ.get("DEEPSEEK_API_KEY", ""))
        self._deepseek_key_edit.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self._deepseek_key_edit, 0, 1)

        api_layout.addWidget(QLabel("Minimax Key:"), 1, 0)
        self._minimax_key_edit = QLineEdit(os.environ.get("MINIMAX_API_KEY", ""))
        self._minimax_key_edit.setEchoMode(QLineEdit.Password)
        api_layout.addWidget(self._minimax_key_edit, 1, 1)

        api_layout.addWidget(QLabel("Pexels Key:"), 2, 0)
        self._pexels_key_edit = QLineEdit(os.environ.get("PEXELS_API_KEY", ""))
        self._pexels_key_edit.setEchoMode(QLineEdit.Password)
        self._pexels_key_edit.setPlaceholderText("用于搜索真实视频/图片素材")
        api_layout.addWidget(self._pexels_key_edit, 2, 1)

        layout.addWidget(api_group)

        # ── 生成设置 ──
        settings_group = QGroupBox("生成设置")
        settings_layout = QGridLayout(settings_group)
        settings_layout.setColumnStretch(1, 1)

        settings_layout.addWidget(QLabel("画面比例:"), 0, 0)
        self._ratio_combo = QComboBox()
        self._ratio_combo.addItems(["9:16 (竖屏/抖音)", "16:9 (横屏/B站)", "1:1 (方形)"])
        settings_layout.addWidget(self._ratio_combo, 0, 1)

        settings_layout.addWidget(QLabel("TTS 声音:"), 1, 0)
        self._voice_combo = QComboBox()
        self._voice_combo.addItems(["male-qn-qingse (男声·清澈)", "female-tianmei (女声·甜美)", "female-qnshaonv (女声·少女)", "male-qn-jingying (男声·精英)"])
        settings_layout.addWidget(self._voice_combo, 1, 1)

        settings_layout.addWidget(QLabel("Whisper 模型:"), 2, 0)
        self._whisper_combo = QComboBox()
        self._whisper_combo.addItems(["base (快速)", "small (均衡)", "medium (精准)"])
        settings_layout.addWidget(self._whisper_combo, 2, 1)

        settings_layout.addWidget(QLabel("目标时长:"), 3, 0)
        self._duration_combo = QComboBox()
        self._duration_combo.addItems(["1分钟", "2分钟", "3分钟"])
        settings_layout.addWidget(self._duration_combo, 3, 1)

        self._subtitle_effects_cb = QCheckBox("启用字幕高亮效果")
        self._subtitle_effects_cb.setChecked(True)
        settings_layout.addWidget(self._subtitle_effects_cb, 4, 0, 1, 2)

        layout.addWidget(settings_group)
        # ── 素材策略 ──
        asset_group = QGroupBox("素材策略")
        asset_layout = QVBoxLayout(asset_group)
        asset_layout.setSpacing(6)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("生成模式:"))
        self._material_mode_combo = QComboBox()
        self._material_mode_combo.addItems(["auto (自动混合)", "ai_preferred (AI优先)", "ai_only (纯AI生成)"])
        mode_row.addWidget(self._material_mode_combo)
        mode_row.addStretch()
        asset_layout.addLayout(mode_row)

        priority_label = QLabel(
            "优先级： ① PDF图表 → ② Pexels视频 → ③ Pexels图片 → ④ AI生成 → ⑤ 模板兜底"
        )
        priority_label.setStyleSheet(
            f"color: {COLORS['text_secondary']}; font-size: 11px; "
            f"background: {COLORS['bg_input']}; padding: 8px; border-radius: 4px;"
        )
        priority_label.setWordWrap(True)
        asset_layout.addWidget(priority_label)

        self._pexels_video_cb = QCheckBox("② Pexels 视频（真实视频素材，与内容关键词匹配）")
        self._pexels_video_cb.setChecked(True)
        asset_layout.addWidget(self._pexels_video_cb)

        self._pexels_photo_cb = QCheckBox("③ Pexels 图片（视频无结果时自动 fallback）")
        self._pexels_photo_cb.setChecked(True)
        asset_layout.addWidget(self._pexels_photo_cb)

        self._ai_image_cb = QCheckBox("④ AI 图片生成（DALL-E 3，较慢且消耗 API）")
        self._ai_image_cb.setChecked(False)
        self._ai_image_cb.setStyleSheet(f"color: {COLORS['text_secondary']};")
        asset_layout.addWidget(self._ai_image_cb)

        quality_row = QHBoxLayout()
        quality_row.addWidget(QLabel("Pexels 视频质量:"))
        self._pexels_quality_combo = QComboBox()
        self._pexels_quality_combo.addItems(["hd (1080p, 推荐)", "sd (720p, 省流量)", "uhd (4K, 最高质量)"])
        self._pexels_quality_combo.setFixedWidth(160)
        quality_row.addWidget(self._pexels_quality_combo)
        quality_row.addStretch()
        asset_layout.addLayout(quality_row)

        layout.addWidget(asset_group)

        # ── 开始按鈕 ──
        self._btn_start = QPushButton("▶  开始生成视频")
        self._btn_start.setObjectName("btn_large")
        self._btn_start.clicked.connect(self._start_build)
        layout.addWidget(self._btn_start)

        self._btn_stop = QPushButton("⏹  停止")
        self._btn_stop.setObjectName("btn_secondary")
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._stop_build)
        layout.addWidget(self._btn_stop)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    def _make_incremental_tab(self):
        # Wrap content in scroll area for small window support
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
        """)

        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(10)

        info_label = QLabel(
            "增量更新：修改字幕后，只重渲变动的片段，\n"
            "大幅节省时间和 API 费用。"
        )
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; "
                                  f"background: {COLORS['bg_input']}; padding: 10px; border-radius: 6px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 项目目录
        proj_group = QGroupBox("选择项目")
        proj_layout = QVBoxLayout(proj_group)

        proj_row = QHBoxLayout()
        self._inc_project_edit = QLineEdit()
        self._inc_project_edit.setPlaceholderText("选择已有项目目录...")
        self._inc_project_edit.setReadOnly(True)
        btn_browse_proj = QPushButton("浏览")
        btn_browse_proj.setObjectName("btn_secondary")
        btn_browse_proj.setFixedWidth(64)
        btn_browse_proj.clicked.connect(self._browse_project)
        proj_row.addWidget(self._inc_project_edit)
        proj_row.addWidget(btn_browse_proj)
        proj_layout.addLayout(proj_row)

        # 新 SRT 文件
        srt_row = QHBoxLayout()
        srt_row.addWidget(QLabel("新 SRT:"))
        self._new_srt_edit = QLineEdit()
        self._new_srt_edit.setPlaceholderText("选择修改后的 SRT 文件（可选）...")
        self._new_srt_edit.setReadOnly(True)
        btn_browse_srt = QPushButton("浏览")
        btn_browse_srt.setObjectName("btn_secondary")
        btn_browse_srt.setFixedWidth(64)
        btn_browse_srt.clicked.connect(self._browse_srt)
        srt_row.addWidget(self._new_srt_edit)
        srt_row.addWidget(btn_browse_srt)
        proj_layout.addLayout(srt_row)

        layout.addWidget(proj_group)

        # 选项
        opt_group = QGroupBox("增量选项")
        opt_layout = QVBoxLayout(opt_group)
        
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("生成模式:"))
        self._inc_material_mode_combo = QComboBox()
        self._inc_material_mode_combo.addItems(["auto (自动混合)", "ai_preferred (AI优先)", "ai_only (纯AI生成)"])
        mode_row.addWidget(self._inc_material_mode_combo)
        mode_row.addStretch()
        opt_layout.addLayout(mode_row)

        duration_row = QHBoxLayout()
        duration_row.addWidget(QLabel("增量时长限制:"))
        self._inc_duration_combo = QComboBox()
        self._inc_duration_combo.addItems(["1 分钟", "2 分钟", "3 分钟"])
        self._inc_duration_combo.setCurrentIndex(0)
        duration_row.addWidget(self._inc_duration_combo)
        duration_row.addStretch()
        opt_layout.addLayout(duration_row)

        self._dry_run_check = QCheckBox("Dry Run（预览变更，不实际执行）")
        self._full_rebuild_check = QCheckBox("强制全量重建")
        opt_layout.addWidget(self._dry_run_check)
        opt_layout.addWidget(self._full_rebuild_check)
        layout.addWidget(opt_group)

        # Diff 预览
        diff_group = QGroupBox("变更预览")
        diff_layout = QVBoxLayout(diff_group)
        self._diff_preview = QTextEdit()
        self._diff_preview.setReadOnly(True)
        self._diff_preview.setMinimumHeight(60)
        self._diff_preview.setMaximumHeight(120)
        self._diff_preview.setPlaceholderText("运行后显示 Diff 摘要...")
        diff_layout.addWidget(self._diff_preview)
        layout.addWidget(diff_group)

        self._btn_inc_start = QPushButton("⟳  开始增量更新")
        self._btn_inc_start.setObjectName("btn_large")
        self._btn_inc_start.clicked.connect(self._start_incremental)
        layout.addWidget(self._btn_inc_start)

        layout.addStretch()
        scroll.setWidget(widget)
        return scroll

    def _make_workbench_tab(self):
        # Wrap content in scroll area for small window support
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
        """)

        widget = QWidget()
        widget.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 8, 4, 4)
        layout.setSpacing(10)

        info_label = QLabel(
            "工作台：配置项目的全局和分段 Pixelle 工作流。"
        )
        info_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px; "
                                  f"background: {COLORS['bg_input']}; padding: 10px; border-radius: 6px;")
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        # 项目目录
        proj_group = QGroupBox("选择项目")
        proj_layout = QVBoxLayout(proj_group)

        proj_row = QHBoxLayout()
        self._wb_project_edit = QLineEdit()
        self._wb_project_edit.setPlaceholderText("选择已有项目目录...")
        self._wb_project_edit.setReadOnly(True)
        btn_browse_proj = QPushButton("浏览")
        btn_browse_proj.setObjectName("btn_secondary")
        btn_browse_proj.setFixedWidth(64)
        btn_browse_proj.clicked.connect(self._browse_wb_project)
        proj_row.addWidget(self._wb_project_edit)
        proj_row.addWidget(btn_browse_proj)
        proj_layout.addLayout(proj_row)
        layout.addWidget(proj_group)

        # Pixelle Workflow
        pixelle_group = QGroupBox("Pixelle 工作流配置")
        pixelle_layout = QVBoxLayout(pixelle_group)
        
        # Global Default
        global_row = QHBoxLayout()
        global_row.addWidget(QLabel("全局默认工作流:"))
        self._wb_global_combo = QComboBox()
        self._populate_pixelle_combo(self._wb_global_combo, none_text="None (Fallback)")
        self._wb_global_combo.currentIndexChanged.connect(self._on_wb_changed)
        global_row.addWidget(self._wb_global_combo)
        pixelle_layout.addLayout(global_row)

        # Per-segment Overrides
        self._wb_segments_layout = QVBoxLayout()
        pixelle_layout.addLayout(self._wb_segments_layout)

        layout.addWidget(pixelle_group)

        # Save Button
        self._btn_wb_save = QPushButton("保存 Workbench Session")
        self._btn_wb_save.setObjectName("btn_large")
        self._btn_wb_save.setEnabled(False)
        self._btn_wb_save.clicked.connect(self._save_wb_session)
        layout.addWidget(self._btn_wb_save)

        layout.addStretch()
        scroll.setWidget(widget)
        
        self._wb_segment_combos = {}
        self._wb_current_session = None
        self._wb_paths = None
        
        return scroll

    def _populate_pixelle_combo(self, combo: QComboBox, none_text: str = "None"):
        combo.clear()
        combo.addItem(none_text, None)
        
        workflows = [
            ("digital_human", "Digital Human"),
            ("i2v", "Image to Video (I2V)"),
            ("action_transfer", "Action Transfer")
        ]
        
        try:
            from pixelle_snapshot.adapters import is_capability_available
        except ImportError:
            is_capability_available = lambda x: False
            
        for wf_id, wf_name in workflows:
            available = is_capability_available(wf_id)
            if available:
                combo.addItem(wf_name, wf_id)
            else:
                combo.addItem(f"{wf_name} (Unavailable)", wf_id)
                idx = combo.count() - 1
                
                # Type-safe way to disable the item
                from PyQt5.QtGui import QStandardItemModel
                model = combo.model()
                if isinstance(model, QStandardItemModel):
                    item = model.item(idx)
                    if item is not None:
                        item.setEnabled(False)

    def _browse_wb_project(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择项目目录",
            str(APP_DIR / "projects")
        )
        if path:
            self._wb_project_edit.setText(path)
            self._log(f"已选择工作台项目: {path}")
            self._load_wb_session(path)

    def _load_wb_session(self, project_path: str):
        from src.workbench.state import init_workbench, load_session
        try:
            self._wb_paths = init_workbench(project_path)
            self._wb_current_session = load_session(self._wb_paths)
            
            # Update global combo
            global_wf = self._wb_current_session.pixelle_default_workflow
            idx = self._wb_global_combo.findData(global_wf)
            if idx >= 0:
                self._wb_global_combo.setCurrentIndex(idx)
            else:
                self._wb_global_combo.setCurrentIndex(0)
                
            # Clear existing segment combos
            for i in reversed(range(self._wb_segments_layout.count())): 
                widget = self._wb_segments_layout.itemAt(i).widget()
                if widget:
                    widget.setParent(None)
            self._wb_segment_combos.clear()
            
            # Load segments from manifest to show overrides
            import json
            manifest_path = self._wb_paths.manifest_json
            if manifest_path.exists():
                manifest_data = json.loads(manifest_path.read_text())
                segments = manifest_data.get("segments", [])
                if segments:
                    self._wb_segments_layout.addWidget(QLabel("分段工作流覆盖 (Overrides):"))
                    for seg in segments:
                        seg_key = seg.get("segment_key")
                        if not seg_key:
                            continue
                        
                        row = QHBoxLayout()
                        row.addWidget(QLabel(f"[{seg_key}] {seg.get('text', '')[:20]}..."))
                        
                        combo = QComboBox()
                        self._populate_pixelle_combo(combo, none_text="Use Global Default")
                        
                        override_wf = self._wb_current_session.pixelle_segment_overrides.get(seg_key)
                        idx = combo.findData(override_wf)
                        if idx >= 0:
                            combo.setCurrentIndex(idx)
                        else:
                            combo.setCurrentIndex(0)
                            
                        combo.currentIndexChanged.connect(self._on_wb_changed)
                        row.addWidget(combo)
                        
                        widget = QWidget()
                        widget.setLayout(row)
                        self._wb_segments_layout.addWidget(widget)
                        self._wb_segment_combos[seg_key] = combo
            
            self._btn_wb_save.setEnabled(False)
            self._log(f"<span style='color:{COLORS['success']};'>✓ 工作台 Session 已加载</span>")
            
        except Exception as e:
            self._log(f"<span style='color:{COLORS['error']};'>✗ 加载工作台失败: {e}</span>")
            self._btn_wb_save.setEnabled(False)

    def _on_wb_changed(self):
        if self._wb_current_session:
            self._btn_wb_save.setEnabled(True)

    def _save_wb_session(self):
        if not self._wb_current_session or not self._wb_paths:
            return
            
        from src.workbench.state import save_session
        
        # Update global default
        global_wf = self._wb_global_combo.currentData()
        self._wb_current_session.pixelle_default_workflow = global_wf
        
        # Update segment overrides
        overrides = {}
        for seg_key, combo in self._wb_segment_combos.items():
            wf = combo.currentData()
            if wf is not None:
                overrides[seg_key] = wf
                
        self._wb_current_session.pixelle_segment_overrides = overrides
        
        try:
            save_session(self._wb_paths, self._wb_current_session)
            self._btn_wb_save.setEnabled(False)
            self._log(f"<span style='color:{COLORS['success']};'>✓ 工作台 Session 已保存</span>")
        except Exception as e:
            self._log(f"<span style='color:{COLORS['error']};'>✗ 保存工作台失败: {e}</span>")

    def _make_right_panel(self):
        panel = QWidget()
        panel.setStyleSheet(f"background-color: {COLORS['bg_dark']};")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        # 进度条
        progress_layout = QHBoxLayout()
        self._progress_label = QLabel("就绪")
        self._progress_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setFixedHeight(8)
        progress_layout.addWidget(self._progress_label, 1)
        layout.addLayout(progress_layout)
        layout.addWidget(self._progress_bar)

        # 日志输出
        log_group = QGroupBox("运行日志")
        log_layout = QVBoxLayout(log_group)
        self._log_text = QTextEdit()
        self._log_text.setReadOnly(True)
        self._log_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_dark']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                font-family: "Consolas", "Courier New", monospace;
                font-size: 12px;
                color: {COLORS['text_primary']};
                padding: 8px;
            }}
        """)
        log_layout.addWidget(self._log_text)

        btn_clear_log = QPushButton("清空日志")
        btn_clear_log.setObjectName("btn_secondary")
        btn_clear_log.setFixedWidth(80)
        btn_clear_log.clicked.connect(self._log_text.clear)
        log_layout.addWidget(btn_clear_log, 0, Qt.AlignRight)
        layout.addWidget(log_group, 1)

        # 结果区域
        result_group = QGroupBox("输出结果")
        result_layout = QVBoxLayout(result_group)

        result_info_layout = QHBoxLayout()
        self._result_path_label = QLabel("尚未生成视频")
        self._result_path_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self._result_path_label.setWordWrap(True)
        result_info_layout.addWidget(self._result_path_label, 1)
        result_layout.addLayout(result_info_layout)

        result_btn_layout = QHBoxLayout()
        self._btn_open_video = QPushButton("▶  播放视频")
        self._btn_open_video.setObjectName("btn_success")
        self._btn_open_video.setEnabled(False)
        self._btn_open_video.clicked.connect(self._open_video)
        result_btn_layout.addWidget(self._btn_open_video)

        self._btn_open_folder = QPushButton("📁  打开目录")
        self._btn_open_folder.setObjectName("btn_secondary")
        self._btn_open_folder.setEnabled(False)
        self._btn_open_folder.clicked.connect(self._open_folder)
        result_btn_layout.addWidget(self._btn_open_folder)

        self._btn_copy_path = QPushButton("复制路径")
        self._btn_copy_path.setObjectName("btn_secondary")
        self._btn_copy_path.setEnabled(False)
        self._btn_copy_path.clicked.connect(self._copy_path)
        result_btn_layout.addWidget(self._btn_copy_path)

        result_btn_layout.addStretch()
        result_layout.addLayout(result_btn_layout)
        layout.addWidget(result_group)

        return panel

    def _init_status_bar(self):
        self.statusBar().showMessage("就绪  |  DeepSeek ✓  Minimax ✓")

    # ── 事件处理 ──
    def _browse_pdf(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 PDF 文件", str(Path.home()),
            "PDF 文件 (*.pdf);;所有文件 (*)"
        )
        if path:
            self._pdf_path_edit.setText(path)
            # 自动填充项目名
            stem = Path(path).stem.replace(" ", "_").lower()
            self._project_name_edit.setText(stem[:30])
            self._log(f"已选择 PDF: {path}")

    def _browse_project(self):
        path = QFileDialog.getExistingDirectory(
            self, "选择项目目录",
            str(APP_DIR / "projects")
        )
        if path:
            self._inc_project_edit.setText(path)
            self._log(f"已选择项目: {path}")

    def _browse_srt(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择 SRT 文件", str(Path.home()),
            "SRT 字幕 (*.srt);;所有文件 (*)"
        )
        if path:
            self._new_srt_edit.setText(path)

    def _log(self, msg: str, color: str = None):
        """向日志区域追加消息"""
        if color:
            html = f'<span style="color:{color};">{msg}</span>'
            self._log_text.append(html)
        else:
            self._log_text.append(msg)
        # 滚动到底部
        scrollbar = self._log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _set_step(self, index: int, status: str):
        self._step_indicator.set_status(index, status)

    def _set_progress(self, value: int, label: str = None):
        self._progress_bar.setValue(value)
        if label:
            self._progress_label.setText(label)

    # ── 全量构建 ──
    def _start_build(self):
        pdf_path = self._pdf_path_edit.text().strip()
        if not pdf_path or not Path(pdf_path).exists():
            QMessageBox.warning(self, "错误", "请先选择有效的 PDF 文件！")
            return

        project_name = self._project_name_edit.text().strip() or "my_video"
        deepseek_key = self._deepseek_key_edit.text().strip()
        minimax_key = self._minimax_key_edit.text().strip()

        if not deepseek_key:
            QMessageBox.warning(self, "错误", "请输入 DeepSeek API Key！")
            return
        if not minimax_key:
            QMessageBox.warning(self, "错误", "请输入 Minimax API Key！")
            return

        pexels_key = self._pexels_key_edit.text().strip()
        enable_pexels_video = self._pexels_video_cb.isChecked()
        enable_pexels_photo = self._pexels_photo_cb.isChecked()
        enable_ai_image = self._ai_image_cb.isChecked()

        # 更新环境变量
        os.environ["DEEPSEEK_API_KEY"] = deepseek_key
        os.environ["MINIMAX_API_KEY"] = minimax_key
        if pexels_key:
            os.environ["PEXELS_API_KEY"] = pexels_key

        # 获取设置
        ratio_map = {"9:16 (竖屏/抖音)": "9:16", "16:9 (横屏/B站)": "16:9", "1:1 (方形)": "1:1"}
        aspect_ratio = ratio_map.get(self._ratio_combo.currentText(), "9:16")
        voice_map = {
            "male-qn-qingse (男声·清澈)": "male-qn-qingse",
            "female-tianmei (女声·甜美)": "female-tianmei",
            "female-qnshaonv (女声·少女)": "female-qnshaonv",
            "male-qn-jingying (男声·精英)": "male-qn-jingying",
        }
        tts_voice = voice_map.get(self._voice_combo.currentText(), "male-qn-qingse")
        whisper_map = {"base (快速)": "base", "small (均衡)": "small", "medium (精准)": "medium"}
        whisper_model = whisper_map.get(self._whisper_combo.currentText(), "base")
        
        material_mode_map = {"auto (自动混合)": "auto", "ai_preferred (AI优先)": "ai_preferred", "ai_only (纯AI生成)": "ai_only"}
        material_mode = material_mode_map.get(self._material_mode_combo.currentText(), "auto")

        duration_map = {"1分钟": 1, "2分钟": 2, "3分钟": 3}
        target_duration = duration_map.get(self._duration_combo.currentText(), 1)

        # 重置 UI
        self._step_indicator.reset()
        self._log_text.clear()
        self._progress_bar.setValue(0)
        self._result_path = None
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_open_video.setEnabled(False)
        self._btn_open_folder.setEnabled(False)
        self._btn_copy_path.setEnabled(False)
        self._result_path_label.setText("正在生成...")

        self._log(f"<span style='color:{COLORS['accent']};'>{'='*50}</span>")
        self._log(f"<span style='color:{COLORS['accent']};'>开始构建: {project_name}</span>")
        self._log(f"PDF: {pdf_path}")
        self._log(f"比例: {aspect_ratio}  |  TTS: {tts_voice}  |  Whisper: {whisper_model}")
        pexels_status = f"✓ ({'视频+图片' if enable_pexels_video and enable_pexels_photo else '图片' if enable_pexels_photo else '视频' if enable_pexels_video else '关闭'})" if pexels_key else "✗ (未配置)"
        self._log(f"Pexels: {pexels_status}  |  AI图片: {'✓' if enable_ai_image else '✗'}  |  模式: {material_mode}  |  时长: {target_duration}分钟")
        self._log(f"<span style='color:{COLORS['accent']};'>{'='*50}</span>")

        self._worker = WorkerThread(
            self._run_full_build,
            pdf_path, project_name, aspect_ratio, tts_voice, whisper_model,
            pexels_key, enable_pexels_video, enable_pexels_photo, enable_ai_image, material_mode,
            target_duration=target_duration,
        )
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_finished)
        self._worker.start()

        self.statusBar().showMessage(f"正在构建: {project_name}...")

    def _run_full_build(self, pdf_path, project_name, aspect_ratio, tts_voice, whisper_model,
                        pexels_key="", enable_pexels_video=True, enable_pexels_photo=True, enable_ai_image=False, material_mode="auto", target_duration=1):
        """在工作线程中运行完整构建"""
        import json
        from src.steps.step_pdf import (
            extract_pdf_text, extract_pdf_images,
            generate_script_from_text, generate_tts_minimax,
            generate_tts, generate_tts_elevenlabs,
        )
        from src.steps.step1_align import run_step1
        from src.steps.step2_manifest import run_step2
        from src.steps.step3_visual_plan import run_step3
        from src.steps.step4_assets import run_step4
        from src.steps.step5_render import run_step5
        from src.steps.step6_concat import run_step6
        from src.core.models import GlobalStyle
        from src.integrations.minimax import MINIMAX_VOICES, MinimaxTTSError
        from src.core.api_config import ELEVENLABS_API_KEY

        project_root = str(APP_DIR / "projects" / project_name)
        self._project_root = project_root

        # 初始化目录
        for d in ["input", "extracted/images", "build", "render/segments",
                  "assets/generated", "assets/stock", "cache/visual_plans"]:
            Path(project_root, d).mkdir(parents=True, exist_ok=True)
        pdf_dest = str(Path(project_root) / "input" / "source.pdf")
        shutil.copy2(pdf_path, pdf_dest)

        def cb(msg):
            self._worker.progress_signal.emit(msg)

        STEP_TOTAL = 9

        # P1: PDF 抽取
        self._worker.step_signal.emit(0, "active")
        cb("抽取 PDF 文本...")
        content_md = str(Path(project_root) / "extracted" / "content.md")
        text = extract_pdf_text(pdf_dest, content_md)
        extract_pdf_images(pdf_dest, str(Path(project_root) / "extracted" / "images"))
        self._worker.step_signal.emit(0, "done")
        self._worker.percent_signal.emit(10)

        # P2: DeepSeek 脚本
        self._worker.step_signal.emit(1, "active")
        script_md = str(Path(project_root) / "input" / "script.md")
        script = generate_script_from_text(text, script_md, llm_model="deepseek-chat", progress_cb=cb)
        self._worker.step_signal.emit(1, "done")
        self._worker.percent_signal.emit(20)

        # P3: Minimax TTS
        self._worker.step_signal.emit(2, "active")
        voice_path = str(Path(project_root) / "input" / "voice_full.mp3")
        voice_id = MINIMAX_VOICES.get(tts_voice, MINIMAX_VOICES["default"])
        try:
            generate_tts_minimax(script_md, voice_path, voice_id=voice_id, progress_cb=cb)
        except MinimaxTTSError as minimax_error:
            if "insufficient balance" not in str(minimax_error).lower():
                raise

            cb("Minimax 余额不足，尝试 OpenAI TTS 兜底...")
            try:
                generate_tts(script_md, voice_path, voice="alloy", speed=1.0, progress_cb=cb)
            except Exception as openai_error:
                if ELEVENLABS_API_KEY:
                    cb("OpenAI TTS 失败，尝试 ElevenLabs TTS 兜底...")
                    generate_tts_elevenlabs(script_md, voice_path, progress_cb=cb)
                else:
                    raise RuntimeError(
                        "Minimax 余额不足，且 OpenAI/ElevenLabs 兜底不可用。"
                        "请充值 Minimax 或配置 OPENAI_API_KEY / ELEVENLABS_API_KEY。"
                    ) from openai_error
        self._worker.step_signal.emit(2, "done")
        self._worker.percent_signal.emit(32)

        # S1: Whisper SRT
        self._worker.step_signal.emit(3, "active")
        cb("Whisper 字幕对齐...")
        srt_path = str(Path(project_root) / "build" / "subtitle.srt")
        run_step1(audio_path=voice_path, output_srt=srt_path,
                  script_path=script_md, use_local_whisper=True, whisper_model=whisper_model)
        self._worker.step_signal.emit(3, "done")
        self._worker.percent_signal.emit(45)

        # S2: Manifest
        self._worker.step_signal.emit(4, "active")
        cb("生成 Manifest...")
        res_map = {"9:16": "1080x1920", "16:9": "1920x1080", "1:1": "1080x1080"}
        global_style = GlobalStyle(
            aspect_ratio=aspect_ratio,
            resolution=res_map.get(aspect_ratio, "1080x1920"),
            fps=30, font_size=48, subtitle_style="clean",
            enable_subtitle_effects=self._subtitle_effects_cb.isChecked(),
        )
        manifest_path = str(Path(project_root) / "build" / "manifest.json")
        duration_policy = {"target_duration_minutes": target_duration}
        manifest = run_step2(srt_path=srt_path, audio_path=voice_path,
                             project_id=project_name, output_manifest=manifest_path,
                             global_style=global_style, material_mode=material_mode,
                             duration_policy=duration_policy)
        self._worker.step_signal.emit(4, "done")
        self._worker.percent_signal.emit(55)

        # S3: Visual Plan
        self._worker.step_signal.emit(5, "active")
        cb("DeepSeek 生成 Visual Plan...")
        cache_dir = str(Path(project_root) / "cache" / "visual_plans")
        manifest = run_step3(manifest=manifest, output_manifest=manifest_path,
                             cache_dir=cache_dir, llm_model="deepseek-chat")
        self._worker.step_signal.emit(5, "done")
        self._worker.percent_signal.emit(65)

        # S4: 素材
        self._worker.step_signal.emit(6, "active")
        pexels_info = (
            f"Pexels {'视频+图片' if enable_pexels_video and enable_pexels_photo else '图片' if enable_pexels_photo else '视频' if enable_pexels_video else '关闭'}"
            if pexels_key else "Pexels 未配置"
        )
        cb(f"素材处理... ({pexels_info})")
        manifest = run_step4(
            manifest=manifest,
            output_manifest=manifest_path,
            project_root=project_root,
            pexels_api_key=pexels_key,
            enable_pexels_video=enable_pexels_video,
            enable_pexels_photo=enable_pexels_photo,
            enable_ai_image=enable_ai_image,
        )
        self._worker.step_signal.emit(6, "done")
        self._worker.percent_signal.emit(72)

        # S5: 渲染
        self._worker.step_signal.emit(7, "active")
        cb("FFmpeg 分段渲染...")
        segments_dir = str(Path(project_root) / "render" / "segments")
        manifest = run_step5(manifest=manifest, output_manifest=manifest_path,
                             segments_dir=segments_dir)
        self._worker.step_signal.emit(7, "done")
        self._worker.percent_signal.emit(90)

        # S6: 合成
        self._worker.step_signal.emit(8, "active")
        cb("拼接合成 final.mp4...")
        final_path = str(Path(project_root) / "render" / "final.mp4")
        manifest = run_step6(manifest=manifest, output_manifest=manifest_path,
                             output_video=final_path, audio_path=voice_path)
        self._worker.step_signal.emit(8, "done")
        self._worker.percent_signal.emit(100)

        if not Path(final_path).exists():
            raise RuntimeError("final.mp4 未生成")

        return final_path

    # ── 增量更新 ──
    def _start_incremental(self):
        project_path = self._inc_project_edit.text().strip()
        if not project_path or not Path(project_path).exists():
            QMessageBox.warning(self, "错误", "请先选择有效的项目目录！")
            return

        new_srt = self._new_srt_edit.text().strip() or None
        dry_run = self._dry_run_check.isChecked()
        full_rebuild = self._full_rebuild_check.isChecked()
        
        material_mode_map = {"auto (自动混合)": "auto", "ai_preferred (AI优先)": "ai_preferred", "ai_only (纯AI生成)": "ai_only"}
        material_mode = material_mode_map.get(self._inc_material_mode_combo.currentText(), "auto")

        duration_text = self._inc_duration_combo.currentText()
        try:
            duration_limit = int(duration_text.split()[0])
        except (ValueError, IndexError):
            duration_limit = 1

        self._log_text.clear()
        self._log(f"<span style='color:{COLORS['purple']};'>{'='*50}</span>")
        self._log(f"<span style='color:{COLORS['purple']};'>增量更新: {project_path}</span>")
        self._log(f"dry_run={dry_run}  full_rebuild={full_rebuild}  模式={material_mode}  时长限制={duration_limit}分钟")
        self._log(f"<span style='color:{COLORS['purple']};'>{'='*50}</span>")

        self._btn_inc_start.setEnabled(False)
        self._project_root = project_path

        self._worker = WorkerThread(
            self._run_incremental,
            project_path, new_srt, dry_run, full_rebuild, material_mode,
            duration_limit=duration_limit
        )
        self._worker.progress_signal.connect(self._on_progress)
        self._worker.finished_signal.connect(self._on_incremental_finished)
        self._worker.start()

    def _run_incremental(self, project_path, new_srt, dry_run, full_rebuild, material_mode="auto", duration_limit=1):
        """在工作线程中运行增量更新"""
        sys.path.insert(0, str(APP_DIR))
        from build_incremental import incremental_build

        def cb(msg):
            self._worker.progress_signal.emit(msg)

        result = incremental_build(
            project_root=project_path,
            new_srt_path=new_srt,
            dry_run=dry_run,
            full_rebuild=full_rebuild,
            llm_model="deepseek-chat",
            material_mode=material_mode,
            duration_policy={"target_duration_minutes": duration_limit},
        )
        # 返回 diff 摘要
        diff = result.diff
        added = len(diff.added) if diff else 0
        removed = len(diff.removed) if diff else 0
        text_changed = len([c for c in (diff.changed if diff else []) if c.change_type.name == 'TEXT']) if diff else 0
        timing_changed = len([c for c in (diff.changed if diff else []) if c.change_type.name == 'TIMING']) if diff else 0
        summary = (
            f"Diff 摘要:\n"
            f"  新增: {added} 段\n"
            f"  删除: {removed} 段\n"
            f"  TEXT 变更: {text_changed} 段\n"
            f"  TIMING 变更: {timing_changed} 段\n"
            f"  重渲: {result.rerendered_count} 段\n"
            f"  复用: {result.reused_count} 段\n"
            f"  输出: {result.final_video or '(dry run)'}"
        )
        return summary

    # ── 信号处理 ──
    def _on_progress(self, msg: str):
        self._log(f"  {msg}")
        self._progress_label.setText(msg[:60])

    def _on_finished(self, success: bool, result: str):
        self._btn_start.setEnabled(True)
        self._btn_stop.setEnabled(False)

        if success:
            self._result_path = result
            size_mb = Path(result).stat().st_size / 1024 / 1024 if Path(result).exists() else 0
            self._result_path_label.setText(f"✓ {result}\n({size_mb:.2f} MB)")
            self._result_path_label.setStyleSheet(f"color: {COLORS['success']}; font-size: 12px;")
            self._btn_open_video.setEnabled(True)
            self._btn_open_folder.setEnabled(True)
            self._btn_copy_path.setEnabled(True)
            self._log(f"<span style='color:{COLORS['success']};'>{'='*50}</span>")
            self._log(f"<span style='color:{COLORS['success']};'>✓ 构建完成！{result}</span>")
            self._log(f"<span style='color:{COLORS['success']};'>{'='*50}</span>")
            self.statusBar().showMessage(f"✓ 完成  |  {result}")
            self._set_progress(100, "完成")
        else:
            self._result_path_label.setText("✗ 构建失败")
            self._result_path_label.setStyleSheet(f"color: {COLORS['error']}; font-size: 12px;")
            self._log(f"<span style='color:{COLORS['error']};'>{'='*50}</span>")
            self._log(f"<span style='color:{COLORS['error']};'>✗ 构建失败:</span>")
            self._log(f"<span style='color:{COLORS['error']};'>{result[:500]}</span>")
            self._log(f"<span style='color:{COLORS['error']};'>{'='*50}</span>")
            self.statusBar().showMessage("✗ 构建失败")
            # 标记最后一个 active 步骤为 error
            for i, s in enumerate(self._step_indicator.statuses):
                if s == "active":
                    self._step_indicator.set_status(i, "error")

    def _on_incremental_finished(self, success: bool, result: str):
        self._btn_inc_start.setEnabled(True)
        if success:
            self._diff_preview.setPlainText(result)
            self._log(f"<span style='color:{COLORS['success']};'>增量更新完成</span>")
            self.statusBar().showMessage("✓ 增量更新完成")
            # 检查是否有输出视频
            for line in result.split('\n'):
                if '输出:' in line and 'dry run' not in line:
                    path = line.split('输出:')[-1].strip()
                    if Path(path).exists():
                        self._result_path = path
                        self._btn_open_video.setEnabled(True)
                        self._btn_open_folder.setEnabled(True)
                        self._btn_copy_path.setEnabled(True)
        else:
            self._log(f"<span style='color:{COLORS['error']};'>增量更新失败: {result[:300]}</span>")
            self.statusBar().showMessage("✗ 增量更新失败")

    def _stop_build(self):
        if self._worker and self._worker.isRunning():
            self._worker.terminate()
            self._log(f"<span style='color:{COLORS['warning']};'>⚠ 已停止构建</span>")
            self._btn_start.setEnabled(True)
            self._btn_stop.setEnabled(False)
            self.statusBar().showMessage("已停止")

    def _open_video(self):
        if self._result_path and Path(self._result_path).exists():
            os.system(f'xdg-open "{self._result_path}" &')

    def _open_folder(self):
        if self._project_root and Path(self._project_root).exists():
            os.system(f'xdg-open "{self._project_root}" &')

    def _copy_path(self):
        if self._result_path:
            QApplication.clipboard().setText(self._result_path)
            self.statusBar().showMessage(f"已复制路径: {self._result_path}")


# ─────────────────────────────────────────────
# 连接步骤信号（需要在 WorkerThread 中发出）
# ─────────────────────────────────────────────
def patch_worker_signals(app: VideoAutomationApp):
    """将 worker 的 step_signal 和 percent_signal 连接到 UI"""
    original_start = app._start_build

    def patched_start():
        original_start()
        if app._worker:
            app._worker.step_signal.connect(app._set_step)
            app._worker.percent_signal.connect(
                lambda v: app._set_progress(v)
            )

    app._start_build = patched_start


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────
def main():
    # 设置高 DPI
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)
    app.setApplicationName("AI Video Pipeline")
    app.setOrganizationName("VideoAutomation")

    # 设置应用字体
    font = QFont()
    font.setFamily("Noto Sans CJK SC")
    font.setPointSize(10)
    app.setFont(font)

    window = VideoAutomationApp()

    # 连接步骤信号
    def connect_worker_signals():
        if window._worker:
            window._worker.step_signal.connect(window._set_step)
            window._worker.percent_signal.connect(window._set_progress)

    original_start = window._start_build

    def patched_start():
        original_start()
        QTimer.singleShot(100, connect_worker_signals)

    window._start_build = patched_start

    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

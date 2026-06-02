from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel

class StepIndicator(QWidget):
    def __init__(self, steps, parent=None):
        super().__init__(parent)
        self.steps = steps
        self.statuses = ["pending"] * len(steps)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(0, 0, 0, 0)
        self.labels = []
        for i, step in enumerate(self.steps):
            label = QLabel(f"○  {step}")
            label.setObjectName("step_label")
            layout.addWidget(label)
            self.labels.append(label)

    def set_status(self, index: int, status: str):
        if index >= len(self.labels):
            return
        self.statuses[index] = status
        label = self.labels[index]
        step_name = self.steps[index]
        if status == "done":
            label.setText(f"✓  {step_name}")
            label.setObjectName("step_done")
        elif status == "active":
            label.setText(f"▶  {step_name}")
            label.setObjectName("step_active")
        elif status == "error":
            label.setText(f"✗  {step_name}")
            label.setObjectName("step_error")
        else:
            label.setText(f"○  {step_name}")
            label.setObjectName("step_label")
        # 强制刷新样式
        label.style().unpolish(label)
        label.style().polish(label)

    def reset(self):
        for i in range(len(self.steps)):
            self.set_status(i, "pending")

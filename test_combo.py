import sys
from PyQt5.QtWidgets import QApplication
from src.gui.app import VideoAutomationApp

app = QApplication(sys.argv)
window = VideoAutomationApp()
combo = window._duration_combo
items = [combo.itemText(i) for i in range(combo.count())]
print("Full:", items)
combo2 = window._inc_duration_combo
items2 = [combo2.itemText(i) for i in range(combo2.count())]
print("Inc:", items2)

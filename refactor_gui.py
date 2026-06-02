import re

with open('src/gui/app.py', 'r') as f:
    content = f.read()

# Remove COLORS
content = re.sub(r'COLORS = \{.*?\n\}\n', '', content, flags=re.DOTALL)

# Remove STYLE_SHEET
content = re.sub(r'STYLE_SHEET = f"""\n.*?QMainWindow.*?\}"""\n', '', content, flags=re.DOTALL)

# Remove WorkerThread
content = re.sub(r'class WorkerThread\(QThread\):.*?(?=\n# ─────────────────────────────────────────────\n# 步骤状态指示器)', '', content, flags=re.DOTALL)

# Remove StepIndicator
content = re.sub(r'class StepIndicator\(QWidget\):.*?(?=\n# ─────────────────────────────────────────────\n# 主窗口)', '', content, flags=re.DOTALL)

with open('src/gui/app.py', 'w') as f:
    f.write(content)

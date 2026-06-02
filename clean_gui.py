import re

with open('src/gui/app.py', 'r') as f:
    content = f.read()

content = re.sub(r'STYLE_SHEET = f"""\n.*?\}"""\n', '', content, flags=re.DOTALL)

with open('src/gui/app.py', 'w') as f:
    f.write(content)

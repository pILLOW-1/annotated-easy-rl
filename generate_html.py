#!/usr/bin/env python3
"""
Generate HTML pages from annotated Python files in labml.ai style.
Left column: theory/formulas (docs)
Right column: code implementation
"""

import re
import os
import html as html_mod
import markdown

KATEX_CSS = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css"
KATEX_JS = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"
AUTORENDER_JS = "https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js"

PYGMENTS_CSS = """
.highlight .c { color: #6272a4; }
.highlight .k { color: #ff79c6; }
.highlight .o { color: #ff4996; }
.highlight .p { color: #a8a8a2; }
.highlight .cm { color: #6272a4; }
.highlight .c1 { color: #6272a4; }
.highlight .kc { color: #ff79c6; }
.highlight .kd { color: #8be9fd; font-style: italic; }
.highlight .kn { color: #ff79c6; }
.highlight .kp { color: #ff79c6; }
.highlight .kr { color: #ff79c6; }
.highlight .kt { color: #8be9fd; }
.highlight .m { color: #bd93f9; }
.highlight .s, .highlight .s2, .highlight .s1 { color: #f1fa8c; }
.highlight .na { color: #50fa7b; }
.highlight .nb { color: #8be9fd; font-style: italic; }
.highlight .nc { color: #ffb86c; font-weight: bold; }
.highlight .nd { color: #9d93ff; }
.highlight .nf { color: #ffb86c; }
.highlight .nl { color: #8be9fd; font-style: italic; }
.highlight .nn { color: #f8f8f2; }
.highlight .nt { color: #ff79c6; }
.highlight .nv, .highlight .vc, .highlight .vg, .highlight .vi { color: #8be9fd; font-style: italic; }
.highlight .ow { color: #ff79c6; }
.highlight .n { color: #f8f8f2; }
.highlight .nx { color: #f8f8f2; }
.highlight .bp { color: #50fa7b; }
.highlight .fm { color: #ffb86c; font-style: italic; }
"""

CSS = """
html { font-size: 62.5%; }
body {
  font-size: 1.5em; line-height: 1.6; font-weight: 400;
  font-family: "SF Pro Text", "SF Pro Icons", "Helvetica Neue", "Helvetica", "Arial", sans-serif;
  margin: 0; padding: 0; color: #999; background: #1d2127;
}
a { color: #bbb; }
a:visited { color: #aaa; }
#container { background: #1d2127; position: relative; }
#background {
  position: absolute; top: 0; left: 40%; right: 0; bottom: 0;
  z-index: 0; display: none; background: #282a36;
  border-left: 1px solid #293340;
}
@media (min-width: 768px) { #background { display: block; } }

a.parent {
  text-transform: uppercase; font-weight: bold; font-size: 12px;
  margin-right: 10px; text-decoration: none; color: #ffffff;
}
a.parent:after { content: ">"; font-size: 14px; margin-left: 4px; color: #aaa; }

div.nav-bar {
  padding: 15px 25px; position: relative; z-index: 10;
  border-bottom: 1px solid #353745;
}
div.nav-bar h1 { margin: 10px 0 5px 0; font-size: 2.2rem; color: #fff; }
div.nav-bar p { margin: 0; font-size: 1.3rem; color: #aaa; }

div.section {
  position: relative; border-top: 1px solid #353745;
}
div.section:after { clear: both; content: ""; display: block; }
div.section:hover { background: #080a16; }
div.section:hover div.code { background: #080a16; }

div.section div.docs {
  box-sizing: border-box; padding: 10px 8px 1px 8px;
  vertical-align: top; text-align: left;
}
@media (min-width: 768px) {
  div.section div.docs { float: left; width: 40%; min-height: 5px; }
}
@media (min-width: 1024px) {
  div.section div.docs { padding: 10px 25px 1px 50px; }
}
div.section div.docs .section-link { position: relative; }
div.section div.docs .section-link a {
  font: 12px Arial; text-decoration: none; position: absolute;
  top: 3px; left: -20px; padding: 1px 2px; opacity: 0;
  -webkit-transition: opacity 0.2s linear; color: #454545;
}
div.section:hover div.docs .section-link a { opacity: 1; }
div.section div.docs .katex-display { overflow-x: auto; overflow-y: hidden; }
div.section div.docs img { max-width: 100%; }
div.section div.docs p, div.section div.docs li {
  color: #ccc; font-size: 1.4rem; line-height: 1.7;
}
div.section div.docs h1 { color: #fff; font-size: 2.4rem; margin: 20px 0 10px 0; }
div.section div.docs h2 { color: #fff; font-size: 2.0rem; margin: 18px 0 8px 0; }
div.section div.docs h3 { color: #fff; font-size: 1.7rem; margin: 15px 0 6px 0; }
div.section div.docs h4 { color: #e0e0e0; font-size: 1.5rem; margin: 12px 0 5px 0; }
div.section div.docs ul, div.section div.docs ol { padding-left: 20px; }
div.section div.docs li { margin: 4px 0; }
div.section div.docs code {
  background: #282a36; color: #ccc; padding: 0.2rem 0.5rem;
  margin: 0 0.2rem; font-size: 80%; border-radius: 4px;
  border: 1px solid #484a56;
}
div.section div.docs table {
  border-collapse: collapse; margin: 10px 0; width: 100%;
}
div.section div.docs th, div.section div.docs td {
  border: 1px solid #484a56; padding: 8px 12px; text-align: left;
}
div.section div.docs th { background: #282a36; color: #fff; }

div.section div.code {
  padding: 14px 8px 16px 15px; vertical-align: top;
  background: #282a36;
}
@media (min-width: 768px) {
  div.section div.code { margin-left: 40%; }
}
div.section div.code pre {
  font-size: 12px; white-space: pre;
  font-family: Monaco, Consolas, "Lucida Console", monospace;
  line-height: 18px; margin: 0; overflow-x: auto;
}
div.section div.code .lineno {
  width: 30px; display: inline-block; text-align: right;
  padding-right: 10px; opacity: 0.3; font-size: 10px;
  white-space: pre; color: #6272a4; user-select: none;
}

div.footer {
  margin-top: 25px; position: relative; padding: 10px 0;
  text-align: center; background: #30353d;
}
div.footer a { display: inline-block; margin: 5px; font-size: 1.3rem; }

.katex span { cursor: default; }
.mjx-chtml { color: #ccc; }
"""

REPO_URL = "https://github.com/pILLOW-1/annotated-easy-rl"


def highlight_python(code: str) -> str:
    # Protect math delimiters in code before escaping
    code_math_keys = {}
    code_math_counter = [0]

    def protect_code_math(m):
        key = f'__CODEMATH{code_math_counter[0]}__'
        code_math_keys[key] = m.group(0)
        code_math_counter[0] += 1
        return key

    code = re.sub(r'\$\$([\s\S]*?)\$\$', protect_code_math, code)
    code = re.sub(r'\$([^\$\n]+?)\$', protect_code_math, code)

    code = html_mod.escape(code)
    keywords = {'import', 'from', 'class', 'def', 'return', 'if', 'else', 'elif',
                'for', 'while', 'in', 'not', 'and', 'or', 'is', 'with', 'as',
                'try', 'except', 'finally', 'raise', 'pass', 'break', 'continue',
                'yield', 'lambda', 'assert', 'del', 'global', 'nonlocal'}
    builtins_set = {'True', 'False', 'None', 'self', 'int', 'float', 'str',
                     'list', 'dict', 'tuple', 'set', 'range', 'len', 'print', 'type',
                     'isinstance', 'issubclass', 'hasattr', 'getattr', 'setattr',
                     'zip', 'enumerate', 'map', 'filter', 'reversed', 'sorted',
                     'abs', 'max', 'min', 'sum', 'any', 'all', 'open', 'input'}
    tokens = []
    i = 0
    while i < len(code):
        if code[i:i+3] in ('"""', "'''"):
            q = code[i:i+3]
            j = code.find(q, i+3)
            if j == -1:
                tokens.append(('s', code[i:]))
                break
            tokens.append(('s', code[i:j+3]))
            i = j+3
        elif code[i] in ('"', "'"):
            q = code[i]
            j = i+1
            while j < len(code) and code[j] != q:
                if code[j] == '\\':
                    j += 1
                j += 1
            tokens.append(('s', code[i:j+1]))
            i = j+1
        elif code[i] == '#':
            j = code.find('\n', i)
            if j == -1:
                tokens.append(('c', code[i:]))
                break
            tokens.append(('c', code[i:j]))
            i = j
        elif code[i].isdigit() or (code[i] == '.' and i+1 < len(code) and code[i+1].isdigit()):
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] == '.'):
                j += 1
            tokens.append(('m', code[i:j]))
            i = j
        elif code[i].isalpha() or code[i] == '_':
            j = i
            while j < len(code) and (code[j].isalnum() or code[j] == '_'):
                j += 1
            word = code[i:j]
            if word in keywords:
                tokens.append(('k', word))
            elif word in builtins_set:
                tokens.append(('nb', word))
            elif j < len(code) and code[j] == '(':
                tokens.append(('nf', word))
            elif word[0].isupper():
                tokens.append(('nc', word))
            else:
                tokens.append(('n', word))
            i = j
        elif code[i] in ('+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|', '^', '~'):
            j = i
            while j < len(code) and code[j] in ('+', '-', '*', '/', '%', '=', '<', '>', '!', '&', '|', '^', '~'):
                j += 1
            tokens.append(('o', code[i:j]))
            i = j
        elif code[i] in ('(', ')', '[', ']', '{', '}', ',', '.', ':', ';', '@'):
            tokens.append(('p', code[i]))
            i += 1
        elif code[i] in (' ', '\t', '\n', '\r'):
            j = i
            while j < len(code) and code[j] in (' ', '\t', '\n', '\r'):
                j += 1
            tokens.append(('w', code[i:j]))
            i = j
        else:
            tokens.append(('w', code[i]))
            i += 1
    span_map = {'c': 'c', 's': 's2', 'k': 'k', 'nb': 'nb', 'nf': 'nf',
                'nc': 'nc', 'm': 'm', 'o': 'o', 'p': 'p', 'n': 'n'}
    result = []
    for ttype, tval in tokens:
        if ttype == 'w':
            result.append(tval)
        else:
            cls = span_map.get(ttype, 'n')
            result.append(f'<span class="{cls}">{tval}</span>')
    html_out = ''.join(result)

    # Restore math delimiters (already escaped, so restore as-is)
    for key, value in code_math_keys.items():
        html_out = html_out.replace(key, html_mod.escape(value))
    return html_out


# Math placeholder system for doc text
_math_placeholders = {}
_math_counter = 0

def _protect_math(text):
    """Replace all $...$ and $$...$$ with unique placeholders."""
    global _math_counter
    _math_placeholders.clear()
    _math_counter = 0

    # Convert \begin{align}...\end{align} to $$...$$
    text = re.sub(
        r'(?:^|\n)\s*\\begin\{align\}([\s\S]*?)\\end\{align\}',
        lambda m: f'\n$$\\begin{{align}}{m.group(1)}\\end{{align}}$$\n',
        text
    )

    # First protect $$...$$ (display math, possibly multiline)
    def replace_display(m):
        global _math_counter
        key = f'<!--MATH{_math_counter}-->'
        _math_placeholders[key] = m.group(0)
        _math_counter += 1
        return key

    text = re.sub(r'\$\$([\s\S]*?)\$\$', replace_display, text)

    # Then protect $...$ (inline math, single line)
    def replace_inline(m):
        global _math_counter
        key = f'<!--MATH{_math_counter}-->'
        _math_placeholders[key] = m.group(0)
        _math_counter += 1
        return key

    text = re.sub(r'\$([^\$\n]+?)\$', replace_inline, text)
    return text

def _restore_math(html_text):
    """Replace placeholders back with original math delimiters."""
    for key, value in _math_placeholders.items():
        html_text = html_text.replace(key, value)
        html_text = html_text.replace(html_mod.escape(key), value)
    return html_text


def render_markdown_to_html(md_text: str) -> str:
    """Convert markdown to HTML, preserving KaTeX delimiters."""
    protected = _protect_math(md_text)
    html_text = markdown.markdown(protected, extensions=['tables', 'fenced_code'])
    return _restore_math(html_text)


def _dedent(text: str) -> str:
    """Remove common leading whitespace from text."""
    lines = text.split('\n')
    non_empty = [line for line in lines if line.strip()]
    if not non_empty:
        return text
    indent = min(len(line) - len(line.lstrip()) for line in non_empty)
    if indent == 0:
        return text
    return '\n'.join(line[indent:] if line.strip() else line for line in lines)


def _find_docstring_end(lines: list, start_idx: int) -> int:
    """Find the end line index of a docstring starting at start_idx. Returns -1 if not found."""
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if line.startswith('"""') or line.startswith("'''"):
            quote = line[:3]
            if line.count(quote) >= 2 and len(line) > 6:
                return i
            for j in range(i + 1, len(lines)):
                if quote in lines[j]:
                    return j
    return -1


def _extract_all_docstrings(code_lines: list) -> tuple:
    """Extract ALL docstrings from a code block (class-level + method-level).
    Returns (combined_docs_text, code_without_any_docstrings)."""
    docs = []
    code_lines_out = list(code_lines)

    i = 0
    while i < len(code_lines_out):
        stripped = code_lines_out[i].strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            quote = stripped[:3]
            if stripped.count(quote) >= 2 and len(stripped) > 6:
                doc_text = stripped[3:stripped.rfind(quote)]
                docs.append(doc_text)
                code_lines_out.pop(i)
                continue
            else:
                doc_lines = [stripped[3:]]
                end_j = i + 1
                for j in range(i + 1, len(code_lines_out)):
                    if quote in code_lines_out[j]:
                        end_pos = code_lines_out[j].find(quote)
                        doc_lines.append(code_lines_out[j][:end_pos])
                        end_j = j + 1
                        break
                    doc_lines.append(code_lines_out[j])
                docs.append('\n'.join(doc_lines))
                del code_lines_out[i:end_j]
                continue
        i += 1

    combined = '\n\n'.join(_dedent(d).strip() for d in docs if d.strip())
    return combined, '\n'.join(code_lines_out)


def parse_annotated_python(filepath: str) -> list:
    """Parse an annotated Python file into sections. Each section is (docs_text, code_text)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    sections = []
    module_doc = ''
    doc_match = re.match(r'^("""[\s\S]*?"""|\'\'\'[\s\S]*?\'\'\')', content)
    if doc_match:
        raw_doc = doc_match.group(1)
        module_doc = re.sub(r'^("""|\'\'\')', '', raw_doc)
        module_doc = re.sub(r'("""|\'\'\')$', '', module_doc)
        module_doc = re.sub(r'^---\n[\s\S]*?\n---\n?', '', module_doc)
        module_doc = module_doc.strip()
        content = content[doc_match.end():].lstrip()

    if module_doc:
        sections.append((module_doc, ''))

    lines = content.split('\n')

    # Find all definition starts (class/def at indent 0, and def at any indent)
    def_starts = []
    for i, line in enumerate(lines):
        if re.match(r'^(class |def )', line):
            def_starts.append(i)

    if not def_starts:
        if content.strip():
            sections.append(('', content))
        return sections

    # Add preamble before first definition
    if def_starts[0] > 0:
        preamble = '\n'.join(lines[:def_starts[0]]).strip()
        if preamble:
            sections.append(('', preamble))

    for idx, start in enumerate(def_starts):
        end = def_starts[idx + 1] if idx + 1 < len(def_starts) else len(lines)
        block_lines = lines[start:end]
        sig_line = block_lines[0]
        body_lines = block_lines[1:]

        # Extract ALL docstrings from this block (class docstring + method docstrings)
        doc_text, code_without_docs = _extract_all_docstrings(body_lines)

        if code_without_docs.strip():
            code_text = sig_line + '\n' + code_without_docs
        else:
            code_text = sig_line
        sections.append((doc_text, code_text))

    return sections


def generate_html_page(title, sections, parent_path="..", parent_title="强化学习", subtitle=""):
    nav_html = f'<div class="nav-bar">\n  <a class="parent" href="{REPO_URL}">{parent_title}</a>\n  <h1>{title}</h1>'
    if subtitle:
        nav_html += f'<p>{subtitle}</p>'
    nav_html += '</div>'

    sections_html = ""
    for idx, (docs_text, code_text) in enumerate(sections):
        if not docs_text and not code_text.strip():
            continue
        docs_html = render_markdown_to_html(docs_text) if docs_text else ""
        code_html = ""
        if code_text.strip():
            highlighted = highlight_python(code_text)
            highlighted_lines = highlighted.split('\n')
            numbered_lines = []
            for hline in highlighted_lines:
                line_num = len(numbered_lines) + 1
                numbered_lines.append(f'<span class="lineno">{line_num}</span>{hline}')
            code_html = '<pre>' + '\n'.join(numbered_lines) + '</pre>'

        sections_html += f"""<div class="section">
  <div class="docs">
    <div class="section-link"><a href="#section-{idx}">#</a></div>
    {docs_html}
  </div>
  <div class="code">
    {code_html}
  </div>
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title} - Annotated Easy-RL</title>
  <link rel="stylesheet" href="{KATEX_CSS}">
  <style>
{CSS}
{PYGMENTS_CSS}
  </style>
</head>
<body>
  <div id="container">
    <div id="background"></div>
    {nav_html}
    {sections_html}
    <div class="footer">
      <a href="{REPO_URL}">← 返回强化学习算法列表</a>
      <p style="color:#666; font-size:1.2rem; margin-top:10px;">
        基于<a href="https://github.com/datawhalechina/easy-rl/">蘑菇书EasyRL</a> |
        风格参考<a href="https://github.com/labmlai/annotated_deep_learning_paper_implementations">labml.ai</a>
      </p>
    </div>
  </div>
  <script src="{KATEX_JS}"></script>
  <script src="{AUTORENDER_JS}"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      renderMathInElement(document.body, {{
        delimiters: [
          {{left: '$$', right: '$$', display: true}},
          {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      }});
    }});
  </script>
</body>
</html>"""


def generate_index_page(html_dir):
    pages = [
        ("表格型方法 (Tabular)", "第三章", f"{REPO_URL}/html/tabular/", "Q-Learning, Sarsa, Value Iteration"),
        ("策略梯度 (Policy Gradient)", "第四章", f"{REPO_URL}/html/pg/", "REINFORCE, 基线技巧, 折扣回报"),
        ("PPO (近端策略优化)", "第五章", f"{REPO_URL}/html/ppo/", "PPO-Clip, PPO-Penalty"),
        ("GAE (广义优势估计)", "第五章", f"{REPO_URL}/html/ppo/gae.html", "广义优势估计"),
        ("DQN (深度Q网络)", "第六章", f"{REPO_URL}/html/dqn/", "DQN, Double DQN, Dueling DQN"),
        ("A2C (优势演员-评论员)", "第九章", f"{REPO_URL}/html/a2c/", "Advantage Actor-Critic"),
        ("DDPG (深度确定性策略梯度)", "第十二章", f"{REPO_URL}/html/ddpg/", "连续动作空间控制"),
    ]

    links_html = ""
    for name, chapter, path, desc in pages:
        links_html += f"""<div class="section">
  <div class="docs">
    <div class="section-link"><a href="#">#</a></div>
    <h2><a href="{path}">{name}</a></h2>
    <p style="color:#888; font-size:1.2rem;">{chapter}</p>
    <p>{desc}</p>
  </div>
  <div class="code">
    <pre style="color:#6272a4; font-size:1.3rem; padding:20px;">
<a href="{path}" style="color:#8be9fd;">点击查看详情 →</a>
    </pre>
  </div>
</div>
"""

    html_content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>强化学习算法 - Annotated Easy-RL</title>
  <link rel="stylesheet" href="{KATEX_CSS}">
  <style>
{CSS}
  </style>
</head>
<body>
  <div id="container">
    <div id="background"></div>
    <div class="nav-bar">
      <h1>🍄 蘑菇书EasyRL — 理论公式与代码注释实现</h1>
      <p>基于<a href="https://github.com/datawhalechina/easy-rl/">蘑菇书EasyRL</a>，
         采用<a href="https://github.com/labmlai/annotated_deep_learning_paper_implementations">labml.ai</a>风格的侧边对照注释排版</p>
    </div>
    {links_html}
    <div class="footer">
      <a href="{REPO_URL}">← 返回GitHub仓库</a>
      <p style="color:#666; font-size:1.2rem;">
        基于<a href="https://github.com/datawhalechina/easy-rl/">蘑菇书EasyRL</a> |
        风格参考<a href="https://github.com/labmlai/annotated_deep_learning_paper_implementations">labml.ai</a>
      </p>
    </div>
  </div>
  <script src="{KATEX_JS}"></script>
  <script src="{AUTORENDER_JS}"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function() {{
      renderMathInElement(document.body, {{
        delimiters: [
          {{left: '$$', right: '$$', display: true}},
          {{left: '$', right: '$', display: false}}
        ],
        throwOnError: false,
        ignoredTags: ["script", "noscript", "style", "textarea", "pre", "code"]
      }});
    }});
  </script>
</body>
</html>"""

    with open(os.path.join(html_dir, "index.html"), 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f"Generated: {os.path.join(html_dir, 'index.html')}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    annotated_dir = os.path.join(base_dir, "rl")
    html_dir = os.path.join(base_dir, "html")

    pages = [
        {"py_file": os.path.join(annotated_dir, "ppo", "__init__.py"),
         "html_file": os.path.join(html_dir, "ppo", "index.html"),
         "title": "近端策略优化 (PPO)", "subtitle": "Proximal Policy Optimization — 蘑菇书第五章"},
        {"py_file": os.path.join(annotated_dir, "ppo", "gae.py"),
         "html_file": os.path.join(html_dir, "ppo", "gae.html"),
         "title": "广义优势估计 (GAE)", "subtitle": "Generalized Advantage Estimation — PPO辅助算法"},
        {"py_file": os.path.join(annotated_dir, "dqn", "__init__.py"),
         "html_file": os.path.join(html_dir, "dqn", "index.html"),
         "title": "深度Q网络 (DQN)", "subtitle": "Deep Q Networks — 蘑菇书第六章"},
        {"py_file": os.path.join(annotated_dir, "pg", "__init__.py"),
         "html_file": os.path.join(html_dir, "pg", "index.html"),
         "title": "策略梯度 (Policy Gradient)", "subtitle": "REINFORCE算法 — 蘑菇书第四章"},
        {"py_file": os.path.join(annotated_dir, "a2c", "__init__.py"),
         "html_file": os.path.join(html_dir, "a2c", "index.html"),
         "title": "优势演员-评论员 (A2C)", "subtitle": "Advantage Actor-Critic — 蘑菇书第九章"},
        {"py_file": os.path.join(annotated_dir, "ddpg", "__init__.py"),
         "html_file": os.path.join(html_dir, "ddpg", "index.html"),
         "title": "深度确定性策略梯度 (DDPG)", "subtitle": "Deep Deterministic Policy Gradient — 蘑菇书第十二章"},
        {"py_file": os.path.join(annotated_dir, "tabular", "__init__.py"),
         "html_file": os.path.join(html_dir, "tabular", "index.html"),
         "title": "表格型方法 (Tabular Methods)", "subtitle": "Q-Learning & Sarsa — 蘑菇书第三章"},
    ]

    for page in pages:
        if not os.path.exists(page["py_file"]):
            print(f"Warning: {page['py_file']} not found, skipping")
            continue
        sections = parse_annotated_python(page["py_file"])
        html_content = generate_html_page(
            title=page["title"], sections=sections,
            parent_path="..", subtitle=page["subtitle"],
        )
        os.makedirs(os.path.dirname(page["html_file"]), exist_ok=True)
        with open(page["html_file"], 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Generated: {page['html_file']}")

    generate_index_page(html_dir)


if __name__ == "__main__":
    main()

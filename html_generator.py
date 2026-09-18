
import re
import os
import asyncio
from playwright.async_api import async_playwright
import mistune
from mistune import create_markdown
from mistune.plugins.math import math
from mistune.plugins.table import table

# Define source files
files = [
    "Phase0讲义.md",
    "Week1讲义.md",
    "Week2讲义.md",
    "Week3讲义.md",
    "Week4讲义.md",
    "Week5讲义.md",
    "Week6讲义.md",
    "Week7讲义.md",
    "Week8讲义.md",
    "Week9讲义.md",
    "Week10讲义.md",
    "Week11讲义.md",
    "Week12讲义.md",
    "Week13讲义.md",
    "Week14讲义.md",
]

# CSS for print and readability
css = """
<style>
    body { 
        font-family: "Helvetica Neue", Helvetica, Arial, sans-serif; 
        font-size: 16px; 
        line-height: 1.7; 
        color: #333; 
        margin: 0 auto; 
        padding: 40px; 
        max-width: 900px;
    }
    /* 内容容器：硬性约束宽度 */
    .content-wrapper {
        width: 100%;
        max-width: 100%;
        overflow: hidden;
    }
    @media print {
        @page {
            size: A4;
            margin: 20mm;
        }
        body {
            max-width: 100%;
            padding: 0;
            margin: 0;
            width: 100%;
        }
        .content-wrapper {
            width: 100%;
            max-width: 100%;
            overflow: hidden;
        }
        /* 标题与后续内容保持在一起 */
        h1, h2, h3, h4, h5, h6 {
            page-break-after: avoid;
            break-after: avoid;
        }
        /* 表格：强制适应页面宽度，防止溢出导致整页缩放 */
        table {
            page-break-inside: avoid;
            break-inside: avoid;
            width: 100% !important;
            max-width: 100% !important;
            table-layout: fixed !important;
            word-wrap: break-word;
            overflow-wrap: break-word;
            font-size: 12px;
        }
        th, td {
            white-space: normal !important;
            overflow: hidden !important;
            text-overflow: ellipsis;
            word-break: break-word;
        }
        /* 代码块允许分页，但自动换行 */
        pre {
            white-space: pre-wrap;
            word-wrap: break-word;
            overflow: hidden;
            max-width: 100%;
            /* 不设置 page-break-inside: avoid，允许长代码分页 */
        }
        /* 允许长内容分页，但控制孤行寡行 */
        p, li {
            orphans: 3;
            widows: 3;
        }
        /* 图片不分割 */
        img {
            page-break-inside: avoid;
            break-inside: avoid;
            max-height: 90vh;
            max-width: 100%;
        }
        /* Mermaid 图表缩放以适应页面 */
        .mermaid {
            max-height: 85vh !important;
            max-width: 100% !important;
            overflow: hidden !important;
        }
        .mermaid svg {
            max-height: 85vh !important;
            max-width: 100% !important;
            width: auto !important;
            height: auto !important;
        }
        /* Callout 框尽量不分割，但如果太长则允许 */
        .callout {
            page-break-inside: avoid;
            break-inside: avoid;
        }
        /* 减少标题上方的间距 */
        h1, h2 {
            margin-top: 16px;
        }
        h3, h4, h5, h6 {
            margin-top: 12px;
        }
        /* 数学公式防溢出 */
        .math-inline, .math-block, .math, mjx-container {
            overflow: hidden !important;
            max-width: 100% !important;
        }
    }
    /* Mermaid 图表默认样式 - 限制最大高度 */
    .mermaid {
        text-align: center;
        margin: 16px 0;
    }
    .mermaid svg {
        max-width: 100%;
        max-height: 600px;
        height: auto;
    }
    h1, h2, h3, h4, h5, h6 { 
        color: #111; 
        margin-top: 24px; 
        margin-bottom: 16px; 
        font-weight: 600; 
    }
    h1 { font-size: 2em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
    h2 { font-size: 1.5em; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; }
    h3 { font-size: 1.25em; }
    code { 
        font-family: "Menlo", "Monaco", "Courier New", monospace; 
        padding: 0.2em 0.4em; 
        background-color: #f6f8fa; 
        border-radius: 3px; 
        font-size: 85%;
    }
    pre { 
        background-color: #f6f8fa; 
        padding: 16px; 
        overflow: auto; 
        border-radius: 3px; 
        line-height: 1.45;
    }
    pre code { 
        padding: 0; 
        background-color: transparent; 
    }
    blockquote { 
        padding: 0 1em; 
        color: #6a737d; 
        border-left: 0.25em solid #dfe2e5; 
        margin: 0 0 16px 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin-bottom: 16px;
        table-layout: auto;
        font-size: 14px;
    }
    th, td {
        border: 1px solid #dfe2e5;
        padding: 8px 12px;
        text-align: left;
        vertical-align: top;
        word-wrap: break-word;
    }
    th {
        background-color: #f6f8fa;
        font-weight: 600;
        white-space: nowrap;
    }
    /* 第一列（通常是标签列）不换行 */
    td:first-child {
        white-space: nowrap;
        font-weight: 500;
    }
    /* 内容列允许换行 */
    td:not(:first-child) {
        min-width: 120px;
    }
    tr:nth-child(even) {
        background-color: #f9f9f9;
    }
    img { 
        max-width: 100%; 
    }
    /* MathJax styling */
    .math-inline, .math-block, .math {
        font-size: 1.1em;
    }
    /* GitHub-style Callouts */
    .callout {
        padding: 12px 16px;
        margin: 16px 0;
        border-left: 4px solid;
        border-radius: 4px;
    }
    .callout-title {
        font-weight: 600;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .callout-note {
        background-color: #e7f3ff;
        border-color: #0969da;
    }
    .callout-note .callout-title { color: #0969da; }
    .callout-tip {
        background-color: #e6ffec;
        border-color: #1a7f37;
    }
    .callout-tip .callout-title { color: #1a7f37; }
    .callout-warning {
        background-color: #fff8e6;
        border-color: #9a6700;
    }
    .callout-warning .callout-title { color: #9a6700; }
    .callout-important {
        background-color: #ffebe9;
        border-color: #cf222e;
    }
    .callout-important .callout-title { color: #cf222e; }
    .callout table {
        margin-top: 12px;
    }
</style>
<!-- MathJax Configuration -->
<script>
window.MathJax = {
  tex: {
    inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
    displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  options: {
    skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code', 'div.mermaid'],
    processHtmlClass: 'math'
  }
};
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>
<!-- Mermaid.js for diagrams -->
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>
  mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    securityLevel: 'loose',
    flowchart: { useMaxWidth: true, htmlLabels: true }
  });
</script>
"""

def convert_callouts(md_text):
    """
    Convert GitHub-style callouts like > [!NOTE] to a format we can process.
    """
    callout_types = {
        'NOTE': ('📝', 'note', 'Note'),
        'TIP': ('💡', 'tip', 'Tip'),
        'WARNING': ('⚠️', 'warning', 'Warning'),
        'IMPORTANT': ('❗', 'important', 'Important'),
        'CAUTION': ('🔥', 'warning', 'Caution'),
    }

    lines = md_text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]

        # Check if this is a callout start
        callout_match = re.match(r'^>\s*\[!(NOTE|TIP|WARNING|IMPORTANT|CAUTION)\](.*)$', line)

        if callout_match:
            callout_type = callout_match.group(1)
            extra_text = callout_match.group(2).strip()
            icon, css_class, title = callout_types.get(callout_type, ('📌', 'note', 'Note'))

            # Collect all lines in this blockquote
            callout_content = []
            if extra_text:
                callout_content.append(extra_text)

            i += 1
            while i < len(lines) and lines[i].startswith('>'):
                # Remove the > prefix
                content_line = re.sub(r'^>\s?', '', lines[i])
                callout_content.append(content_line)
                i += 1

            # Convert the callout content to HTML separately
            inner_md = '\n'.join(callout_content)

            # Store as a special block that won't be touched by markdown
            block_id = len(result)
            result.append(f'CALLOUT_BLOCK_{block_id}_TYPE_{css_class}_ICON_{icon}_TITLE_{title}_CONTENT_START')
            result.append(inner_md)
            result.append(f'CALLOUT_BLOCK_{block_id}_END')
        else:
            result.append(line)
            i += 1

    return '\n'.join(result)

def process_callouts_in_html(html):
    """
    Convert callout placeholders to proper HTML divs.
    Process the inner content as markdown.
    """
    # Create a markdown processor for inner content
    md = create_markdown(plugins=[math, table])

    # Pattern to match our callout placeholders
    pattern = r'(?:<p>)?CALLOUT_BLOCK_(\d+)_TYPE_(\w+)_ICON_(.+?)_TITLE_(\w+)_CONTENT_START(?:</p>)?(.*?)(?:<p>)?CALLOUT_BLOCK_\1_END(?:</p>)?'

    def replace_callout(match):
        css_class = match.group(2)
        icon = match.group(3)
        title = match.group(4)
        content = match.group(5).strip()

        # Clean up any paragraph wrappers in content
        content = re.sub(r'^<p>(.*)</p>$', r'\1', content, flags=re.DOTALL)

        return f'''<div class="callout callout-{css_class}">
<div class="callout-title">{icon} {title}</div>
<div class="callout-content">
{content}
</div>
</div>'''

    html = re.sub(pattern, replace_callout, html, flags=re.DOTALL)

    return html

def convert_mermaid_blocks(md_text):
    """
    Convert ```mermaid code blocks to placeholders.
    """
    pattern = r'```mermaid\s*\n(.*?)```'
    mermaid_blocks = []

    def replace_mermaid(match):
        mermaid_code = match.group(1).strip()
        block_id = len(mermaid_blocks)
        mermaid_blocks.append(mermaid_code)
        return f'MERMAID_PLACEHOLDER_{block_id}_END'

    result = re.sub(pattern, replace_mermaid, md_text, flags=re.DOTALL)
    return result, mermaid_blocks

def convert_math_blocks(md_text):
    """
    Protect LaTeX math blocks ($...$ and $$...$$) from markdown processing.
    """
    math_blocks = []

    # Pattern for display math $$...$$
    display_pattern = r'\$\$(.*?)\$\$'
    # Pattern for inline math $...$ (avoiding double $)
    inline_pattern = r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)'

    def replace_math(match, is_display):
        content = match.group(1)
        block_id = len(math_blocks)
        math_blocks.append((content, is_display))
        return f'MATH_PLACEHOLDER_{block_id}_END'

    # Process display math first
    md_text = re.sub(display_pattern, lambda m: replace_math(m, True), md_text, flags=re.DOTALL)
    # Then inline math
    md_text = re.sub(inline_pattern, lambda m: replace_math(m, False), md_text)

    return md_text, math_blocks

def restore_math_blocks(html, math_blocks):
    """
    Restore protected math blocks.
    """
    for i, (content, is_display) in enumerate(math_blocks):
        placeholder = f'MATH_PLACEHOLDER_{i}_END'
        if is_display:
            replacement = f'$$\n{content}\n$$'
        else:
            replacement = f'${content}$'

        # Mistune might wrap placeholder in <p>
        html = html.replace(f'<p>{placeholder}</p>', replacement)
        html = html.replace(placeholder, replacement)

    return html

def restore_mermaid_blocks(html, mermaid_blocks):
    """
    Restore mermaid placeholders to proper div tags.
    """
    for i, mermaid_code in enumerate(mermaid_blocks):
        placeholder = f'MERMAID_PLACEHOLDER_{i}_END'
        replacement = f'<div class="mermaid">\n{mermaid_code}\n</div>'
        # Handle cases where mistune wraps it in <p>
        html = html.replace(f'<p>{placeholder}</p>', replacement)
        html = html.replace(placeholder, replacement)
    return html

def convert_markdown_to_html(md_text):
    """
    Convert markdown to HTML with strict protection for math and mermaid.
    """
    # 1. Protect Math blocks (Crucial for underscores)
    md_text, math_blocks = convert_math_blocks(md_text)

    # 2. Convert mermaid blocks to placeholders
    md_text, mermaid_blocks = convert_mermaid_blocks(md_text)

    # 3. Convert GitHub-style callouts
    md_text = convert_callouts(md_text)

    # 4. Standard Markdown to HTML
    md = create_markdown(escape=False, plugins=[table])
    html = md(md_text)

    # 5. Restore Math (original content)
    html = restore_math_blocks(html, math_blocks)

    # 6. Restore mermaid
    html = restore_mermaid_blocks(html, mermaid_blocks)

    # 7. Process callouts in HTML
    html = process_callouts_in_html(html)

    # 8. Fix image paths
    base_dir = os.getcwd()

    def replace_img_src(match):
        src = match.group(1)
        # If it's a web URL, leave it alone
        if src.startswith('http://') or src.startswith('https://') or src.startswith('//'):
            return f'src="{src}"'

        # If it's an absolute path, leave it alone
        if src.startswith('/'):
            return f'src="{src}"'

        # If it's a relative path, resolve it against the current working directory
        # Handle ./ prefix if present
        if src.startswith('./'):
            src = src[2:]

        abs_path = os.path.join(base_dir, src)
        # Normalize path to remove any . or .. segments
        abs_path = os.path.normpath(abs_path)
        return f'src="file://{abs_path}"'

    html = re.sub(r'src="([^"]+)"', replace_img_src, html)

    return html

async def convert_to_pdf_async(html_path, pdf_path):
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        # 固定 viewport 宽度为 A4 纸可打印区域（210mm - 40mm margins = 170mm ≈ 794px @119dpi）
        # 确保所有文件在相同渲染条件下生成 PDF
        await page.set_viewport_size({"width": 794, "height": 1123})
        url = f"file://{os.path.abspath(html_path)}"
        await page.goto(url)
        # Wait for MathJax to finish - use a longer wait and check for completion
        await page.wait_for_timeout(8000)
        
        await page.pdf(path=pdf_path, format="A4", margin={
            "top": "20mm",
            "bottom": "20mm",
            "left": "20mm",
            "right": "20mm"
        }, print_background=True, scale=1, prefer_css_page_size=True)
        
        await browser.close()

def convert_file(filename):
    if not os.path.exists(filename):
        print(f"File not found: {filename}")
        return

    print(f"Processing {filename}...")
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            md_text = f.read()
        
        # Convert to HTML
        html_body = convert_markdown_to_html(md_text)
        
        # Wrap in full HTML document
        full_html = f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body><div class='content-wrapper'>{html_body}</div></body></html>"
        
        # Output paths
        output_dir = "Lecture_Notes_Export"
        os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(filename)[0]
        output_html = os.path.join(output_dir, f"{base_name}.html")
        output_pdf = os.path.join(output_dir, f"{base_name}.pdf")
        
        # Save HTML
        with open(output_html, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"Created HTML: {output_html}")
        
        # Generate PDF
        asyncio.run(convert_to_pdf_async(output_html, output_pdf))
        print(f"Created PDF: {output_pdf}")
        
    except Exception as e:
        import traceback
        print(f"Failed to convert {filename}: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    for f in files:
        convert_file(f)

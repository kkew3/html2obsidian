#!/usr/bin/env python3
"""
This quick and dirty script inlines all xsltml_2.1.2/*.xsl into
xsltml_2.1.2/mmltex_inlined.xsl.
"""

from pathlib import Path
import re


root = Path(__file__).parent.parent / 'html2obsidian/xsltml_2.1.2'

# ------------------------------------------------------------------
# 1) Read the main file
# ------------------------------------------------------------------
main = (root / 'mmltex.xsl').read_text(encoding='utf-8')

# ------------------------------------------------------------------
# 2) Replace every <xsl:include href="..."/> with the file content
# ------------------------------------------------------------------
for f in (
    'tokens.xsl',
    'glayout.xsl',
    'scripts.xsl',
    'tables.xsl',
    'entities.xsl',
    'cmarkup.xsl',
):
    snippet = (root / f).read_text(encoding='utf-8')
    snippet = re.sub(r'^\s*<\?xml[^>]*\?>\s*', '', snippet)
    # strip outer <xsl:stylesheet ...> ... </xsl:stylesheet>
    snippet = re.search(
        r'<xsl:stylesheet[^>]*>(.*?)</xsl:stylesheet>',
        snippet,
        flags=re.DOTALL,
    ).group(1)
    # Very small & safe: literal match of the include line
    include_line = f'<xsl:include href="{f}"/>'
    main = main.replace(include_line, snippet)

# ------------------------------------------------------------------
# 3) Save or use directly
# ------------------------------------------------------------------
(root / 'mmltex_inlined.xsl').write_text(main, encoding='utf-8')

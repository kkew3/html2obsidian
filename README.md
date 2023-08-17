# html2obsidian

## Introduction

This lib (a simple `__main__` interface attached) converts an HTML to [Obsidian][1]-style Markdown.

## Features

- Supported tags:
  `<hr>`,
  `<p>`,
  `<h1>`, `<h2>`, `<h3>`, `<h4>`, `<h5>`, `<h6>`, 
  `<ul>`, `<ol>`, `<li>`,
  `<a>`,
  `<blockquote>`,
  `<table>`, `<tr>`, `<th>`, `<td>`,
  `<img>`,
  `<b>`, `<strong>`, `<i>`, `<em>`, `<mark>`, `<del>`, `<s>`,
  `<sub>`, `<sup>`,
  `<pre>`,
  `<div>` (partial),
  `<code>`, `<samp>`, `<kbd>`,
  `<span>` (partial)
- Math style support: `$...$`, `\(...\)`, `$$...$$`, `\[...\]`
- Within-document link support
- Within-site hyperlink/image support

## Installation

This library requires `python>=3.6`.

To install dependencies (please refer to `requirements.txt` for detail),

```bash
pip install -r requirements.txt
```

or

```bash
conda install pytest lxml
```

## Usage

### Run as executable

To use the attached `__main__`, refer to

```bash
python3 convert_html.py --help
```

for help.

### Use the library

```python
from lxml import etree
import convert_html

html_file = ...
options = ...  # may be empty dict
url = ...  # may be None

with open(html_file, encoding='utf-8') as infile:
    html = infile.read()
parser = etree.HTMLParser(target=convert_html.KeepOnlySupportedTarget(strict=True))
elements = etree.HTML(html, parser)
# this is the string output containing the markdown
output = convert_html.StackMarkdownGenerator(options, elements, url)
```

Please refer to `convert_html.StackMarkdownGenerator.default_options` for help on available options.

### Run tests

To run the tests, simply

```bash
pytest
```


[1]: https://obsidian.md

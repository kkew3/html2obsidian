# html2obsidian

## Introduction

This utility converts an HTML to [Obsidian][1]-style Markdown.

## Features

- Supported tags:
  `<body>`, `<section>`, `<aside>`,
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
- Math style support: `$...$`, `\(...\)`, `$$...$$`, `\[...\]`, and MathML
- Within-document link support
- Within-site hyperlink/image support

## Installation

This utility requires `python>=3.9`.

To install:

```bash
git clone https://github.com/kkew3/html2obsidian.git && cd html2obsidian
pip install -e .
```

or use [`uv`](https://docs.astral.sh/uv/) (recommended):

```bash
uv tool install git+https://github.com/kkew3/html2obsidian.git
```

## Examples

Check [sample_html](./sample_html) for example input html and [sample_output](./sample_output) for corresponding output markdown.

## Usage

### Run as executable

Example usage:

```bash
curl -fsSL the-url | html2obsidian --url the-url - > output.md
```

For detailed help, refer to

```bash
html2obsidian --help
```

which is quoted below for reference:

```
usage: html2obsidian [-h] [--ul-bullet {-,+,*}] [--strong-symbol {*,_}]
                     [--em-symbol {*,_}] [--sub-start-symbol CHARS]
                     [--sub-end-symbol CHARS] [--sup-start-symbol CHARS]
                     [--sup-end-symbol CHARS] [--join] [--elevate-header-to N]
                     [--indent-list-with-tab]
                     [--write-base64-img-to WRITE_BASE64_IMG_TO] [--url URL]
                     html_file

Convert an HTML file to Obsidian-style markdown and write to stdout.

positional arguments:
  html_file             the html file to read; pass `-` to read from stdin

options:
  -h, --help            show this help message and exit
  --ul-bullet {-,+,*}
  --strong-symbol {*,_}
  --em-symbol {*,_}
  --sub-start-symbol CHARS
  --sub-end-symbol CHARS
  --sup-start-symbol CHARS
  --sup-end-symbol CHARS
  --join
  --elevate-header-to N
  --indent-list-with-tab
  --write-base64-img-to WRITE_BASE64_IMG_TO
  --url URL             url if the html is downloaded from web; this helps
                        resolve within-doc link
```

Note that sometimes there are warnings issued, e.g.

```
/path/to/convert_html.py:1183: UserWarning: illegal linebreaks in <a>; ignored
  warnings.warn('illegal linebreaks in <a>; ignored')
```

Most of the time, however, it does not imply errors in conversion.

### Use as a library

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

Note that sometimes there are warnings issued, like mentioned above.
Please refer to `test_convert_html.py` (in particular, the comments), to see whether such warnings imply error or not.


## Bugs

- `$$..$$`-style math is not recognized when embedded in `<p>` rather than in `<div class="math">`.


[1]: https://obsidian.md

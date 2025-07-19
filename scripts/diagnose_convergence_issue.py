#!/usr/bin/env python3
"""
This utitlity is used to find the minimal reproducible example that causes
`convert_html.ProcessingNotConvergedError`.
"""

import sys
import random

from lxml import etree
from lxml import html as lhtml

from html2obsidian import convert_html
from html2obsidian.convert_html import (
    KeepOnlySupportedTarget,
    StackMarkdownGenerator,
    ProcessingNotConvergedError,
)


def remove_random_node(html: str) -> str:
    """
    Removes a single, randomly-selected node from the provided HTML string and
    returns the modified HTML.

    Parameters
    ----------
    html : str
        A valid HTML fragment or document.

    Returns
    -------
    truncated_html : str
        The HTML with one node removed.  If the input contains no elements,
        the original string is returned unchanged.
    """
    if not html.strip():
        return html

    # Parse the HTML.  `lxml.html.fromstring` creates an
    # <html><body>...</body></html> wrapper if the string looks like a full
    # document, otherwise it returns a <div>...</div> wrapper containing the
    # fragment.
    root = lhtml.fromstring(html)

    # Collect every element node (including the root) in depth-first order
    all_nodes = root.xpath(
        '//*'
    )  # root is included because root.xpath('//*') includes root itself

    if not all_nodes:  # Nothing to remove
        return html

    # Pick one at random
    victim = random.choice(all_nodes)

    # If the victim is the root itself, we cannot use .remove() on it.
    # Instead, we return an empty string.
    if victim is root:
        return ''

    parent = victim.getparent()
    if parent is None:  # Should not happen, but be safe
        return html

    parent.remove(victim)

    # Convert the tree back to a string.
    # `lxml.html.tostring` returns bytes in Py3; decode to str.
    return lhtml.tostring(root, encoding='unicode')


def run_html(options, url, html):
    parser = etree.HTMLParser(target=KeepOnlySupportedTarget(True))
    elements = etree.HTML(html, parser)
    _ = StackMarkdownGenerator(options, elements, url).generate()


def get_mrx_greedy(options, url, html):
    while True:
        prev_html = html
        reproducible = False
        for _ in range(100):
            html = remove_random_node(prev_html)
            try:
                run_html(options, url, html)
            except ProcessingNotConvergedError:
                reproducible = True
                break
        if reproducible:
            prev_html = html
        else:
            return prev_html


def main():
    args = convert_html.make_cli_parser(
        'diagnose_convergence_issue.py',
        description=(
            'Find a minimal reproducible document (MRD) that does not converge '
            'by greedily removing random nodes from the problematic html. The '
            'MRD will be written to stdout.'
        ),
    ).parse_args()
    keys = [
        'ul_bullet',
        'strong_symbol',
        'em_symbol',
        'sub_start_symbol',
        'sub_end_symbol',
        'sup_start_symbol',
        'sup_end_symbol',
        'join_lines_when_possible',
        'try_make_highest_header_hn',
        'indent_list_with_tab',
        'write_base64_img_to',
    ]
    options = {k: getattr(args, k) for k in keys}
    if args.html_file == '-':
        html = sys.stdin.read()
    else:
        with open(args.html_file, encoding='utf-8') as infile:
            html = infile.read()
    print(get_mrx_greedy(options, args.url, html), end='')


if __name__ == '__main__':
    main()

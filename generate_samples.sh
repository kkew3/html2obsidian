#!/usr/bin/env bash
python3 convert_html.py \
    --url "https://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/" \
    sample_html/sample1.html sample_output/sample1.md

python3 convert_html.py \
    --url "https://omoindrot.github.io/triplet-loss" \
    sample_html/sample2.html sample_output/sample2.md

python3 convert_html.py \
    --url "https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/" \
    sample_html/sample3.html sample_output/sample3.md

python3 convert_html.py \
    --elevate-header-to=2 \
    --url "https://qnscholar.github.io//2021-12/zotero-if/" \
    sample_html/sample4.html sample_output/sample4.md

python3 convert_html.py \
    --url "https://timvieira.github.io/blog/post/2014/02/12/visualizing-high-dimensional-functions-with-cross-sections/" \
    sample_html/sample5.html sample_output/sample5.md

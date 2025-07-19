#!/usr/bin/env bash
html2obsidian \
    --url "https://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/" \
    sample_html/sample1.html > sample_output/sample1.md

html2obsidian \
    --url "https://omoindrot.github.io/triplet-loss" \
    sample_html/sample2.html > sample_output/sample2.md

html2obsidian \
    --url "https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/" \
    sample_html/sample3.html > sample_output/sample3.md

html2obsidian \
    --elevate-header-to=2 \
    --url "https://qnscholar.github.io//2021-12/zotero-if/" \
    sample_html/sample4.html > sample_output/sample4.md

html2obsidian \
    --url "https://timvieira.github.io/blog/post/2014/02/12/visualizing-high-dimensional-functions-with-cross-sections/" \
    sample_html/sample5.html > sample_output/sample5.md

html2obsidian \
    --url "https://www.pinecone.io/learn/batch-layer-normalization/" \
    sample_html/sample6.html > sample_output/sample6.md

html2obsidian \
    --url "https://www.pinecone.io/learn/batch-layer-normalization/" \
    sample_html/sample6_clean.html > sample_output/sample6_clean.md

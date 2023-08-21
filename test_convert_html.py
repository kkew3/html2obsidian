from pathlib import Path

from lxml import etree

import convert_html


class TestKeepOnlySupportedTarget():
    def test_known_unknown_known(self):
        html = '<p>hello <foo>world <p>again</p></foo>!</p><bar>hola</bar>'
        parser = etree.HTMLParser(
            target=convert_html.KeepOnlySupportedTarget(True))
        elements = convert_html.as_text(
            etree.HTML(html, parser), 'pass', 'pass')
        assert elements == [
            convert_html.StartElement('p'),
            'hello',
            convert_html.Space(),
            convert_html.StartElement('p'),
            'again',
            convert_html.EndElement('p'),
            '!',
            convert_html.EndElement('p'),
        ]

    def test_unknown_known_unknown(self):
        html = '<foo>hello <p>world <bar>again</bar>!</p></foo><baz>hola</baz>'
        parser = etree.HTMLParser(
            target=convert_html.KeepOnlySupportedTarget(True))
        elements = convert_html.as_text(
            etree.HTML(html, parser), 'pass', 'pass')
        assert elements == [
            convert_html.StartElement('p'),
            'world',
            convert_html.Space(),
            '!',
            convert_html.EndElement('p'),
        ]


def merge_eq(e1, e2):
    return e1 if e1 == e2 else None


def test_stack_merge():
    assert convert_html.stack_merge(['a', 'b', 'c'],
                                    merge_eq) == ['a', 'b', 'c']
    assert convert_html.stack_merge(['a', 'b', 'b', 'c', 'c', 'c'],
                                    merge_eq) == ['a', 'b', 'c']
    assert convert_html.stack_merge(['a', 'a', 'a', 'c', 'b', 'b'],
                                    merge_eq) == ['a', 'c', 'b']


def test_stop_merging_on_seen():
    assert convert_html.stack_merge(['a', 'a', 'a', 'c', 'b', 'b'],
                                    convert_html.stop_merging_on_seen(
                                        'c',
                                        merge_eq)) == ['a', 'c', 'b', 'b']


def test_as_text():
    assert (convert_html.as_text(['abc ', ' ', '  ghi'],
                                 'ignore',
                                 eval_whitespace=True) == ['abc ghi'])
    assert convert_html.as_text(['abc ',
                                 convert_html.InlineCode('de  '), 'f'],
                                'ignore',
                                eval_whitespace=True,
                                eval_verb=True) == ['abc `de  `f']
    assert convert_html.as_text(['abc\n  def'],
                                'ignore',
                                merge_whitespace=False,
                                eval_whitespace=True) == ['abc\n  def']


def test_recognize_merge_whitespace():
    assert convert_html.recognize_merge_whitespace('a  b\n  c \n\n\n d') == [
        'a',
        convert_html.Space(), 'b',
        convert_html.Newline(), 'c',
        convert_html.LineBreak(), 'd'
    ]


def read_test_case(name: str):
    basedir = Path('test_cases')
    with open(basedir / (name + '.html'), encoding='utf-8') as infile:
        html = infile.read()
    with open(basedir / (name + '.md'), encoding='utf-8') as infile:
        md = infile.read()
    return html, md


def read_sample(name: str):
    basedir = Path('sample_html')
    with open(basedir / (name + '.html'), encoding='utf-8') as infile:
        html = infile.read()
    return html


class TestStackMarkdownGenerator:
    def _case(self, name: str, options: dict = None):
        html, md = read_test_case(name)
        elements = etree.HTML(
            html,
            etree.HTMLParser(
                target=convert_html.KeepOnlySupportedTarget(True)))
        if options is None:
            options = {}
        return convert_html.StackMarkdownGenerator(options, elements), md

    def test_escape(self):
        g, md = self._case('escape')
        assert g.generate() == md

    def test_simple_par(self):
        g, md = self._case('simple_par')
        assert g.generate() == md

    def test_multiline_par(self):
        g, md = self._case('multiline_par', {'join_lines_when_possible': True})
        assert g.generate() == md

    def test_nojoin_multiline_par(self):
        g, md = self._case('nojoin_multiline_par')
        # if this line fails, there may be bugs in pytest or python
        assert g.options['join_lines_when_possible'] is False
        assert g.generate() == md

    def test_mixed_par_raw(self):
        g, md = self._case('mixed_par_raw')
        assert g.generate() == md

    def test_simple_ul(self):
        g, md = self._case('simple_ul')
        assert g.generate() == md

    def test_multiline_li(self):
        g, md = self._case('multiline_li', {'join_lines_when_possible': True})
        assert g.generate() == md

    def test_nojoin_multiline_li(self):
        g, md = self._case('nojoin_multiline_li')
        # if this line fails, there may be bugs in pytest or python
        assert g.options['join_lines_when_possible'] is False
        assert g.generate() == md

    def test_ul(self):
        g, md = self._case('ul')
        assert g.generate() == md

    def test_ol(self):
        g, md = self._case('ol')
        assert g.generate() == md

    def test_ol_start(self):
        g, md = self._case('ol_start')
        assert g.generate() == md

    def test_mixed_ul_ol(self):
        g, md = self._case('mixed_ul_ol')
        assert g.generate() == md

    def test_header(self):
        g, md = self._case('header')
        assert g.generate() == md

    def test_header_newline(self):
        g, md = self._case('header_newline')
        assert g.generate() == md

    def test_header_to_elevate(self):
        g, md = self._case('header_to_elevate',
                           {'try_make_highest_header_hn': 1})
        assert g.generate() == md

    def test_strong_in_anchor(self):
        g, md = self._case('strong_in_anchor')
        assert g.generate() == md

    def test_sub_sup(self):
        g, md = self._case('sub_sup')
        assert g.generate() == md

    def test_sub_sup_as_ast(self):
        opts = {
            'sub_start_symbol': ' *',
            'sub_end_symbol': '* ',
            'sup_start_symbol': ' *',
            'sup_end_symbol': '* ',
        }
        g, md = self._case('sub_sup_as_ast', opts)
        assert g.generate() == md

    def test_blockquote(self):
        g, md = self._case('blockquote')
        assert g.generate() == md

    def test_mixed_ul_escape(self):
        g, md = self._case('mixed_ul_escape')
        assert g.generate() == md

    def test_ul_star_bullet(self):
        g, md = self._case('ul_star_bullet', {'ul_bullet': '*'})
        assert g.generate() == md

    def test_ul_with_a_newline(self):
        g, md = self._case('ul_with_a_newline')
        assert g.generate() == md

    def test_table(self):
        g, md = self._case('table')
        assert g.generate() == md

    def test_table_one_row(self):
        g, md = self._case('table_one_row')
        assert g.generate() == md

    def test_img(self):
        g, md = self._case('img')
        assert g.generate() == md

    def test_code(self):
        g, md = self._case('code')
        assert g.generate() == md

    def test_code_join(self):
        g, md = self._case('code_join', {'join_lines_when_possible': True})
        assert g.generate() == md

    def test_math(self):
        g, md = self._case('math')
        assert g.generate() == md

    def test_cycle1(self):
        g, md = self._case('cycle1')
        assert g.generate() == md

    def test_space_in_dollar_inline_math(self):
        g, md = self._case('space_in_dollar_inline_math')
        assert g.generate() == md

    def test_empty_div_between_a(self):
        g, md = self._case('empty_div_between_a')
        assert g.generate() == md

    def _sample(self, name: str, url: str = None):
        html = read_sample(name)
        elements = etree.HTML(
            html,
            etree.HTMLParser(
                target=convert_html.KeepOnlySupportedTarget(True)))
        convert_html.StackMarkdownGenerator({}, elements, url).generate()

    def test_sample1_validity(self):
        self._sample('sample1')
        self._sample(
            'sample1',
            'https://timvieira.github.io/blog/post/2017/08/18/backprop-is-not-just-the-chain-rule/'
        )

    def test_sample2_validity(self):
        self._sample('sample2')
        self._sample('sample2', 'https://omoindrot.github.io/triplet-loss')

    def test_sample3_validity(self):
        self._sample('sample3')
        self._sample(
            'sample3',
            'https://timvieira.github.io/blog/post/2017/04/21/how-to-test-gradient-implementations/'
        )

    def test_sample4_validity(self):
        self._sample('sample4')
        self._sample('sample4',
                     'https://qnscholar.github.io//2021-12/zotero-if/')

    def test_sample5_validity(self):
        self._sample('sample5')
        self._sample(
            'sample5',
            'https://timvieira.github.io/blog/post/2014/02/12/visualizing-high-dimensional-functions-with-cross-sections/'
        )

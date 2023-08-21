import argparse
import typing as ty
import hashlib
import re
import functools
from urllib.parse import urlparse
import warnings

T = ty.TypeVar('T')


class Pat:
    whitespace = re.compile(r'[\t ]+')
    newline = re.compile(r'\n')  # must be matched after linebreak
    linebreak = re.compile(r'\n{2,}')
    slashsquare_math_block = re.compile(r'(\\\[|\\])', flags=re.MULTILINE)
    slashparenthesis_inline_math = re.compile(
        r'(\\\(|\\\))', flags=re.MULTILINE)
    dollar_inline_math = re.compile(r'\$[^$\s][^$]*[^$\s]\$|\$[^$\s]\$')


def subset_dict(
    dict_: ty.Mapping[str, T],
    keys: ty.List[str],
) -> ty.Dict[str, T]:
    ss = {}
    for k in keys:
        if k in dict_:
            ss[k] = dict_[k]
    return ss


def bookmark_hash(text: str) -> str:
    return hashlib.sha1(text.encode('utf-8')).hexdigest()[:6]


class StartElement:
    __slots__ = ['tag', 'attrib']

    def __init__(
        self,
        tag: str,
        attrib: ty.Dict[str, str] = None,
    ):
        self.tag = tag
        self.attrib = attrib or {}

    def paired_with(self, e):
        if isinstance(e, EndElement):
            return self.tag == e.tag
        return False

    def __eq__(self, other):
        if isinstance(other, StartElement):
            return self.tag == other.tag and self.attrib == other.attrib
        if isinstance(other, str):
            return self.tag == other
        return NotImplemented

    def __hash__(self):
        return hash((self.tag, self.attrib))

    def __repr__(self):
        return 'StartElement(tag={!r}, attrib={!r})'.format(
            self.tag, self.attrib)


class EndElement:
    __slots__ = ['tag']

    def __init__(self, tag: str):
        self.tag = tag

    def paired_with(self, e):
        if isinstance(e, StartElement):
            return self.tag == e.tag
        return False

    def __eq__(self, other):
        if isinstance(other, EndElement):
            return self.tag == other.tag
        if isinstance(other, str):
            return self.tag == other
        return NotImplemented

    def __hash__(self):
        return hash(self.tag)

    def __repr__(self):
        return 'EndElement(tag={!r})'.format(self.tag)


SupportedElementType = ty.Union[StartElement, EndElement, str]


class KeepOnlySupportedTarget:
    """
    This target keeps only supported elements.
    """
    def __init__(self, strict: bool):
        self.strict = strict
        self.nodes: ty.List[SupportedElementType] = []
        self.stack: ty.List[str] = []
        self.active = False

    def start(self, tag, attrib):
        self.active = True
        tag = tag.lower()
        # the separator
        if tag == 'hr':
            self.nodes.append(StartElement(tag))
            self.stack.append(tag)
        # headers
        elif tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            self.nodes.append(StartElement(tag, subset_dict(attrib, ['id'])))
            self.stack.append(tag)
        # lists
        elif tag == 'li':
            self.nodes.append(StartElement(tag, subset_dict(attrib, ['id'])))
            self.stack.append(tag)
        elif tag == 'ul':
            self.nodes.append(StartElement(tag))
            self.stack.append(tag)
        elif tag == 'ol':
            self.nodes.append(
                StartElement(tag, subset_dict(attrib, ['start'])))
            self.stack.append(tag)
        # paragraphs
        elif tag == 'p':
            self.nodes.append(StartElement(tag))
            self.stack.append(tag)
        # anchors
        elif tag == 'a':
            self.nodes.append(
                StartElement(tag, subset_dict(attrib, ['id', 'href'])))
            self.stack.append(tag)
        # quotes
        elif tag == 'blockquote':
            self.nodes.append(StartElement(tag))
            self.stack.append(tag)
        # tables, <tbody> and <thead> are not supported currently
        elif tag in ['table', 'tr', 'th', 'td']:
            self.nodes.append(StartElement(tag))
            self.stack.append(tag)
        # figures
        elif tag == 'img':
            self.nodes.append(
                StartElement(tag, subset_dict(attrib, ['src', 'alt'])))
            self.stack.append(tag)
        # basic text layout
        elif tag in ['b', 'strong', 'i', 'em', 'mark', 'del', 's']:
            self.nodes.append(StartElement(tag))
            self.stack.append(tag)
        # sub and sup
        elif tag in ['sub', 'sup']:
            self.nodes.append(StartElement(tag, subset_dict(attrib, ['id'])))
            self.stack.append(tag)
        # code blocks
        elif tag == 'pre':
            self.nodes.append(StartElement(tag))
            self.stack.append(tag)
        elif tag == 'div' and 'class' in attrib:
            for v in attrib['class'].split():
                if v.startswith('language-'):
                    self.nodes.append(StartElement(tag, {'class': v}))
                    self.stack.append(tag)
                    break
                # also math
                if v == 'math':
                    self.nodes.append(StartElement(tag, {'class': 'math'}))
                    self.stack.append(tag)
                    break
        elif tag in ['code', 'samp', 'kbd']:
            self.nodes.append(StartElement(tag, attrib))
            self.stack.append(tag)
        # math (alternative)
        elif tag == 'span' and 'class' in attrib:
            if attrib['class'] == 'math':
                self.nodes.append(StartElement(tag, {'class': 'math'}))
                self.stack.append(tag)
        # misc
        elif tag == 'div' and 'id' in attrib:
            self.nodes.append(StartElement(tag, {'id': attrib['id']}))
            self.stack.append(tag)
        else:
            self.active = False

    def end(self, tag):
        if self.stack and self.stack[-1] == tag:
            self.nodes.append(EndElement(tag))
            del self.stack[-1]
        self.active = True

    def data(self, data):
        if self.active:
            data = data.replace('\r\n', '\n')
            self.nodes.append(data)

    def close(self):
        if self.stack:
            if self.strict:
                raise ValueError('stack not empty at EOF')
            warnings.warn('stack not empty at EOF')
            while self.stack:
                self.nodes.append(EndElement(self.stack.pop()))
        res = self.nodes.copy()
        self.nodes.clear()
        return res


class PhantomElement:
    """Represents a partially resolved element."""
    def __eq__(self, other):
        return type(self) is type(other)

    def __str__(self):
        return ''

    def __repr__(self):
        return type(self).__name__


class Anchor(PhantomElement):
    __slots__ = ['id_']

    def __eq__(self, other):
        return super().__eq__(other) and self.id_ == other.id_

    def __init__(self, id_: str):
        self.id_ = id_

    def __repr__(self):
        return '{}(id={!r})'.format(type(self).__name__, self.id_)


class MdList(PhantomElement):
    __slots__ = []


class MdListItemIndentPointAtBullet(PhantomElement):
    """Should be replaced by proper indentation"""
    __slots__ = []


class MdLIstItemIndentPointOtherwise(PhantomElement):
    """Should be replaced by proper indentation"""
    __slots__ = []


class MdListItemBullet(PhantomElement):
    """Should be replaced by bullet"""
    __slots__ = []


class MdTableCell(PhantomElement):
    __slots__ = []


class MdTableRow(PhantomElement):
    __slots__ = ['n_cells']

    def __init__(self, n_cells: int):
        self.n_cells = n_cells

    def __eq__(self, other):
        return super().__eq__(other) and self.n_cells == other.n_cells

    def __repr__(self):
        return '{}(n_cells={!r})'.format(type(self).__name__, self.n_cells)


class VerbText:
    """
    Group of text free from being escaped.
    """
    __slots__ = ['text']

    def __init__(self, text: str):
        self.text = text

    def __eq__(self, other):
        return type(self) is type(other) and self.text == other.text

    def __str__(self):
        return self.text

    def __repr__(self):
        return '{}({})'.format(type(self).__name__, repr(self.text))


class SyntacticMarker:
    Asterisk = VerbText('*')
    Underscore = VerbText('_')
    Tilde = VerbText('~')
    Eq = VerbText('=')


class InlineMath(VerbText):
    def __init__(self, text):
        super().__init__(text)
        self.text = self.text

    def __str__(self):
        return '${}$'.format(self.text)


class Indentation(VerbText):
    def __str__(self):
        return self.text


def search_dollar_inline_math(text: str) -> ty.List[ty.Union[str, InlineMath]]:
    res = []
    start = 0
    for m in Pat.dollar_inline_math.finditer(text):
        if start < m.start():
            res.append(text[start:m.start()])
        res.append(InlineMath(m.group()[1:-1]))
        start = m.end()
    if start < len(text):
        res.append(text[start:])
    return res


def search_slashparenthesis_inline_math(
        text: str) -> ty.List[ty.Union[str, InlineMath]]:
    tokens = Pat.slashparenthesis_inline_math.split(text)
    res = []
    i = 0
    while i < len(tokens):
        if (tokens[i] == '\\(' and i + 2 < len(tokens)
                and tokens[i + 2] == '\\)'):
            res.append(InlineMath(tokens[i + 1].strip()))
            i += 3
        else:
            if tokens[i]:
                res.append(tokens[i])
            i += 1
    return res


class MathBlock(VerbText):
    def __init__(self, text):
        super().__init__(text)
        self.text = self.text.strip()

    def __str__(self):
        return '$$\n{}\n$$'.format(self.text)


def search_slashsquare_math_block(
        text: str) -> ty.List[ty.Union[str, MathBlock]]:
    tokens = Pat.slashsquare_math_block.split(text)
    res = []
    i = 0
    while i < len(tokens):
        if (tokens[i] == '\\[' and i + 2 < len(tokens)
                and tokens[i + 2] == '\\]'):
            res.append(MathBlock(tokens[i + 1]))
            i += 3
        else:
            res.append(tokens[i])
            i += 1
    return res


class InlineCode(VerbText):
    def __str__(self):
        return '`{}`'.format(self.text)


class CodeBlock(VerbText):
    def __init__(self, text: str, language: str = None):
        super().__init__(text)
        self.language = language

    def __str__(self):
        sbuf = ['```']
        if self.language:
            sbuf.append(self.language)
        if not self.text.startswith('\n'):
            sbuf.append('\n')
        sbuf.append(self.text)
        if not self.text.endswith('\n'):
            sbuf.append('\n')
        sbuf.append('```')
        return ''.join(sbuf)


class HeaderHashes(VerbText):
    def __init__(self, text: str):
        super().__init__(text)
        if set(self.text) != {'#'}:
            raise ValueError('invalid characters in HeaderHashes: {}'.format(
                self.text))

    @property
    def n(self):
        return len(self.text)

    @n.setter
    def n(self, value):
        value = int(value)
        if value < 1 or value > 6:
            raise ValueError('invalid value: {}'.format(value))
        self.text = '#' * value


class Whitespace:
    __slots__ = []

    def __eq__(self, other):
        return type(self) is type(other)

    def __repr__(self):
        return type(self).__name__


class LineBreak(Whitespace):
    def __str__(self):
        return '\n\n'


class Newline(Whitespace):
    def __str__(self):
        return '\n'


class Space(Whitespace):
    def __str__(self):
        return ' '


class Tab(Whitespace):
    def __str__(self):
        return '\t'


def lstrip_whitespace(
    elements: ty.List[T],
    strip_type: ty.Union[ty.Type, ty.List[ty.Type]] = Whitespace,
) -> ty.List[T]:
    """
    Strip leading whitespace elements in place.

    :param elements:
    :param strip_type: whitespace type(s)
    :return: stripped elements
    """
    if not isinstance(strip_type, list):
        strip_type = [strip_type]
    for t in strip_type:
        if not issubclass(t, Whitespace):
            raise TypeError('strip_type ({}) is not Whitespace'.format(
                strip_type.__name__))
    stripped_elements = []
    i = index_beg(elements)
    while i is not None and isinstance(elements[i], tuple(strip_type)):
        stripped_elements.append(elements[i])
        del elements[i]
        i = index_beg(elements)
    return stripped_elements


def rstrip_whitespace(
    elements: ty.List[T],
    strip_type: ty.Union[ty.Type, ty.List[ty.Type]] = Whitespace,
) -> ty.List[T]:
    """
    Strip trailing whitespace elements in place.

    :param elements:
    :param strip_type: whitespace type(s)
    :return: stripped elements (not in reversed order)
    """
    if not isinstance(strip_type, list):
        strip_type = [strip_type]
    for t in strip_type:
        if not issubclass(t, Whitespace):
            raise TypeError('strip_type ({}) is not Whitespace'.format(
                strip_type.__name__))
    stripped_elements = []
    i = index_end(elements)
    while i is not None and isinstance(elements[i], tuple(strip_type)):
        stripped_elements.append(elements[i])
        del elements[i]
        i = index_end(elements)
    stripped_elements.reverse()
    return stripped_elements


def index_beg(elements) -> ty.Optional[int]:
    """
    Returns the first non-Phantom index of ``elements``.
    """
    for i, e in enumerate(elements):
        if not isinstance(e, PhantomElement):
            return i
    return None


def index_end(elements) -> ty.Optional[int]:
    """
    Returns the last non-Phantom index of ``elements``.
    """
    index_el = list(enumerate(elements))
    for i, e in reversed(index_el):
        if not isinstance(e, PhantomElement):
            return i
    return None


def stack_merge(
    elements: ty.List[T],
    rule: ty.Callable[[ty.Optional[T], T], ty.Union[None, T,
                                                    ty.List[ty.Optional[T]]]],
) -> ty.List[T]:
    """
    :param elements: elements to merge
    :param rule: a callable, if returning ``None``, current two elements cannot
           be merged; otherwise, current two elements are merged into the
           return value. If a list is returned, the first of current two
           elements is replaced by the list. If a list starting with ``None``
           is returned, the list is appended after the first of current two
           elements. The first arg of the callable may be ``None``, when
           the first of current two elements is not present yet.
    :return: merged elements
    """
    res = []
    prev_e = None
    for e in elements:
        new = rule(prev_e, e)
        if new is None:
            res.append(e)
        elif isinstance(new, list):
            if new[0] is not None:
                del res[-1]
            else:
                new = new[1:]
            res.extend(new)
        else:
            del res[-1]
            res.append(new)
        prev_e = res[-1]
    return res


def stop_merging_on_seen(
    stop_instance,
    rule: ty.Callable,
) -> ty.Callable:
    """The ``rule`` perform merging before seen the ``stop_instance``."""
    class _StoppableRule:
        def __init__(self):
            self.seen_stop_instance = False

        def __call__(self, e1, e2):
            if not self.seen_stop_instance:
                if e2 == stop_instance:
                    self.seen_stop_instance = True
                else:
                    return rule(e1, e2)
            return None

    return _StoppableRule()


def merge_whitespace_rule(e1, e2):
    """
    Returns a ``Whitespace`` if both ``e1`` and ``e2`` are of type
    ``Whitespace``, otherwise returns ``None``.
    """
    if isinstance(e1, Whitespace) and isinstance(e2, Whitespace):
        if isinstance(e1, LineBreak) and isinstance(e2, LineBreak):
            return LineBreak()
        if isinstance(e1, LineBreak) and isinstance(e2, Newline):
            return LineBreak()
        if isinstance(e1, LineBreak) and isinstance(e2, Space):
            return LineBreak()
        if isinstance(e1, LineBreak) and isinstance(e2, Tab):
            return LineBreak()
        if isinstance(e1, Newline) and isinstance(e2, LineBreak):
            return LineBreak()
        if isinstance(e1, Newline) and isinstance(e2, Newline):
            return LineBreak()
        if isinstance(e1, Newline) and isinstance(e2, Space):
            return Newline()
        if isinstance(e1, Newline) and isinstance(e2, Tab):
            return Newline()
        if isinstance(e1, Space) and isinstance(e2, LineBreak):
            return LineBreak()
        if isinstance(e1, Space) and isinstance(e2, Newline):
            return Newline()
        if isinstance(e1, Space) and isinstance(e2, Space):
            return Space()
        if isinstance(e1, Space) and isinstance(e2, Tab):
            return Space()
        if isinstance(e1, Tab) and isinstance(e2, LineBreak):
            return LineBreak()
        if isinstance(e1, Tab) and isinstance(e2, Newline):
            return Newline()
        if isinstance(e1, Tab) and isinstance(e2, Space):
            return Space()
        return Space()
    return None


def recognize_whitespace(text: str) -> ty.List[ty.Union[str, Whitespace]]:
    tr_dict = {
        ' ': Space(),
        '\n': Newline(),
        '\t': Tab(),
    }
    res = [tr_dict.get(ch, ch) for ch in text]

    def merge_str_rule(e1, e2):
        if isinstance(e1, str) and isinstance(e2, str):
            return e1 + e2
        return None

    res = stack_merge(res, merge_str_rule)
    return res


def recognize_merge_whitespace(
        text: str) -> ty.List[ty.Union[str, Whitespace]]:
    # recognize linebreaks
    res1 = []
    start = 0
    for m in Pat.linebreak.finditer(text):
        if start < m.start():
            res1.append(text[start:m.start()])
        res1.append(LineBreak())
        start = m.end()
    if start < len(text):
        res1.append(text[start:])
    # recognize remaining newlines
    res2 = []
    for text in res1:
        if isinstance(text, str):
            start = 0
            for m in Pat.newline.finditer(text):
                if start < m.start():
                    res2.append(text[start:m.start()])
                res2.append(Newline())
                start = m.end()
            if start < len(text):
                res2.append(text[start:])
        else:
            res2.append(text)
    # recognize remaining whitespaces
    res3 = []
    for text in res2:
        if isinstance(text, str):
            start = 0
            for m in Pat.whitespace.finditer(text):
                if start < m.start():
                    res3.append(text[start:m.start()])
                res3.append(Space())
                start = m.end()
            if start < len(text):
                res3.append(text[start:])
        else:
            res3.append(text)
    # merge whitespace objects
    res4 = stack_merge(res3, merge_whitespace_rule)
    return res4


class LocalHref:
    """
    Represents the text part of an <a> tag referencing a within-document
    location.
    """
    __slots__ = ['elements', 'href', 'ref']

    def __init__(self, elements: ty.List[ty.Union[str, VerbText, Whitespace]],
                 href: str):
        """
        :param elements: children elements
        :param the href without the leading '#'
        """
        self.elements = elements
        self.href = href
        self.ref = None

    def __repr__(self):
        return '{}(href={!r}, ref={!r}, elements={!r})'.format(
            type(self).__name__, self.href, self.ref, self.elements)

    def as_text(self):
        if self.ref is None:
            raise ValueError('self.ref is None')
        res = ['[[#', self.ref, '|']
        res.extend(self.elements)
        res.append(']]')
        return res


IntermediateElementType = ty.Union[SupportedElementType, PhantomElement,
                                   VerbText, Whitespace, LocalHref]


def resolve_local_hrefs(
    elements: ty.List[IntermediateElementType],
    ref_context: ty.Dict[str, ty.Dict[str, str]],
) -> None:
    """Resolve ``LocalHref``s inplace."""
    for e in elements:
        if isinstance(e, LocalHref):
            try:
                ctx = ref_context[e.href]
                if ctx['type'] == 'header':
                    e.ref = ctx['ref']
                else:
                    e.ref = '^' + ctx['ref']
            except KeyError:
                pass


def escape(text: str) -> str:
    """Escape backslashes, asterisks and underscores."""
    text = text.replace('\\', '\\\\')
    text = text.replace('*', '\\*')
    text = text.replace('_', '\\_')
    return text


def as_text(
    elements: ty.List[IntermediateElementType],
    phantom_policy: ty.Literal['pass', 'ignore', 'warn', 'raise'],
    tag_policy: ty.Literal['pass', 'raise'] = 'raise',
    merge_whitespace: bool = True,
    eval_local_href: ty.Literal['elements', 'text'] = False,
    eval_whitespace: bool = False,
    escape_text: bool = False,
    eval_verb: bool = False,
) -> ty.List[IntermediateElementType]:
    """
    :param elements: elements to collapse
    :param phantom_policy: policy when finding ``PhantomElement``
    :param tag_policy: policy when finding ``StartElement`` or ``EndElement``
    :param merge_whitespace: if ``True``, successive linebreaks are
           substituted by one single ``LineBreak``, and then remaining
           successive whitespace characters are substituded by one ``Space``
    :param eval_local_href: if 'text', evaluate ``LocalHref`` to text; if
           'elements', evaluate it to its children elements; if ``False``,
           don't evaluate it
    :param eval_whitespace: if ``True``, evaluate ``Whitespace`` after
           evaluating ``LocalHref``
    :param escape_text: if ``True``, escape non-``VerbText`` text after
           evaluating whitespace
    :param eval_verb: also evaluate ``VerbText`` to ``str`` after escaping
    :raise TypeError: if ``elements`` contain types other than ``str`` or
           ``VerbText``
    """
    # escape and type check
    res1 = []
    for e in elements:
        if isinstance(e, (str, VerbText, Whitespace, LocalHref)):
            res1.append(e)
        elif isinstance(e, PhantomElement):
            if phantom_policy == 'pass':
                res1.append(e)
            elif phantom_policy == 'ignore':
                pass
            elif phantom_policy == 'warn':
                warnings.warn('unexpected PhantomElement: {!r}'.format(e))
            else:
                raise TypeError('unexpected PhantomElement: {!r}'.format(e))
        elif isinstance(e, (StartElement, EndElement)):
            if tag_policy == 'raise':
                raise TypeError('invalid type: {}'.format(type(e)))
            res1.append(e)
        else:
            raise TypeError('invalid type: {}'.format(type(e)))
    # merge whitespace in each element
    res2 = []
    sbuf = []
    for e in res1:
        if isinstance(e, str):
            sbuf.append(e)
        else:
            s = ''.join(sbuf)
            if merge_whitespace:
                res2.extend(recognize_merge_whitespace(s))
            else:
                res2.extend(recognize_whitespace(s))
            sbuf.clear()
            res2.append(e)
    if sbuf:
        s = ''.join(sbuf)
        if merge_whitespace:
            res2.extend(recognize_merge_whitespace(s))
        else:
            res2.extend(recognize_whitespace(s))
    if merge_whitespace:
        # merge whitespace across elements
        res3 = stack_merge(res2, merge_whitespace_rule)
    else:
        res3 = res2.copy()
    if eval_local_href == 'text':
        res3_2 = res3.copy()
        res3.clear()
        for e in res3_2:
            if isinstance(e, LocalHref):
                res3.extend(e.as_text())
            else:
                res3.append(e)
        del res3_2
    elif eval_local_href == 'elements':
        res3_2 = res3.copy()
        res3.clear()
        for e in res3_2:
            if isinstance(e, LocalHref):
                res3.extend(e.elements)
            else:
                res3.append(e)
        del res3_2
    elif eval_local_href:
        raise ValueError(
            'invalid eval_local_href value: {}'.format(eval_local_href))
    if eval_whitespace:
        res3 = [str(e) if isinstance(e, Whitespace) else e for e in res3]
    if escape_text:
        res3 = [escape(e) if isinstance(e, str) else e for e in res3]
    if eval_verb:
        res3 = [str(e) if isinstance(e, VerbText) else e for e in res3]

    def merge_str_rule(e1, e2):
        if isinstance(e1, str) and isinstance(e2, str):
            return e1 + e2
        return None

    res4 = stack_merge(res3, merge_str_rule)
    return res4


def check_converged(elements: ty.List[IntermediateElementType]) -> bool:
    """
    Returns ``True`` when ``elements`` only consists of ``str``,
    ``VerbTect`` and ``Whitespace``.
    """
    for e in elements:
        if not (isinstance(e, (str, VerbText, Whitespace, PhantomElement)) or
                (isinstance(e, LocalHref) and e.ref is not None)):
            return False
    return True


def contains_unparsed_element(
        elements: ty.List[IntermediateElementType]) -> bool:
    return any(isinstance(e, (StartElement, EndElement)) for e in elements)


def collect_phantom(
    elements: ty.List[IntermediateElementType],
    phantom_type: ty.Type[T],
) -> ty.Tuple[ty.List[IntermediateElementType], ty.List[T]]:
    res = []
    phantoms = []
    if not phantom_type:
        phantom_type = PhantomElement
    for e in elements:
        if isinstance(e, phantom_type):
            phantoms.append(e)
        else:
            res.append(e)
    return res, phantoms


class ProcessingNotConvergedError(Exception):
    pass


class StackMarkdownGenerator:
    default_options = {
        # bullet for <li> of <ul>; valid values: '-', '*', '+'
        'ul_bullet': '-',
        # symbol for <strong> or <b>; valid values: '*', '_'
        'strong_symbol': '*',
        # symbol for <em> or <i>; valid values: '*', '_'
        'em_symbol': '_',
        # character(s) to insert before <sub>; may be empty string;
        # wherein whitespace characters are treated as ``Whitespace`` and
        # others as ``VerbText``
        'sub_start_symbol': ' ',
        # character(s) to insert after </sub>; may be empty string;
        # wherein whitespace characters are treated as ``Whitespace`` and
        # others as ``VerbText``
        'sub_end_symbol': ' ',
        # character(s) to insert before <sup>; may be empty string;
        # wherein whitespace characters are treated as ``Whitespace`` and
        # others as ``VerbText``
        'sup_start_symbol': ' ',
        # character(s) to insert after </sup>; may be empty string;
        # wherein whitespace characters are treated as ``Whitespace`` and
        # others as ``VerbText``
        'sup_end_symbol': ' ',
        # if `True`, try to join lines in <li> and <p>
        'join_lines_when_possible': False,
        # if not `None`, try to make the highest header level be that high;
        # e.g. if the value is 1, and the highest level in the html is <h2>,
        # then all <h2> will become <h1>, <h3> become <h2>, etc.
        'try_make_highest_header_hn': None,
        # if `True`, indent embedded lists using Tab rather than four spaces
        'indent_list_with_tab': False
    }

    def __init__(
        self,
        options: ty.Dict[str, ty.Any],
        elements: ty.List[SupportedElementType],
        page_url: str = None,
        max_loop: int = 100,
    ) -> None:
        self.options = self.default_options.copy()
        for k in self.options:
            if options.get(k, None) is not None:
                self.options[k] = options[k]

        self.stack: ty.List[IntermediateElementType] = elements
        # to store reference context
        self.ref_context: ty.Dict[str, ty.Dict[str, str]] = {}

        if page_url:
            self.page_url_info = urlparse(page_url)
        else:
            self.page_url_info = None
        self.max_loop = max_loop

    def generate(self) -> str:
        queue = []
        n_loop = 1
        while not check_converged(self.stack):
            if n_loop > self.max_loop:
                raise ProcessingNotConvergedError
            queue.clear()
            queue.extend(self.stack)
            self.stack.clear()
            for e in queue:
                if isinstance(e, EndElement):
                    elements = []
                    while (not isinstance(self.stack[-1], StartElement)
                           or not self.stack[-1].paired_with(e)):
                        elements.append(self.stack.pop())
                    elements.reverse()
                    start = self.stack.pop()
                    assert isinstance(start, StartElement), (e, start)
                    procf = getattr(self, 'proc_' + start.tag)
                    parents = []
                    for p in reversed(self.stack):
                        if isinstance(p, StartElement):
                            parents.append(p)
                        elif isinstance(p, EndElement):
                            break
                    res = procf(start.attrib, elements, parents)
                    if res is None:
                        self.stack.append(start)
                        self.stack.extend(elements)
                        self.stack.append(e)
                    else:
                        self.stack.extend(res)
                else:
                    self.stack.append(e)

            # There may still be some raw math blocks in certain str nodes
            res1 = []
            for e in self.stack:
                if isinstance(e, str):
                    res1.extend(search_slashsquare_math_block(e))
                else:
                    res1.append(e)

            self.stack = as_text(res1, 'pass', 'pass')
            resolve_local_hrefs(self.stack, self.ref_context)
            n_loop += 1

        # if requested, make header levels as high as possible
        if self.options['try_make_highest_header_hn']:
            target_level = self.options['try_make_highest_header_hn']
            title_levels = [
                e.n for e in self.stack if isinstance(e, HeaderHashes)
            ]
            if title_levels:
                min_level = min(title_levels)
                if min_level > target_level:

                    def decrease_header_level_rule(e1, e2):
                        if isinstance(e2, HeaderHashes):
                            e2.n -= min_level - target_level
                            return [e1, e2]
                        return None

                    self.stack = stack_merge(self.stack,
                                             decrease_header_level_rule)

        # (attempt to) ensure no duplicate newline at EOF
        rstrip_whitespace(self.stack, [LineBreak, Newline])
        i = index_end(self.stack)
        if i is None or not isinstance(self.stack[i], Newline):
            self.stack.append(Newline())
        # (attempt to) ensure no leading newline at BOF
        lstrip_whitespace(self.stack, [LineBreak, Newline])

        return ''.join(
            as_text(
                self.stack,
                phantom_policy='warn',
                eval_local_href='text',
                eval_whitespace=True,
                escape_text=True,
                eval_verb=True))

    def try_resolve_local_link(self, url: str):
        if self.page_url_info:
            url_info = urlparse(url)
            if not url_info.scheme and not url_info.netloc:
                if url.startswith('/'):
                    return '{}://{}{}'.format(self.page_url_info.scheme,
                                              self.page_url_info.netloc, url)
                return '{}://{}/{}'.format(self.page_url_info.scheme,
                                           self.page_url_info.netloc, url)
        return url

    def proc_hr(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        res = [
            '---',
            LineBreak(),
        ]
        res.extend(elements)
        return as_text(res, 'pass')

    def proc_p(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[ty.Union[str, VerbText, Whitespace]]]:
        """
        May contain::

            - Anchor
            - $...$ style inline math
        """
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        res, anchors = collect_phantom(res, Anchor)
        if anchors:
            h = bookmark_hash(''.join(
                as_text(
                    res,
                    phantom_policy='warn',
                    eval_local_href='elements',
                    eval_whitespace=True,
                    eval_verb=True)))
            for a in anchors:
                self.ref_context[a.id_] = {
                    'type': 'hash',
                    'ref': h,
                }
            res.append(' ^{}'.format(h))

        if self.options['join_lines_when_possible']:

            def sub_newline_with_space_rule(e1, e2):
                if e1 is not None and isinstance(e2, Newline):
                    return [e1, Space()]
                return None

            res = stack_merge(res, sub_newline_with_space_rule)

        res.append(LineBreak())
        res = as_text(res, phantom_policy='warn', eval_whitespace=True)
        res2 = []
        for e in res:
            if isinstance(e, str):
                res2.extend(search_dollar_inline_math(e))
            else:
                res2.append(e)
        res3 = as_text(res2, phantom_policy='ignore')
        return res3

    def _proc_headers(
        self,
        n: int,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        All children must already be parsed.

        May contain::

            - Anchor
        """
        if contains_unparsed_element(elements):
            raise ValueError('<h{}> contains unparsed element'.format(n))
        res = as_text(elements, 'pass')
        if any(isinstance(e, LineBreak) for e in res):
            warnings.warn('illegal linebreaks in <h{}>; ignored'.format(n))
            res = [e for e in res if not isinstance(e, LineBreak)]

        def sub_newline_with_space_rule(e1, e2):
            if isinstance(e2, Newline):
                return [e1, Space()]
            return None

        res = stack_merge(res, sub_newline_with_space_rule)
        lstrip_whitespace(res, Space)
        rstrip_whitespace(res, Space)

        res.insert(0, HeaderHashes('#' * n))
        res.insert(1, Space())

        res, anchors = collect_phantom(res, Anchor)
        res.append(LineBreak())
        if 'id' in attrib or anchors:
            ref = ''.join(
                as_text(
                    res[2:-1],
                    phantom_policy='ignore',
                    eval_local_href='elements',
                    eval_whitespace=True,
                    eval_verb=True))
            for a in anchors:
                self.ref_context[a.id_] = {
                    'type': 'header',
                    'ref': ref,
                }
            if 'id' in attrib:
                self.ref_context[attrib['id']] = {
                    'type': 'header',
                    'ref': ref,
                }
        return as_text(res, 'pass')

    proc_h1 = functools.partialmethod(_proc_headers, 1)
    proc_h2 = functools.partialmethod(_proc_headers, 2)
    proc_h3 = functools.partialmethod(_proc_headers, 3)
    proc_h4 = functools.partialmethod(_proc_headers, 4)
    proc_h5 = functools.partialmethod(_proc_headers, 5)
    proc_h6 = functools.partialmethod(_proc_headers, 6)

    def proc_a(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        All children must already be parsed.
        There should be no children ``PhantomElement`` too.
        """
        if contains_unparsed_element(elements):
            raise ValueError('<a> contains unparsed element')
        elements = as_text(elements, 'pass')

        elements = as_text(elements, 'warn')
        if any(isinstance(e, LineBreak) for e in elements):
            warnings.warn('illegal linebreaks in <a>; ignored')
            elements = [e for e in elements if not isinstance(e, LineBreak)]

        def sub_newline_with_space_rule(e1, e2):
            if isinstance(e2, Newline):
                return [e1, Space()]
            return None

        elements = stack_merge(elements, sub_newline_with_space_rule)
        front_spaces = lstrip_whitespace(elements, Space)
        back_spaces = rstrip_whitespace(elements, Space)

        res = []
        res.extend(front_spaces)
        if 'id' in attrib:
            res.append(Anchor(attrib['id']))
        if 'href' in attrib and attrib['href'].startswith('#'):
            res.append(LocalHref(elements, attrib['href'][1:]))
        elif 'href' in attrib:
            link = self.try_resolve_local_link(attrib['href'])
            res.append('[')
            res.extend(elements)
            res.extend(['](', VerbText(link), ')'])
        else:
            res.extend(elements)
        res.extend(back_spaces)
        return as_text(res, 'pass')

    def proc_li(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        May contain::

            - Anchor
            - MdList
            - MdListItemIndentPointAtBullet
            - MdListItemIndentPointOtherwise
        """
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        res, anchors = collect_phantom(res, Anchor)
        if anchors or 'id' in attrib:
            h = bookmark_hash(''.join(
                as_text(
                    res,
                    'ignore',
                    eval_local_href='elements',
                    eval_whitespace=True,
                    eval_verb=True)))
            for a in anchors:
                self.ref_context[a.id_] = {
                    'type': 'hash',
                    'ref': h,
                }
            if 'id' in attrib:
                self.ref_context[attrib['id']] = {
                    'type': 'hash',
                    'ref': h,
                }
        else:
            h = None

        lstrip_whitespace(res)
        rstrip_whitespace(res)

        stop_on_mdlist = functools.partial(stop_merging_on_seen, MdList())

        if self.options['join_lines_when_possible']:

            @stop_on_mdlist
            def sub_newline_with_space_rule(e1, e2):
                if e1 is not None and isinstance(e2, Newline):
                    return [e1, Space()]
                return None

            res = stack_merge(res, sub_newline_with_space_rule)

        @stop_on_mdlist
        def insert_sub_indent_rule(e1, e2):
            if (isinstance(e1, (LineBreak, Newline))
                    and isinstance(e2, (str, VerbText))):
                return [e1, Indentation('  '), e2]
            return None

        res = stack_merge(res, insert_sub_indent_rule)
        res.insert(0, MdListItemBullet())

        @stop_on_mdlist
        def insert_indent_point_rule(e1, e2):
            if e1 is None:
                return [None, MdListItemIndentPointAtBullet(), e2]
            if (isinstance(e1, (LineBreak, Newline))
                    and isinstance(e2, Indentation)):
                return [e1, MdLIstItemIndentPointOtherwise(), e2]
            return None

        res = stack_merge(res, insert_indent_point_rule)

        if any(isinstance(e, MdList) for e in res):

            def sub_whitespace_before_mdlist_rule(e1, e2):
                if isinstance(e1, Whitespace) and isinstance(e2, MdList):
                    if h is not None:
                        return [' ^{}'.format(h), Newline(), e2]
                    return [Newline(), e2]
                return None

            res = stack_merge(res, sub_whitespace_before_mdlist_rule)
        else:
            if h is not None:
                res.append(' ^{}'.format(h))

        return as_text(res, 'pass')

    def proc_ul(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        May contains::

            - MdListItemIndentPoint
            - MdListItemBullet
            - MdList
        """
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        bullet_char = self.options['ul_bullet']
        if bullet_char not in ['-', '*', '+']:
            raise ValueError('invalid <ul> bullet: {}'.format(bullet_char))

        def sub_ul_bullet_rule(e1, e2):
            if isinstance(e2, MdListItemBullet):
                return [e1, VerbText(bullet_char), Space()]
            return None

        res = stack_merge(res, sub_ul_bullet_rule)

        stop_on_mdlist = functools.partial(stop_merging_on_seen, MdList())

        @stop_on_mdlist
        def one_newline_between_li_rule(e1, e2):
            if (isinstance(e1, LineBreak)
                    and isinstance(e2, MdListItemIndentPointAtBullet)):
                return [Newline(), e2]
            # if there's no newline at all, add one
            if (isinstance(e1, (str, VerbText))
                    and isinstance(e2, MdListItemIndentPointAtBullet)):
                return [e1, Newline(), e2]
            return None

        res = stack_merge(res, one_newline_between_li_rule)

        # remove MdList from embedded <ul>/<ol>
        res = [e for e in res if not isinstance(e, MdList)]

        if 'li' in parents:

            def indent_one_level_rule(e1, e2):
                if isinstance(e2, (MdListItemIndentPointAtBullet,
                                   MdLIstItemIndentPointOtherwise)):
                    if self.options['indent_list_with_tab']:
                        return [e1, Indentation('\t'), e2]
                    return [e1, Indentation('    '), e2]
                return None

            res = stack_merge(res, indent_one_level_rule)
            res.insert(0, MdList())
            lstrip_whitespace(res)
        else:
            res = [
                e for e in res
                if not isinstance(e, (MdListItemIndentPointAtBullet,
                                      MdLIstItemIndentPointOtherwise))
            ]
            res.append(LineBreak())

        return as_text(res, 'pass')

    def proc_ol(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        May contains::

            - MdListItemIndentPoint
            - MdListItemBullet
            - MdList
        """
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        class sub_bullet_rule:
            def __init__(self, start: int):
                self.ind = start - 1

            def __call__(self, e1, e2):
                if isinstance(e2, MdListItemBullet):
                    self.ind += 1
                    return [e1, '{}. '.format(self.ind)]
                return None

        res = stack_merge(res, sub_bullet_rule(int(attrib.get('start', 1))))

        stop_on_mdlist = functools.partial(stop_merging_on_seen, MdList())

        @stop_on_mdlist
        def one_newline_between_li_rule(e1, e2):
            if (isinstance(e1, LineBreak)
                    and isinstance(e2, MdListItemIndentPointAtBullet)):
                return [Newline(), e2]
            # if there's no newline at all, add one
            if (isinstance(e1, (str, VerbText))
                    and isinstance(e2, MdListItemIndentPointAtBullet)):
                return [e1, Newline(), e2]
            return None

        res = stack_merge(res, one_newline_between_li_rule)

        # remove MdList from embedded <ul>/<ol>
        res = [e for e in res if not isinstance(e, MdList)]

        if 'li' in parents:

            def indent_one_level_rule(e1, e2):
                if isinstance(e2, (MdListItemIndentPointAtBullet,
                                   MdLIstItemIndentPointOtherwise)):
                    if self.options['indent_list_with_tab']:
                        return [e1, Indentation('\t'), e2]
                    return [e1, Indentation('    '), e2]
                return None

            res = stack_merge(res, indent_one_level_rule)
            res.insert(0, MdList())
            lstrip_whitespace(res)
        else:
            res = [
                e for e in res
                if not isinstance(e, (MdListItemIndentPointAtBullet,
                                      MdLIstItemIndentPointOtherwise))
            ]
            res.append(LineBreak())

        return as_text(res, 'pass')

    def proc_blockquote(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        May contain::

            - Anchor
        """
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        res, anchors = collect_phantom(res, Anchor)
        if anchors:
            h = bookmark_hash(''.join(
                as_text(
                    res,
                    'ignore',
                    eval_local_href='elements',
                    eval_whitespace=True,
                    eval_verb=True)))
            for a in anchors:
                self.ref_context[a.id_] = {
                    'type': 'hash',
                    'ref': h,
                }
        else:
            h = None

        lstrip_whitespace(res)
        rstrip_whitespace(res)

        def prepend_quote_symbol_at_newline_rule(e1, e2):
            if e1 is None or isinstance(e1, Newline):
                return [e1, '> ', e2]
            if isinstance(e1, LineBreak):
                return [Newline(), '>', Newline(), '> ', e2]
            return None

        res = stack_merge(res, prepend_quote_symbol_at_newline_rule)

        if h:
            res.append(' ^{}'.format(h))

        res.append(LineBreak())
        return as_text(res, 'pass')

    def proc_strong(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None

        sym = {
            '*': SyntacticMarker.Asterisk,
            '_': SyntacticMarker.Underscore,
        }[self.options['strong_symbol']]
        res = [sym, sym]
        res.extend(elements)
        res.extend([sym, sym])
        return res

    proc_b = proc_strong

    def proc_em(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None

        sym = {
            '*': SyntacticMarker.Asterisk,
            '_': SyntacticMarker.Underscore,
        }[self.options['strong_symbol']]
        res = [sym]
        res.extend(elements)
        res.append(sym)
        return res

    proc_i = proc_em

    def proc_mark(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None
        res = [SyntacticMarker.Eq, SyntacticMarker.Eq]
        res.extend(elements)
        res.extend([SyntacticMarker.Eq, SyntacticMarker.Eq])
        return res

    def proc_del(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None
        res = [SyntacticMarker.Tilde, SyntacticMarker.Tilde]
        res.extend(elements)
        res.extend([SyntacticMarker.Tilde, SyntacticMarker.Tilde])
        return res

    proc_s = proc_del

    def proc_sub(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None

        def handle_sub_symbol(
                symbol: str) -> ty.List[ty.Union[VerbText, Whitespace]]:
            texts = recognize_whitespace(symbol)
            texts = filter(None, texts)
            return [VerbText(e) if isinstance(e, str) else e for e in texts]

        sub_start = self.options['sub_start_symbol']
        sub_end = self.options['sub_end_symbol']
        res = []
        if 'id' in attrib:
            res.append(Anchor(attrib['id']))
        if sub_start:
            res.extend(handle_sub_symbol(sub_start))
        res.extend(elements)
        if sub_end:
            res.extend(handle_sub_symbol(sub_end))
        return as_text(res, 'pass')

    def proc_sup(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None

        def handle_sup_symbol(
                symbol: str) -> ty.List[ty.Union[VerbText, Whitespace]]:
            texts = recognize_whitespace(symbol)
            texts = filter(None, texts)
            return [VerbText(e) if isinstance(e, str) else e for e in texts]

        sup_start = self.options['sup_start_symbol']
        sup_end = self.options['sup_end_symbol']
        res = []
        if 'id' in attrib:
            res.append(Anchor(attrib['id']))
        if sup_start:
            res.extend(handle_sup_symbol(sup_start))
        res.extend(elements)
        if sup_end:
            res.extend(handle_sup_symbol(sup_end))
        return as_text(res, 'pass')

    def proc_td(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        lstrip_whitespace(res)
        rstrip_whitespace(res)

        res.insert(0, MdTableCell())
        res.insert(1, ' ')
        res.append(' |')
        return as_text(res, 'pass')

    proc_th = proc_td

    def proc_tr(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        May contains::

            - MdTableCell
        """
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        def rm_whitespace_before_cell_rule(e1, e2):
            if isinstance(e1, Whitespace) and isinstance(e2, MdTableCell):
                return e2
            return None

        res = stack_merge(res, rm_whitespace_before_cell_rule)
        rstrip_whitespace(res)

        res, cells = collect_phantom(res, MdTableCell)
        if cells:
            res.insert(0, MdTableRow(len(cells)))
            res.insert(1, '|')
        return as_text(res, 'pass')

    def proc_table(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        May contains::

            - MdTableRow
        """
        if contains_unparsed_element(elements):
            return None
        res = as_text(elements, 'pass')

        n_cells = max([e.n_cells for e in res if isinstance(e, MdTableRow)],
                      default=0)
        if not n_cells:
            return res

        lstrip_whitespace(res)
        rstrip_whitespace(res)

        class insert_headline_and_one_newline_between_row_rule:
            def __init__(self):
                self.row_ind = 0

            def __call__(self, e1, e2):
                if isinstance(e2, MdTableRow):
                    self.row_ind += 1
                    if isinstance(e1, (LineBreak, Newline)):
                        if self.row_ind == 2:
                            headerline = ('|'
                                          + '|'.join('---'
                                                     for _ in range(n_cells))
                                          + '|')
                            return [Newline(), headerline, Newline()]
                        if self.row_ind > 2:
                            return Newline()
                    if self.row_ind == 2:
                        headerline = ('|'
                                      + '|'.join('---' for _ in range(n_cells))
                                      + '|')
                        return [e1, Newline(), headerline, Newline()]
                    if self.row_ind > 2:
                        return [e1, Newline()]
                return None

        res2 = stack_merge(res,
                           insert_headline_and_one_newline_between_row_rule())
        if res2 != res:
            res = res2
        else:
            res.append(Newline())
            headerline = ('|' + '|'.join('---' for _ in range(n_cells)) + '|')
            res.append(headerline)

        res = [e for e in res if not isinstance(e, MdTableRow)]
        res.append(LineBreak())
        return as_text(res, 'pass')

    def proc_img(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        if contains_unparsed_element(elements):
            return None

        elements = as_text(elements, 'pass')
        res = ['![', attrib.get('alt', '')]
        res.extend(elements)
        link = self.try_resolve_local_link(attrib.get('src', ''))
        res.extend(['](', VerbText(link), ')'])
        return as_text(res, 'pass')

    def proc_code(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        All children must already be parsed.
        No ``PhantomElement`` children allowed.
        """
        if contains_unparsed_element(elements):
            raise ValueError('<code> contains unparsed element')

        if 'pre' in parents:
            text = ''.join(
                as_text(
                    elements,
                    'warn',
                    merge_whitespace=False,
                    eval_whitespace=True,
                    eval_verb=True))
            return [CodeBlock(text), LineBreak()]
        res = as_text(
            elements,
            'warn',
            merge_whitespace=self.options['join_lines_when_possible'])
        if self.options['join_lines_when_possible']:

            def sub_newline_with_space_rule(e1, e2):
                if e1 is not None and isinstance(e2, Newline):
                    return [e1, Space()]
                return None

            res = stack_merge(res, sub_newline_with_space_rule)
        text = ''.join(
            as_text(res, 'ignore', eval_whitespace=True, eval_verb=True))
        return [InlineCode(text)]

    proc_samp = proc_code
    proc_kbd = proc_code

    def proc_pre(
        self,
        _attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        All children must already be parsed.
        No ``PhantomElement`` children allowed.
        """
        if contains_unparsed_element(elements):
            raise ValueError('<pre> contains unparsed element')

        res = elements.copy()

        if any(isinstance(e, CodeBlock) for e in res):
            res = as_text(res, 'pass', merge_whitespace=False)
            if any(not isinstance(e, (CodeBlock, Whitespace)) for e in res):
                warnings.warn('<pre> contains non code block element; ignored')
            res = [e for e in res if isinstance(e, CodeBlock)]
            if len(res) > 1:
                warnings.warn('<pre> contains more than one code block; '
                              'using the first one only')
            return [res[0], LineBreak()]

        text = ''.join(
            as_text(
                elements,
                'warn',
                merge_whitespace=False,
                eval_whitespace=True,
                eval_verb=True))
        return [CodeBlock(text), LineBreak()]

    def proc_div(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        All children must already be parsed if having attribute 'class'.
        """
        if 'class' in attrib:
            if contains_unparsed_element(elements):
                raise ValueError('<pre> contains unparsed element')
            res = elements.copy()

            if attrib['class'].startswith('language-'):
                res = as_text(res, 'pass', merge_whitespace=False)
                language = attrib['class'][9:]

                if any(isinstance(e, CodeBlock) for e in res):
                    if any(not isinstance(e, (CodeBlock, Whitespace))
                           for e in res):
                        warnings.warn(
                            '<pre> contains non code block element; ignored')
                    res = [e for e in res if isinstance(e, CodeBlock)]
                    if len(res) > 1:
                        warnings.warn(
                            '<pre> contains more than one code block; '
                            'using the first one only')
                    res = [res[0]]
                    res[0].language = language
                    res.append(LineBreak())
                    return res

                warnings.warn(
                    ('<div class=""> contains unexpected non-code elements; '
                     'passed as is').format(attrib['class']))
                return as_text(res, 'pass')

            if attrib['class'] == 'math':
                text = ''.join(
                    as_text(res, 'warn', eval_whitespace=True, eval_verb=True))
                text = text.strip()
                if text.startswith('$$') and text.endswith('$$'):
                    return [MathBlock(text[2:-2]), LineBreak()]
                return [MathBlock(text), LineBreak()]

                # res = search_slashsquare_math_block(text)
                # if any(isinstance(e, MathBlock) for e in res):
                #     if any(not isinstance(e, MathBlock) for e in res):
                #         warnings.warn(
                #             '<div class="math"> contains unexpected non-math '
                #             'elements; ignored')
                #     res = [e for e in res if isinstance(e, MathBlock)]
                #
                #     def insert_linebreak_between_mathblock_rule(e1, e2):
                #         if isinstance(e1, MathBlock) and isinstance(
                #                 e2, MathBlock):
                #             return [e1, LineBreak(), e2]
                #         return None
                #
                #     res = stack_merge(res,
                #                       insert_linebreak_between_mathblock_rule)
                #
                #     res.append(LineBreak())
                #     return res
                #
                # warnings.warn(
                #     '<div class="math"> contains unexpected non-math '
                #     'elements; passed as is')
                # return as_text(res, 'pass')

            raise NotImplementedError(
                'not implemented for <div> class "{}"'.format(attrib['class']))

        if 'id' in attrib:
            if contains_unparsed_element(elements):
                return None
            res = as_text(elements, 'pass')

            h = bookmark_hash(''.join(
                as_text(
                    res,
                    'ignore',
                    eval_local_href='elements',
                    eval_whitespace=True,
                    eval_verb=True)))
            self.ref_context[attrib['id']] = {
                'type': 'hash',
                'ref': h,
            }
            res.append(' ^{}'.format(h))
            res.append(LineBreak())
            return as_text(res, 'pass')

        raise NotImplementedError('not implemented for <div>')

    def proc_span(
        self,
        attrib: ty.Dict[str, str],
        elements: ty.List[IntermediateElementType],
        _parents: ty.List[StartElement],
    ) -> ty.Optional[ty.List[IntermediateElementType]]:
        """
        All children must already be parsed.
        """
        if contains_unparsed_element(elements):
            raise ValueError('<pre> contains unparsed element')

        res = elements.copy()

        if attrib['class'] == 'math':
            text = ''.join(
                as_text(res, 'warn', eval_whitespace=True, eval_verb=True))

            res = search_slashparenthesis_inline_math(text)
            if any(isinstance(e, InlineMath) for e in res):
                if any(not isinstance(e, InlineMath) for e in res):
                    warnings.warn(
                        '<div class="math"> contains unexpected non-math '
                        'elements; ignored')
                res = [e for e in res if isinstance(e, InlineMath)]
                if len(res) > 1:
                    warnings.warn(
                        '<div class="math"> contains more than one inline '
                        'math; using the first one only')
                return [res[0]]

            warnings.warn('<span class="math"> contains unexpected non-math '
                          'elements; passed as is')
            return as_text(res, 'pass')

        raise NotImplementedError(
            'not implemented for <span> class "{}"'.format(attrib['class']))


def _make_parser():
    from pathlib import Path

    args = argparse.ArgumentParser(
        description='Convert an HTML file to Obsidian markdown.')
    args.add_argument('--ul-bullet', dest='ul_bullet', choices=['-', '+', '*'])
    args.add_argument(
        '--strong-symbol', dest='strong_symbol', choices=['*', '_'])
    args.add_argument('--em-symbol', dest='em_symbol', choices=['*', '_'])
    args.add_argument(
        '--sub-start-symbol', dest='sub_start_symbol', metavar='CHARS')
    args.add_argument(
        '--sub-end-symbol', dest='sub_end_symbol', metavar='CHARS')
    args.add_argument(
        '--sup-start-symbol', dest='sup_start_symbol', metavar='CHARS')
    args.add_argument(
        '--sup-end-symbol', dest='sup_end_symbol', metavar='CHARS')
    args.add_argument(
        '--join', dest='join_lines_when_possible', action='store_true')
    args.add_argument(
        '--elevate-header-to',
        dest='try_make_highest_header_hn',
        type=int,
        metavar='N')
    args.add_argument(
        '--indent-list-with-tab',
        dest='indent_list_with_tab',
        action='store_true')
    args.add_argument('--url', help='url if the html is downloaded from web')
    args.add_argument('html_file', type=Path, help='the html file to read')
    args.add_argument(
        'output_file', type=Path, help='the markdown file to write')
    return args


def _main():
    from lxml import etree

    args = _make_parser().parse_args()
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
    ]
    options = {k: getattr(args, k) for k in keys}
    with open(args.html_file, encoding='utf-8') as infile:
        html = infile.read()
    parser = etree.HTMLParser(target=KeepOnlySupportedTarget(True))
    elements = etree.HTML(html, parser)
    output = StackMarkdownGenerator(options, elements, args.url).generate()
    with open(args.output_file, 'w', encoding='utf-8') as outfile:
        outfile.write(output)


if __name__ == '__main__':
    _main()

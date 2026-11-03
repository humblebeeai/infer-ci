#!/usr/bin/env python3
"""Check translated README syntax, local targets and shared executable examples.

Install requirements/requirements.docs.txt first. External HTTP availability is
not checked. Known invalid document host www.docs is reported instead of silently
counting such links as valid external references. Run again after manual edits.
"""
from html.parser import HTMLParser
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

import markdown

ROOT = Path(__file__).resolve().parents[1]


class Document(HTMLParser):
    def __init__(self):
        super().__init__()
        self.targets = []
        self.headings = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'h1':
            self.headings += 1
        if tag in ('a', 'img'):
            target = attrs.get('href' if tag == 'a' else 'src')
            if target:
                self.targets.append(target)


def main():
    errors = []
    sources = {}
    for name in ('README.md', 'README.ko.md'):
        source = (ROOT / name).read_text(encoding='utf-8')
        sources[name] = source
        document = Document()
        document.feed(markdown.markdown(source, extensions=['fenced_code', 'tables']))
        if not document.headings:
            errors.append(f'{name}: no rendered H1; check outer Markdown fences')
        for number, line in enumerate(source.splitlines(), 1):
            if 'git clone [' in line:
                errors.append(f'{name}:{number}: clone command contains Markdown link syntax')
            if 'https://www.docs/' in line:
                errors.append(f'{name}:{number}: replace invalid document URL with repository path')
        for target in document.targets:
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            path = (ROOT / unquote(url.path)).resolve()
            if not path.exists():
                errors.append(f'{name}: missing local target {target}')
    blocks = lambda source: re.findall(r'^```python\n([\s\S]*?)^```', source, re.M)
    english, korean = (blocks(sources[name]) for name in ('README.md', 'README.ko.md'))
    if not english or english != korean:
        errors.append('README Python examples differ across languages or are missing')
    print(f'Shared Python examples: {len(english)}')
    for error in errors:
        print(f'ERROR: {error}')
    print(f'Result: {len(errors)} unresolved item(s). External HTTP links were not fetched.')
    return 1 if errors else 0


if __name__ == '__main__':
    sys.exit(main())

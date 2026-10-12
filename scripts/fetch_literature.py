"""Download adopted references explicitly; retain failures in sources.json.

Usage: python scripts/fetch_literature.py [--ids SOURCE_ID ...]
Only a real PDF is archived. No network access occurs at import time.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'doc/literature/sources.json'


def check_pdf(path):
    with path.open('rb') as handle:
        if handle.read(5) != b'%PDF-':
            raise ValueError('response is not a PDF')
    check = subprocess.run(['pdfinfo', str(path)], capture_output=True, text=True)
    if check.returncode:
        raise ValueError('PDF parser rejected the archive')
    pages = next((line.split(':', 1)[1].strip() for line in check.stdout.splitlines() if line.startswith('Pages:')), None)
    return int(pages) if pages else None


def download(source):
    target = ROOT / 'doc/literature/pdf' / source['filename']
    try:
        if target.exists():
            pages = check_pdf(target)
            return source['id'], {'pdf_status': 'downloaded', 'pdf_path': f"pdf/{target.name}", 'bytes': target.stat().st_size, 'pages': pages}
        request = Request(source['pdf_url'], headers={'User-Agent': 'Mozilla/5.0 (research reference archive)'})
        with urlopen(request, timeout=40) as response:
            content = response.read(80 * 1024 * 1024 + 1)
        if len(content) > 80 * 1024 * 1024 or not content.startswith(b'%PDF-'):
            raise ValueError('response is not a PDF within the archive size limit')
        with tempfile.TemporaryDirectory(prefix='wyckoff-reference-') as scratch:
            temporary = Path(scratch) / 'reference.pdf'
            temporary.write_bytes(content)
            pages = check_pdf(temporary)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return source['id'], {'pdf_status': 'downloaded', 'pdf_path': f"pdf/{target.name}", 'bytes': len(content), 'pages': pages}
    except Exception as exc:
        return source['id'], {'pdf_status': 'unavailable', 'pdf_error': f'{type(exc).__name__}: {exc}'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', nargs='+')
    args = parser.parse_args(argv)
    manifest = json.loads(MANIFEST.read_text())
    selected = [s for s in manifest['sources'] if not args.ids or s['id'] in args.ids]
    if args.ids and set(args.ids) - {s['id'] for s in selected}:
        parser.error('unknown source id')
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(download, selected))
    for identity, result in results:
        source = next(s for s in manifest['sources'] if s['id'] == identity)
        source.update(result)
        if result['pdf_status'] == 'downloaded':
            source.pop('pdf_error', None)
        print(identity, result['pdf_status'], result.get('pages', ''), result.get('pdf_error', ''), flush=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return 0 if all(result['pdf_status'] == 'downloaded' for _, result in results) else 1


if __name__ == '__main__':
    raise SystemExit(main())

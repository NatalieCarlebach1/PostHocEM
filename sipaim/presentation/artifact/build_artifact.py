"""Embed the paper's Fig. 1 panels into the talk page as data URIs."""
import base64, re
from pathlib import Path
HERE = Path(__file__).parent
PANELS = HERE.parent.parent / 'figures' / 'panels'
html = (HERE / 'talk_template.html').read_text()
def data_uri(key):
    return 'data:image/png;base64,' + base64.b64encode((PANELS / f'{key}.png').read_bytes()).decode()
keys = sorted(set(re.findall(r'__IMG_([a-z0-9_]+)__', html)))
for k in keys:
    html = html.replace(f'__IMG_{k}__', data_uri(k))
assert '__IMG_' not in html
out = HERE / 'pem_talk.html'
out.write_text(html)
print(f'{len(keys)} images embedded -> {out.name} ({out.stat().st_size/1e6:.2f} MB)')

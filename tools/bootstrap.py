"""Download project-local build tools and NVIDIA's runtime compiler (no installer)."""
import json
import pathlib
import urllib.request
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEST = ROOT / 'tools' / 'local'
DEST.mkdir(parents=True, exist_ok=True)

def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)

def download(url):
    target = DEST / url.rsplit('/', 1)[-1]
    if not target.exists():
        print('Downloading', target.name, flush=True)
        urllib.request.urlretrieve(url, target)
    return target

if not list(DEST.glob('llvm-mingw*/bin/clang++.exe')):
    release = get_json('https://api.github.com/repos/mstorsjo/llvm-mingw/releases/latest')
    asset = next(a for a in release['assets'] if 'ucrt-x86_64' in a['name'] and a['name'].endswith('.zip'))
    with zipfile.ZipFile(download(asset['browser_download_url'])) as archive:
        archive.extractall(DEST)

for package in ('cmake', 'ninja', 'nvidia-cuda-nvrtc-cu12'):
    metadata = get_json('https://pypi.org/pypi/' + package + '/json')
    asset = next(a for a in metadata['urls'] if a['filename'].endswith('win_amd64.whl'))
    with zipfile.ZipFile(download(asset['url'])) as archive:
        archive.extractall(DEST)
    print(package, metadata['info']['version'], flush=True)
print('Project-local tools ready:', DEST, flush=True)

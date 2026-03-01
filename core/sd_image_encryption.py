import asyncio
import hashlib
import base64
import sys
import io
import os
import gradio as gr
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Response
from PIL.PngImagePlugin import PngInfo
from urllib.parse import unquote
from packaging import version
from pathlib import Path
from PIL import Image as PILImage, PngImagePlugin, _util, ImagePalette

from modules.paths_internal import models_path
from modules import shared, images
from modules.api import api


# ~~ Constants ~~

ENCRYPT_PREFIX = 'SOBA:'
ENCRYPT_MARKER = 'pixel_shuffle_3'
TAG_LIST = ['parameters', 'UserComment']
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp', '.avif'}
IMAGE_KEYS = ['Encrypt', 'EncryptPwdSha']
HEADERS = {'Cache-Control': 'public, max-age=2592000'}
MISMATCH_ERROR = "axes don't match array"

SKIP_PATTERNS = {'card-no-preview.'}

password = getattr(shared.cmd_opts, 'encrypt_pass', None)
embed_dir = Path(shared.cmd_opts.embeddings_dir)
models_dir = Path(models_path)


# ~~ Logger ~~

class Logger:
    """Coloured console logger for image encryption events"""
    _PREFIX = '\033[34m[ImageEncryption]\033[0m -'
    _COLORS = {
        'info':    '\033[34m',
        'success': '\033[32m',
        'warning': '\033[33m',
        'error':   '\033[31m',
    }
    _RESET = '\033[0m'

    def _log(self, level: str, msg: str):
        color = self._COLORS[level]
        tag = f" [{level.upper()}] -" if level in ('warning', 'error') else ''
        print(f"{self._PREFIX}{tag} {color}{msg}{self._RESET}")

    def info(self, msg: str):    self._log('info', msg)
    def success(self, msg: str): self._log('success', msg)
    def warning(self, msg: str): self._log('warning', msg)
    def error(self, msg: str):   self._log('error', msg)

log = Logger()


# ~~ Helpers ~~

def get_sha256(text: str) -> str:
    """Return SHA-256 hex digest of a string"""
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def _get_range(s: str, offset: int, length: int = 4) -> str:
    """Return a circular substring of *s* starting at *offset*"""
    offset %= len(s)
    return (s * 2)[offset:offset + length]

def _shuffle(arr: np.ndarray, key: str) -> np.ndarray:
    """Deterministic Fisher-Yates shuffle driven by SHA-256(key)"""
    sha = get_sha256(key)
    n = len(arr)
    for i in range(n):
        j = int(_get_range(sha, i, 8), 16) % (n - i)
        k = n - i - 1
        arr[k], arr[j] = arr[j], arr[k]
    return arr

def _xor_str(text: str, pwd: str) -> str:
    """XOR each character of text against the repeating password"""
    return ''.join(chr(ord(c) ^ ord(pwd[i % len(pwd)])) for i, c in enumerate(text))


# ~~ Metadata encryption / decryption ~~

def encrypt_tags(metadata: dict, pwd: str) -> dict:
    """XOR-encrypt all TAG_LIST values, base64-encode, prepend prefix"""
    out = metadata.copy()
    for key in TAG_LIST:
        val = str(out.get(key) or '')
        if val and not val.startswith(ENCRYPT_PREFIX):
            out[key] = ENCRYPT_PREFIX + base64.b64encode(_xor_str(val, pwd).encode()).decode()
    return out

def decrypt_tags(metadata: dict, pwd: str) -> dict:
    """Reverse of encrypt_tags; values without the prefix pass through"""
    out = metadata.copy()
    for key in TAG_LIST:
        val = str(out.get(key, ''))
        if val.startswith(ENCRYPT_PREFIX):
            try:
                out[key] = _xor_str(base64.b64decode(val[len(ENCRYPT_PREFIX):]).decode(), pwd)
            except Exception:
                pass
    return out


# ~~ Pixel-shuffle image encryption / decryption ~~

def _permute_image(image: PILImage.Image, pwd: str, inverse: bool = False) -> np.ndarray:
    """Permute rows then columns of image pixels; set inverse=True to reverse"""
    try:
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        w, h = image.size
        px = np.array(image, dtype=np.uint8)
        
        y_perm = _shuffle(np.arange(h), get_sha256(pwd))
        x_perm = _shuffle(np.arange(w), pwd)
        if inverse:
            y_perm, x_perm = np.argsort(y_perm), np.argsort(x_perm)
            
        return px[y_perm].transpose(1, 0, 2)[x_perm].transpose(1, 0, 2)
    except Exception as e:
        if MISMATCH_ERROR not in str(e):
            log.error(f"_permute_image: {e}")
        return np.array(image.convert('RGBA'), dtype=np.uint8)

def encrypt_image(image: PILImage.Image, pwd: str) -> np.ndarray:
    """Pixel-shuffle encrypt an image (rows then columns)"""
    return _permute_image(image, pwd)

def decrypt_image(image: PILImage.Image, pwd: str) -> np.ndarray:
    """Exact inverse of encrypt_image"""
    return _permute_image(image, pwd, inverse=True)


# ~~ EncryptedImage ~~

class EncryptedImage(PILImage.Image):
    """PIL Image subclass that transparently encrypts on save()"""
    __name__ = 'EncryptedImage'

    @staticmethod
    def from_image(src: PILImage.Image) -> 'EncryptedImage':
        """Construct an EncryptedImage by copying all attributes from an existing PIL Image"""
        src = src.copy()
        img = EncryptedImage()
        img.im = src.im
        img._mode = src.mode
        try:
            if src.im.mode:
                img.mode = src.im.mode
        except Exception:
            pass
        img._size = src.size
        img.format = src.format
        if src.mode in ('P', 'PA'):
            img.palette = src.palette.copy() if src.palette else ImagePalette.ImagePalette()
        img.info = src.info.copy()
        return img

    def save(self, fp, format=None, **params):
        """Save the image, encrypting pixels and metadata on the fly"""
        filename = ''
        if isinstance(fp, Path):
            filename = str(fp)
        elif _util.is_path(fp):
            filename = fp
        elif fp == sys.stdout:
            fp = getattr(sys.stdout, 'buffer', fp)
        if not filename and hasattr(fp, 'name') and _util.is_path(fp.name):
            filename = fp.name

        if not filename or not password or self.info.get('Encrypt') == ENCRYPT_MARKER:
            super().save(fp, format=format, **params)
            return

        file_path = Path(filename)
        if any(d in file_path.parents for d in (models_dir, embed_dir)):
            if (self.format or '').upper() != 'PNG':
                png = PILImage.new('RGBA', self.size)
                png.paste(self)
                self = png
                self.format = 'PNG'

        encrypted_info = encrypt_tags(self.info, password)
        pnginfo = params.get('pnginfo', PngImagePlugin.PngInfo()) or PngImagePlugin.PngInfo()

        backup = PILImage.new('RGBA', self.size)
        backup.paste(self)
        try:
            self.paste(PILImage.fromarray(encrypt_image(self, get_sha256(password))))
        except Exception as e:
            if MISMATCH_ERROR in str(e):
                os.system(f"rm -f {filename}")
                return
            raise

        for key, value in encrypted_info.items():
            if value:
                pnginfo.add_text(key, str(value))
        pnginfo.add_text('Encrypt', ENCRYPT_MARKER)
        pnginfo.add_text('EncryptPwdSha', get_sha256(f"{get_sha256(password)}Encrypt"))

        params.update(pnginfo=pnginfo)
        self.format = PngImagePlugin.PngImageFile.format
        super().save(fp, format=self.format, **params)
        self.paste(backup)
        backup.close()


# ~~ PIL monkey-patch ~~

def _patched_open(fp, *args, **kwargs):
    """Patched PILImage.open — transparently decrypts encrypted PNG files"""
    try:
        if isinstance(fp, bytes):
            fp = io.BytesIO(fp)

        if not _util.is_path(fp) or not Path(fp).suffix:
            return _pil_open_original(fp, *args, **kwargs)

        img = _pil_open_original(fp, *args, **kwargs)
        try:
            meta = img.info or {}
            if password and img.format and img.format.lower() == 'png':
                meta = decrypt_tags(meta, password)
                if meta.get('Encrypt') == ENCRYPT_MARKER:
                    img.paste(PILImage.fromarray(decrypt_image(img, get_sha256(password))))
                    meta['Encrypt'] = None
            img.info = meta
            return EncryptedImage.from_image(img)
        except Exception as e:
            log.error(f"_patched_open({fp}): {e}")
            return None
        finally:
            img.close()
    except Exception as e:
        log.error(f"_patched_open({fp}): {e}")
        return None

def encode_pil_to_base64(img: PILImage.Image):
    """Convert PIL image to base64-encoded PNG, decrypting pixels and metadata if needed"""
    meta = img.info or {}
    with io.BytesIO() as buf:
        meta = decrypt_tags(meta, password)
        if meta.get('Encrypt') == ENCRYPT_MARKER:
            img.paste(PILImage.fromarray(decrypt_image(img, get_sha256(password))))
        meta['Encrypt'] = None
        img.save(buf, format=PngImagePlugin.PngImageFile.format, quality=shared.opts.jpeg_quality)
        return base64.b64encode(buf.getvalue())


# ~~ Async image serving ~~

_executor = ThreadPoolExecutor(max_workers=100)
_semaphore_cache = {}
_response_cache = {}

def _get_semaphore():
    """Return (or create) a per-event-loop semaphore that caps concurrent image processing"""
    loop = asyncio.get_running_loop()
    if loop not in _semaphore_cache:
        _semaphore_cache[loop] = asyncio.Semaphore(min(os.cpu_count() * 2, 10))
    return _semaphore_cache[loop]

def _resize_image(image, target_height=512):
    """Resize image to target_height maintaining aspect ratio"""
    w, h = image.size
    if h > target_height:
        new_w = int(target_height * w / h)
        return image.resize((new_w, target_height), PILImage.Resampling.LANCZOS)
    return image

def process_image_file(fp: Path, should_resize: bool) -> bytes | None:
    """Encrypt the file on disk if needed, return decrypted PNG bytes for the HTTP response"""
    try:
        with PILImage.open(str(fp)) as image:
            try:
                image.verify()
            except Exception as e:
                log.error(f"Invalid image file: {fp.name}: {e}")
                return None

            if should_resize:
                image = _resize_image(image)
                image.save(str(fp))  # real path → EncryptedImage.save → encrypted to disk

            pnginfo = image.info or {}

            if not all(k in pnginfo for k in IMAGE_KEYS):
                try:
                    EncryptedImage.from_image(image).save(str(fp))
                    image = PILImage.open(str(fp))
                    pnginfo = image.info or {}
                except Exception as e:
                    log.error(f"process_image_file encrypt({fp.name}): {e}")
                    return None

            buf = io.BytesIO()
            info = PngImagePlugin.PngInfo()
            for key, value in pnginfo.items():
                if value is None or key == 'icc_profile':
                    continue
                if isinstance(value, bytes):
                    for enc in ('utf-8', 'utf-16'):
                        try:
                            info.add_text(key, value.decode(enc))
                            break
                        except (UnicodeDecodeError, Exception):
                            pass
                    else:
                        log.warning(f"Cannot decode bytes key '{key}' in {fp.name}")
                else:
                    info.add_text(key, str(value))

            image.save(buf, format=PngImagePlugin.PngImageFile.format, pnginfo=info)
            return buf.getvalue()

    except Exception as e:
        log.error(f"process_image_file({fp.name}): {e}")
        return None

async def serve_image(fp: Path, should_resize: bool):
    """Async wrapper: semaphore-limited, per-request dedup cache, fallback to raw read"""
    semaphore = _get_semaphore()
    try:
        async with semaphore:
            if fp in _response_cache:
                return _response_cache[fp]
            try:
                content = await asyncio.get_running_loop().run_in_executor(
                    _executor, lambda: process_image_file(fp, should_resize)
                )
            except Exception as e:
                log.error(f"serve_image({fp.name}): {e}")
                content = None
            if content:
                _response_cache[fp] = content
            return content
    except Exception as e:
        log.error(f"serve_image semaphore({fp.name}): {e}")
        try:
            with open(str(fp), 'rb') as f:
                return f.read()
        except Exception as ie:
            log.error(f"serve_image fallback({fp.name}): {ie}")
            return None
    finally:
        _response_cache.pop(fp, None)


# ~~ HTTP middleware ~~

def _query_param(query_str: str, prefix: str) -> str:
    """Return value of the first query param whose key matches prefix"""
    return next((part.split('=', 1)[1] for part in query_str.split('&') if part.startswith(prefix)), '')

async def handle_image_request(endpoint: str, query, res):
    """Resolve known endpoints to a file path, encrypt on disk, serve decrypted bytes"""
    query_str = unquote(query())

    # sd-hub gallery: /sdhub-gallery-image=<path>
    if endpoint.startswith('/sdhub-gallery-image='):
        img_path = endpoint[len('/sdhub-gallery-image='):]
        if img_path:
            endpoint = f"/file={img_path}"

    # infinite-image-browsing: path in query string
    if endpoint.startswith(('/infinite_image_browsing/image-thumbnail', '/infinite_image_browsing/file')):
        path = _query_param(query_str, 'path=')
        if path:
            endpoint = f"/file={path}"

    # extra networks thumbnail: filename in query string
    if endpoint.startswith('/sd_extra_networks/thumb'):
        fn = _query_param(query_str, 'filename=')
        if fn:
            endpoint = f"/file={fn}"

    if not endpoint.startswith('/file='):
        return False, None

    fp = Path(endpoint[6:])
    if any(p in fp.name for p in SKIP_PATTERNS):
        return False, None
    if fp.suffix.lower().split('?')[0] not in IMAGE_EXTENSIONS:
        return False, None

    should_resize = str(models_dir) in str(fp) or str(embed_dir) in str(fp)
    content = await serve_image(fp, should_resize)
    if content:
        return True, res(content)
    return False, None

def register_middleware_gradio3(app: FastAPI):
    """Register HTTP middleware for Gradio 3 (FastAPI-based) applications"""
    @app.middleware('http')
    async def _(req: Request, call_next):
        endpoint = '/' + req.scope.get('path', 'err').strip('/')
        def query(): return req.scope.get('query_string', b'').decode('utf-8')
        def res(c): return Response(content=c, media_type='image/png', headers=HEADERS)
        ok, response = await handle_image_request(endpoint, query, res)
        return response if ok else await call_next(req)

def register_middleware_gradio4(app):
    """Register Starlette ASGI middleware for Gradio 4+ applications"""
    import starlette.responses as starlette_resp
    from starlette.types import ASGIApp, Receive, Scope, Send

    class _DecryptMiddleware:
        def __init__(self, inner: ASGIApp):
            self.inner = inner

        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            if scope['type'] == 'http':
                endpoint = '/' + scope.get('path', 'err').strip('/')
                def query(): return scope.get('query_string', b'').decode('utf-8')
                def res(c): return starlette_resp.Response(content=c, media_type='image/png', headers=HEADERS)
                ok, response = await handle_image_request(endpoint, query, res)
                if ok:
                    await response(scope, receive, send)
                    return
            await self.inner(scope, receive, send)

    app.middleware_stack = _DecryptMiddleware(app.middleware_stack)


# ~~ save_image_with_geninfo ~~

def save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name=None):
    """Save image with generation info, encrypting automatically via EncryptedImage.save"""
    try:
        if extension is None:
            extension = os.path.splitext(filename)[1].lower()

        if password:
            enc_img = EncryptedImage.from_image(image)
            enc_img.info = {**(existing_pnginfo or {}), (pnginfo_section_name or 'parameters'): geninfo}
            enc_img.save(filename, quality=shared.opts.jpeg_quality if extension != '.png' else None)
            enc_img.close()
        else:
            if extension == '.png':
                pi = PngInfo()
                pi.add_text(pnginfo_section_name or 'parameters', geninfo)
                for k, v in (existing_pnginfo or {}).items():
                    if k != pnginfo_section_name and v:
                        pi.add_text(k, str(v))
                image.save(filename, format='PNG', pnginfo=pi)
            else:
                image.save(filename, quality=shared.opts.jpeg_quality)

    except Exception as e:
        log.error(f"save_image_with_geninfo({filename!r}): {e}")
        raise


# ~~ App startup ~~

def app(_: gr.Blocks, app: FastAPI):
    """Register the appropriate middleware based on Gradio version"""
    if version.parse(gr.__version__).major > 3:
        register_middleware_gradio4(app)
    else:
        app.middleware_stack = None
        register_middleware_gradio3(app)
        app.build_middleware_stack()

# ~~ Init ~~

if PILImage.Image.__name__ != 'EncryptedImage':
    _pil_open_original = PILImage.open
    super_encode_pil_to_base64 = api.encode_pil_to_base64
    super_modules_images_save_image = images.save_image
    super_api_middleware = api.api_middleware

    if password is not None:
        PILImage.Image = EncryptedImage
        PILImage.open = _patched_open
        api.encode_pil_to_base64 = encode_pil_to_base64
        # images.save_image_with_geninfo = save_image_with_geninfo  # TODO: Fix this
from PIL import Image as PILImage, PngImagePlugin, _util, ImagePalette
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Response
from threading import Lock, Event, Thread
from urllib.parse import unquote
from queue import Queue, Empty
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
import hashlib
import asyncio
import base64
import time
import sys
import io
import os

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from modules import shared, script_callbacks, images
from modules.paths_internal import models_path
from modules.api import api

# ANSI color codes for console output
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"

# Constants
ENCRYPT_PREFIX = "ENC:"
TAG_LIST = ['parameters', 'UserComment']
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.avif']
IMAGE_KEYS = ['Encrypt', 'EncryptPwdSha']
HEADERS = {"Cache-Control": "public, max-age=2592000"}

try:
    from modules_forge.forge_canvas.canvas import ForgeCanvas
    FORGE_AVAILABLE = True
except ModuleNotFoundError:
    FORGE_AVAILABLE = False

# Configuration
password = getattr(shared.cmd_opts, 'encrypt_pass', None)
Embed = shared.cmd_opts.embeddings_dir
Models = Path(models_path)

class ImageEncryptionLogger:
    @staticmethod
    def log(message, level="info"):
        prefix = "[ImageEncryption]"
        if level == "error":
            print(f"{prefix} - {COLOR_RED}ERROR:{COLOR_RESET} {message}")
        elif level == "warning":
            print(f"{prefix} - {COLOR_YELLOW}WARNING:{COLOR_RESET} {message}")
        elif level == "success":
            print(f"{prefix} - {COLOR_GREEN}{message}{COLOR_RESET}")
        else:
            print(f"{prefix} - {message}")

def set_shared_options():
    section = ("encrypt_image_is_enable", "Encrypt image")
    option = shared.OptionInfo(
        default="Yes",
        label="Whether the encryption plug-in is enabled",
        section=section
    )
    option.do_not_save = True
    shared.opts.add_option("encrypt_image_is_enable", option)
    shared.opts.data['encrypt_image_is_enable'] = "Yes"

def get_range(input_str: str, offset: int, range_len=4) -> str:
    offset = offset % len(input_str)
    return (input_str * 2)[offset:offset + range_len]

def get_sha256(input_str: str) -> str:
    return hashlib.sha256(input_str.encode('utf-8')).hexdigest()

def shuffle_array(arr, key):
    sha_key = get_sha256(key)
    arr_len = len(arr)
    for i in range(arr_len):
        s_idx = arr_len - i - 1
        to_index = int(get_range(sha_key, i, range_len=8), 16) % (arr_len - i)
        arr[s_idx], arr[to_index] = arr[to_index], arr[s_idx]
    return arr

def encrypt_tags(metadata, password):
    encrypted_metadata = metadata.copy()
    for key in TAG_LIST:
        if key in metadata:
            value = str(metadata[key])
            encrypted_value = ''.join(
                chr(ord(c) ^ ord(password[i % len(password)]))
                for i, c in enumerate(value)
            )
            encrypted_value = base64.b64encode(encrypted_value.encode('utf-8')).decode('utf-8')
            encrypted_metadata[key] = f"{ENCRYPT_PREFIX}{encrypted_value}"
    return encrypted_metadata

def decrypt_tags(metadata, password):
    decrypted_metadata = metadata.copy()
    for key in TAG_LIST:
        if key in metadata and str(metadata[key]).startswith(ENCRYPT_PREFIX):
            encrypted_value = metadata[key][len(ENCRYPT_PREFIX):]
            try:
                decoded = base64.b64decode(encrypted_value).decode('utf-8')
                decrypted_value = ''.join(
                    chr(ord(c) ^ ord(password[i % len(password)]))
                    for i, c in enumerate(decoded)
                )
                decrypted_metadata[key] = decrypted_value
            except Exception as e:
                ImageEncryptionLogger.log(f"Failed to decrypt tag {key}: {e}", "error")
                decrypted_metadata[key] = metadata[key]
    return decrypted_metadata

def encrypt_image(image: Image.Image, password):
    try:
        width, height = image.size
        x_arr = np.arange(width)
        shuffle_array(x_arr, password)
        y_arr = np.arange(height)
        shuffle_array(y_arr, get_sha256(password))
        pixel_array = np.array(image)

        _pixel_array = pixel_array.copy()
        for x in range(height):
            pixel_array[x] = _pixel_array[y_arr[x]]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        _pixel_array = pixel_array.copy()
        for x in range(width):
            pixel_array[x] = _pixel_array[x_arr[x]]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        return pixel_array
    except Exception as e:
        ImageEncryptionLogger.log(f"Encryption error: {e}", "error")
        return np.array(image)

def decrypt_image(image: Image.Image, password):
    try:
        width, height = image.size
        x_arr = np.arange(width)
        shuffle_array(x_arr, password)
        y_arr = np.arange(height)
        shuffle_array(y_arr, get_sha256(password))
        pixel_array = np.array(image)

        _pixel_array = pixel_array.copy()
        for x in range(height):
            pixel_array[y_arr[x]] = _pixel_array[x]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        _pixel_array = pixel_array.copy()
        for x in range(width):
            pixel_array[x_arr[x]] = _pixel_array[x]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        return pixel_array
    except Exception as e:
        ImageEncryptionLogger.log(f"Decryption error: {e}", "error")
        return np.array(image)

class EncryptedImage(PILImage.Image):
    __name__ = "EncryptedImage"

    @staticmethod
    def from_image(image: PILImage.Image):
        image = image.copy()
        img = EncryptedImage()
        img.im = image.im
        img._mode = image.mode
        if image.im.mode:
            try:
                img.mode = image.im.mode
            except Exception:
                pass

        img._size = image.size
        img.format = image.format
        if image.mode in ("P", "PA"):
            img.palette = image.palette.copy() if image.palette else ImagePalette.ImagePalette()

        img.info = image.info.copy()
        return img

    def save(self, fp, format=None, **params):
        filename = ""

        if isinstance(fp, Path):
            filename = str(fp)
        elif _util.is_path(fp):
            filename = fp
        elif fp == sys.stdout:
            try:
                fp = sys.stdout.buffer
            except AttributeError:
                pass

        if not filename and hasattr(fp, "name") and _util.is_path(fp.name):
            filename = fp.name

        if not filename or not password:
            super().save(fp, format=format, **params)
            return

        if self.info.get('Encrypt') == 'pixel_shuffle_3':
            super().save(fp, format=format, **params)
            return

        encrypted_info = encrypt_tags(self.info, password)
        pnginfo = params.get('pnginfo', PngImagePlugin.PngInfo()) or PngImagePlugin.PngInfo()

        back_img = PILImage.new('RGBA', self.size)
        back_img.paste(self)

        try:
            encrypted_img = PILImage.fromarray(encrypt_image(self, get_sha256(password)))
            self.paste(encrypted_img)
            encrypted_img.close()
        except Exception as e:
            if "axes don't match array" in str(e):
                fn = Path(filename)
                os.system(f'rm -f {fn}')
                return

        for key, value in encrypted_info.items():
            if value:
                pnginfo.add_text(key, str(value))

        pnginfo.add_text('Encrypt', 'pixel_shuffle_3')
        pnginfo.add_text('EncryptPwdSha', get_sha256(f'{get_sha256(password)}Encrypt'))

        params.update(pnginfo=pnginfo)
        self.format = PngImagePlugin.PngImageFile.format
        super().save(fp, format=self.format, **params)
        self.paste(back_img)
        back_img.close()

def open_image(fp, *args, **kwargs):
    try:
        if not _util.is_path(fp) or not Path(fp).suffix:
            return super_open(fp, *args, **kwargs)

        if isinstance(fp, bytes):
            return encode_pil_to_base64(fp)

        img = super_open(fp, *args, **kwargs)
        try:
            pnginfo = img.info or {}

            if password and img.format.lower() == PngImagePlugin.PngImageFile.format.lower():
                pnginfo = decrypt_tags(pnginfo, password)

                if pnginfo.get("Encrypt") == 'pixel_shuffle_3':
                    decrypted_img = PILImage.fromarray(decrypt_image(img, get_sha256(password)))
                    img.paste(decrypted_img)
                    decrypted_img.close()
                    pnginfo["Encrypt"] = None

            img.info = pnginfo
            return EncryptedImage.from_image(img)

        except Exception as e:
            ImageEncryptionLogger.log(f"Error processing image {fp}: {e}", "error")
            return None
        finally:
            img.close()

    except Exception as e:
        ImageEncryptionLogger.log(f"Error opening image {fp}: {e}", "error")
        return None

def encode_pil_to_base64(img: PILImage.Image):
    pnginfo = img.info or {}

    with io.BytesIO() as output_bytes:
        pnginfo = decrypt_tags(pnginfo, password)
        if pnginfo.get("Encrypt") == 'pixel_shuffle_3':
            img.paste(PILImage.fromarray(decrypt_image(img, get_sha256(password))))

        pnginfo["Encrypt"] = None
        img.save(output_bytes, format=PngImagePlugin.PngImageFile.format,
                quality=shared.opts.jpeg_quality)
        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

# Async processing setup
_executor = ThreadPoolExecutor(max_workers=100)
_semaphore_factory = lambda: asyncio.Semaphore(min(os.cpu_count() * 2, 10))
_semaphores = {}
p_cache = {}

def resize_image(image, target_height=500):
    width, height = image.size
    if height > target_height:
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        return image.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
    return image

async def process_image_async(fp, should_resize=False):
    loop = asyncio.get_running_loop()
    if loop not in _semaphores:
        _semaphores[loop] = _semaphore_factory()
    semaphore = _semaphores[loop]

    try:
        async with semaphore:
            if fp in p_cache:
                return p_cache[fp]

            try:
                content = await loop.run_in_executor(
                    _executor,
                    lambda: process_image_file(fp, should_resize)
                )
            except Exception as e:
                ImageEncryptionLogger.log(f"Error processing {fp}: {e}", "error")
                return None

            p_cache[fp] = content
            return content
    except Exception as e:
        ImageEncryptionLogger.log(f"Async processing error for {fp}: {e}", "error")
        try:
            with open(fp, 'rb') as f:
                return f.read()
        except Exception as inner_e:
            ImageEncryptionLogger.log(f"File read error for {fp}: {inner_e}", "error")
            return None
    finally:
        if fp in p_cache:
            del p_cache[fp]

def process_image_file(fp, should_resize):
    try:
        with PILImage.open(fp) as image:
            try:
                image.verify()
            except Exception as e:
                ImageEncryptionLogger.log(f"Invalid image file {fp}: {e}", "error")
                return None

            if should_resize:
                image = resize_image(image)
                image.save(fp)

            pnginfo = image.info or {}

            if not all(k in pnginfo for k in IMAGE_KEYS):
                try:
                    EncryptedImage.from_image(image).save(fp)
                    image = PILImage.open(fp)
                    pnginfo = image.info or {}
                except Exception as e:
                    ImageEncryptionLogger.log(f"Encryption error for {fp}: {e}", "error")
                    return None

            buffered = io.BytesIO()
            info = PngImagePlugin.PngInfo()

            for key, value in pnginfo.items():
                if value is None or key == 'icc_profile':
                    continue
                if isinstance(value, bytes):
                    try:
                        info.add_text(key, value.decode('utf-8'))
                    except UnicodeDecodeError:
                        try:
                            info.add_text(key, value.decode('utf-16'))
                        except UnicodeDecodeError:
                            info.add_text(key, str(value))
                            ImageEncryptionLogger.log(f"Decoding error for '{key}' in {fp}", "error")
                else:
                    info.add_text(key, str(value))

            image.save(buffered, format=PngImagePlugin.PngImageFile.format, pnginfo=info)
            image.close()
            return buffered.getvalue()
    except Exception as e:
        ImageEncryptionLogger.log(f"Processing error for {fp}: {e}", "error")
        return None

class FileWatcher:
    def __init__(self, paths, extensions):
        self.observer = Observer()
        self.file_queue = Queue(maxsize=1000)
        self.processed_files = set()
        self.lock = Lock()
        self.shutdown_event = Event()
        self.num_cpus = os.cpu_count()
        self.thread_pool = ThreadPoolExecutor(max_workers=self.num_cpus * 4)
        self.paths = paths
        self.extensions = extensions

    def start(self):
        workers = []
        for _ in range(self.num_cpus * 4):
            worker = Thread(target=self._process_queue, daemon=True)
            worker.start()
            workers.append(worker)

        handler = self.FileEventHandler(self)
        for path in self.paths:
            self.observer.schedule(handler, path, recursive=True)
        self.observer.start()

    def _process_queue(self):
        futures = []
        while not self.shutdown_event.is_set():
            try:
                fp = self.file_queue.get(timeout=0.1)
                if fp:
                    future = self.thread_pool.submit(self._process_file, fp)
                    futures.append(future)
                    futures = [f for f in futures if not f.done()]
                self.file_queue.task_done()
            except Empty:
                continue
            except Exception as e:
                ImageEncryptionLogger.log(f"Queue processing error: {e}", "error")

    def _process_file(self, fp):
        with self.lock:
            if fp in self.processed_files:
                return
            self.processed_files.add(fp)
        try:
            img = PILImage.open(fp)
            pnginfo = img.info or {}
            if not all(k in pnginfo for k in IMAGE_KEYS):
                ImageEncryptionLogger.log(f"Encrypting new file: {fp}")
                EncryptedImage.from_image(img).save(fp)
        except Exception as e:
            ImageEncryptionLogger.log(f"File processing error {fp}: {e}", "error")
            with self.lock:
                self.processed_files.discard(fp)

    class FileEventHandler(FileSystemEventHandler):
        def __init__(self, watcher):
            super().__init__()
            self.watcher = watcher
            self.event_buffer = {}
            self.last_processed = time.time()
            self.buffer_lock = Lock()

        def on_any_event(self, event):
            if event.is_directory:
                return
            fp = Path(event.src_path)
            if fp.suffix.lower() not in self.watcher.extensions:
                return
            if event.event_type in ('created', 'modified', 'moved'):
                with self.buffer_lock:
                    self.event_buffer[fp] = time.time()
                current_time = time.time()
                if current_time - self.last_processed > 0.1:
                    self._process_buffer()

        def _process_buffer(self):
            with self.buffer_lock:
                current_time = time.time()
                files_to_process = []
                for fp, timestamp in list(self.event_buffer.items()):
                    if current_time - timestamp >= 0.1 and fp.exists():
                        files_to_process.append(fp)
                        del self.event_buffer[fp]
                for fp in files_to_process:
                    self.watcher.file_queue.put(fp)
                self.last_processed = current_time

async def handle_image_request(endpoint, query, full_path, response_builder):
    # Handle sd-hub-gallery endpoint
    if endpoint.startswith('/sd-hub-gallery/image='):
        img_path = endpoint[len('/sd-hub-gallery/image='):]
        if img_path:
            endpoint = f'/file={img_path}'

    # Handle infinite image browsing endpoints
    if endpoint.startswith(('/infinite_image_browsing/image-thumbnail', '/infinite_image_browsing/file')):
        query_string = unquote(query())
        if query_string and 'path=' in query_string:
            query_parts = query_string.split('&')
            path = ''
            for sub in query_parts:
                if sub.startswith('path='):
                    path = sub[sub.index('=')+1:]
            if path:
                endpoint = f'/file={path}'

    # Handle extra networks thumbnail endpoint
    if endpoint.startswith('/sd_extra_networks/thumb'):
        query_string = unquote(query())
        filename = next((sub.split('=')[1] for sub in query_string.split('&') if sub.startswith('filename=')), '')
        if filename:
            endpoint = f'/file={filename}'

    if endpoint.startswith('/file='):
        fp = full_path(endpoint[6:])
        ext = fp.suffix.lower().split('?')[0]

        if 'card-no-preview.png' in str(fp):
            return False, None

        if ext in IMAGE_EXTENSIONS:
            should_resize = str(Models) in str(fp) or str(Embed) in str(fp)
            content = await process_image_async(fp, should_resize)
            if content:
                return True, response_builder(content)

    return False, None

def setup_http_middleware(app: FastAPI):
    @app.middleware("http")
    async def image_decryption_middleware(req: Request, call_next):
        endpoint = '/' + req.scope.get('path', 'err').strip('/')

        def get_query():
            return req.scope.get('query_string', b'').decode('utf-8')

        def build_response(content):
            return Response(content=content, media_type='image/png', headers=HEADERS)

        success, response = await handle_image_request(
            endpoint=endpoint,
            query=get_query,
            full_path=Path,
            response_builder=build_response
        )
        if success:
            return response

        return await call_next(req)

def setup_forge_middleware(app):
    import starlette.responses as responses
    from starlette.types import ASGIApp, Receive, Scope, Send

    class ForgeMiddleware:
        def __init__(self, app: ASGIApp):
            self.app = app

        async def __call__(self, scope: Scope, receive: Receive, send: Send):
            if scope["type"] == "http":
                endpoint = '/' + scope.get('path', 'err').strip('/')

                def get_query():
                    return scope.get('query_string', b'').decode('utf-8')

                def build_response(content):
                    return responses.Response(content=content, media_type='image/png', headers=HEADERS)

                success, response = await handle_image_request(
                    endpoint=endpoint,
                    query=get_query,
                    full_path=Path,
                    response_builder=build_response
                )
                if success:
                    await response(scope, receive, send)
                    return

            await self.app(scope, receive, send)

    app.middleware_stack = ForgeMiddleware(app.middleware_stack)

def on_app_started(_: gr.Blocks, app: FastAPI):
    set_shared_options()

    if not FORGE_AVAILABLE:
        app.middleware_stack = None
        setup_http_middleware(app)
        app.build_middleware_stack()
    else:
        setup_forge_middleware(app)

# Main initialization
if PILImage.Image.__name__ != 'EncryptedImage':
    super_open = PILImage.open
    super_encode_pil_to_base64 = api.encode_pil_to_base64
    super_modules_images_save_image = images.save_image
    super_api_middleware = api.api_middleware

    if password is not None:
        PILImage.Image = EncryptedImage
        PILImage.open = open_image
        api.encode_pil_to_base64 = encode_pil_to_base64

    # Start file watcher
    watcher = FileWatcher([Models, Embed], set(IMAGE_EXTENSIONS))
    watcher.start()
    ImageEncryptionLogger.log("File watcher started")


# Fix XYZ-Plot Saving Encrypt-Image
def save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name=None):
    """
    Save image with generation info, supporting all formats in IMAGE_EXTENSIONS.
    Handles encryption for both single images and XYZ grids.
    """
    try:
        # Determine file extension if not provided
        if extension is None:
            extension = os.path.splitext(filename)[1].lower()

        # Prepare metadata
        parameters = geninfo
        metadata = existing_pnginfo or {}

        # For encrypted images, prepare special handling
        if password and extension in IMAGE_EXTENSIONS:
            # Create encrypted metadata
            encrypted_metadata = encrypt_tags(metadata, password)
            encrypted_metadata[pnginfo_section_name or 'parameters'] = parameters

            # Create new image object for encryption
            enc_image = EncryptedImage.from_image(image)
            enc_image.info = encrypted_metadata

            # Special handling for PNG to preserve metadata
            if extension == '.png':
                pnginfo = PngImagePlugin.PngInfo()
                for k, v in encrypted_metadata.items():
                    if v: pnginfo.add_text(k, str(v))
                enc_image.save(filename, format='PNG', pnginfo=pnginfo)
            else:
                # For non-PNG formats, we can't store metadata as extensively
                enc_image.save(filename, quality=shared.opts.jpeg_quality)

            enc_image.close()
        else:
            # Standard saving for non-encrypted or unsupported formats
            if extension == '.png':
                pnginfo = PngImagePlugin.PngInfo()
                pnginfo.add_text(pnginfo_section_name or 'parameters', parameters)
                if existing_pnginfo:
                    for k, v in existing_pnginfo.items():
                        if k != pnginfo_section_name and v:
                            pnginfo.add_text(k, str(v))
                image.save(filename, format='PNG', pnginfo=pnginfo)
            else:
                image.save(filename, quality=shared.opts.jpeg_quality)

    except Exception as e:
        ImageEncryptionLogger.log(f"Error saving image {filename}: {e}", "error")
        raise

# Replace the original save_image_with_geninfo in images module
images.save_image_with_geninfo = save_image_with_geninfo


# Handle different password states
if password == '':
    ImageEncryptionLogger.log("Disabled - empty password provided", "error")
elif not password:
    ImageEncryptionLogger.log("Disabled - missing password argument", "error")
else:
    script_callbacks.on_app_started(on_app_started)
    ImageEncryptionLogger.log("Enabled V1", "success")
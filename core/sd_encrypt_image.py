from PIL import Image as PILImage, PngImagePlugin, _util, ImagePalette
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, Request, Response
from urllib.parse import unquote
from pathlib import Path
from PIL import Image
import gradio as gr
import numpy as np
import asyncio
import hashlib
import base64
import sys
import io
import os

from modules.paths_internal import models_path
from modules import shared, images
from modules.api import api

# ANSI color codes for console output
COLOR_RED = "\033[31m"
COLOR_GREEN = "\033[32m"
COLOR_YELLOW = "\033[33m"
COLOR_BLUE = "\033[34m"
COLOR_RESET = "\033[0m"

# Configuration constants
ENCRYPT_PREFIX = "ENC:"
TAG_LIST = ['parameters', 'UserComment']
IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.webp', '.avif']
IMAGE_KEYS = ['Encrypt', 'EncryptPwdSha']
HEADERS = {"Cache-Control": "public, max-age=2592000"}

# Check for Forge availability
try:
    from modules_forge.forge_canvas.canvas import ForgeCanvas
    FORGE_AVAILABLE = True
except ModuleNotFoundError:
    FORGE_AVAILABLE = False

# Get configuration from command line arguments
password = getattr(shared.cmd_opts, 'encrypt_pass', None)
embed_dir = shared.cmd_opts.embeddings_dir
models_dir = Path(models_path)

class ImageEncryptionLogger:
    """Centralized logging for image encryption operations"""

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

def get_range(input_str: str, offset: int, range_len=4) -> str:
    """Extract a range of characters from a string with wrapping"""
    offset = offset % len(input_str)
    return (input_str * 2)[offset:offset + range_len]

def get_sha256(input_str: str) -> str:
    """Generate SHA256 hash of input string"""
    return hashlib.sha256(input_str.encode('utf-8')).hexdigest()

def shuffle_array(arr, key):
    """Shuffle array using deterministic algorithm based on key"""
    sha_key = get_sha256(key)
    arr_len = len(arr)
    for i in range(arr_len):
        s_idx = arr_len - i - 1
        to_index = int(get_range(sha_key, i, range_len=8), 16) % (arr_len - i)
        arr[s_idx], arr[to_index] = arr[to_index], arr[s_idx]
    return arr

def encrypt_tags(metadata, password):
    """Encrypt metadata tags using XOR cipher with base64 encoding"""
    encrypted_metadata = metadata.copy()
    for key in TAG_LIST:
        if key in metadata:
            value = str(metadata[key])
            # XOR encrypt each character with password
            encrypted_value = ''.join(
                chr(ord(c) ^ ord(password[i % len(password)]))
                for i, c in enumerate(value)
            )
            # Base64 encode the encrypted value
            encrypted_value = base64.b64encode(encrypted_value.encode('utf-8')).decode('utf-8')
            encrypted_metadata[key] = f"{ENCRYPT_PREFIX}{encrypted_value}"
    return encrypted_metadata

def decrypt_tags(metadata, password):
    """Decrypt metadata tags using XOR cipher with base64 decoding"""
    decrypted_metadata = metadata.copy()
    for key in TAG_LIST:
        if key in metadata and str(metadata[key]).startswith(ENCRYPT_PREFIX):
            encrypted_value = metadata[key][len(ENCRYPT_PREFIX):]
            try:
                # Base64 decode the encrypted value
                decoded = base64.b64decode(encrypted_value).decode('utf-8')
                # XOR decrypt each character with password
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
    """Encrypt image pixels using pixel shuffling algorithm"""
    try:
        width, height = image.size

        # Create shuffle arrays for both dimensions
        x_arr = np.arange(width)
        shuffle_array(x_arr, password)
        y_arr = np.arange(height)
        shuffle_array(y_arr, get_sha256(password))

        pixel_array = np.array(image)

        # Shuffle rows
        _pixel_array = pixel_array.copy()
        for x in range(height):
            pixel_array[x] = _pixel_array[y_arr[x]]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        # Shuffle columns
        _pixel_array = pixel_array.copy()
        for x in range(width):
            pixel_array[x] = _pixel_array[x_arr[x]]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        return pixel_array
    except Exception as e:
        ImageEncryptionLogger.log(f"Encryption error: {e}", "error")
        return np.array(image)

def decrypt_image(image: Image.Image, password):
    """Decrypt image pixels using reverse pixel shuffling algorithm"""
    try:
        width, height = image.size

        # Create shuffle arrays (same as encryption)
        x_arr = np.arange(width)
        shuffle_array(x_arr, password)
        y_arr = np.arange(height)
        shuffle_array(y_arr, get_sha256(password))

        pixel_array = np.array(image)

        # Reverse shuffle rows
        _pixel_array = pixel_array.copy()
        for x in range(height):
            pixel_array[y_arr[x]] = _pixel_array[x]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        # Reverse shuffle columns
        _pixel_array = pixel_array.copy()
        for x in range(width):
            pixel_array[x_arr[x]] = _pixel_array[x]
        pixel_array = np.transpose(pixel_array, axes=(1, 0, 2))

        return pixel_array
    except Exception as e:
        ImageEncryptionLogger.log(f"Decryption error: {e}", "error")
        return np.array(image)

class EncryptedImage(PILImage.Image):
    """Extended PIL Image class with encryption capabilities"""

    __name__ = "EncryptedImage"

    @staticmethod
    def from_image(image: PILImage.Image):
        """Create EncryptedImage from PIL Image"""
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

        # Handle palette images
        if image.mode in ("P", "PA"):
            img.palette = image.palette.copy() if image.palette else ImagePalette.ImagePalette()

        img.info = image.info.copy()
        return img

    def save(self, fp, format=None, **params):
        """Save image with encryption if password is available"""
        filename = ""

        # Extract filename from various input types
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

        # Skip encryption if no password or already encrypted
        if not filename or not password:
            super().save(fp, format=format, **params)
            return

        if self.info.get('Encrypt') == 'pixel_shuffle_3':
            super().save(fp, format=format, **params)
            return

        # Encrypt metadata
        encrypted_info = encrypt_tags(self.info, password)
        pnginfo = params.get('pnginfo', PngImagePlugin.PngInfo()) or PngImagePlugin.PngInfo()

        # Create backup of original image
        back_img = PILImage.new('RGBA', self.size)
        back_img.paste(self)

        try:
            # Encrypt image pixels
            encrypted_img = PILImage.fromarray(encrypt_image(self, get_sha256(password)))
            self.paste(encrypted_img)
            encrypted_img.close()
        except Exception as e:
            if "axes don't match array" in str(e):
                # Handle dimension mismatch by removing file
                fn = Path(filename)
                if fn.exists():
                    fn.unlink()
                return

        # Add encrypted metadata to PNG info
        for key, value in encrypted_info.items():
            if value:
                pnginfo.add_text(key, str(value))

        # Add encryption markers
        pnginfo.add_text('Encrypt', 'pixel_shuffle_3')
        pnginfo.add_text('EncryptPwdSha', get_sha256(f'{get_sha256(password)}Encrypt'))

        # Save encrypted image
        params.update(pnginfo=pnginfo)
        self.format = PngImagePlugin.PngImageFile.format
        super().save(fp, format=self.format, **params)

        # Restore original image data
        self.paste(back_img)
        back_img.close()

def open_image(fp, *args, **kwargs):
    """Open image with automatic decryption if needed"""
    try:
        if not _util.is_path(fp) or not Path(fp).suffix:
            return super_open(fp, *args, **kwargs)

        if isinstance(fp, bytes):
            return encode_pil_to_base64(fp)

        img = super_open(fp, *args, **kwargs)
        try:
            pnginfo = img.info or {}

            # Decrypt if password available and image is PNG
            if password and img.format.lower() == PngImagePlugin.PngImageFile.format.lower():
                pnginfo = decrypt_tags(pnginfo, password)

                # Check if image is encrypted and decrypt
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
    """Convert PIL image to base64 with automatic decryption"""
    pnginfo = img.info or {}

    with io.BytesIO() as output_bytes:
        # Decrypt if needed
        pnginfo = decrypt_tags(pnginfo, password)
        if pnginfo.get("Encrypt") == 'pixel_shuffle_3':
            img.paste(PILImage.fromarray(decrypt_image(img, get_sha256(password))))

        pnginfo["Encrypt"] = None
        img.save(output_bytes, format=PngImagePlugin.PngImageFile.format,
                quality=shared.opts.jpeg_quality)
        bytes_data = output_bytes.getvalue()

    return base64.b64encode(bytes_data)

# Async processing configuration
_executor = ThreadPoolExecutor(max_workers=100)
_semaphore_factory = lambda: asyncio.Semaphore(min(os.cpu_count() * 2, 10))
_semaphores = {}
p_cache = {}

def resize_image(image, target_height=500):
    """Resize image maintaining aspect ratio"""
    width, height = image.size
    if height > target_height:
        aspect_ratio = width / height
        new_width = int(target_height * aspect_ratio)
        return image.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
    return image

async def process_image_async(fp, should_resize=False):
    """Asynchronously process image file"""
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
    """Process image file with encryption handling"""
    try:
        with PILImage.open(fp) as image:
            # Verify image integrity
            try:
                image.verify()
            except Exception as e:
                ImageEncryptionLogger.log(f"Invalid image file {fp}: {e}", "error")
                return None

            # Resize if requested
            if should_resize:
                image = resize_image(image)
                image.save(fp)

            pnginfo = image.info or {}

            # Encrypt image if not already encrypted
            if not all(k in pnginfo for k in IMAGE_KEYS):
                try:
                    EncryptedImage.from_image(image).save(fp)
                    image = PILImage.open(fp)
                    pnginfo = image.info or {}
                except Exception as e:
                    ImageEncryptionLogger.log(f"Encryption error for {fp}: {e}", "error")
                    return None

            # Prepare image for HTTP response
            buffered = io.BytesIO()
            info = PngImagePlugin.PngInfo()

            # Add metadata to PNG info
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

async def handle_image_request(endpoint, query, full_path, response_builder):
    """Handle HTTP requests for images with proper endpoint parsing"""

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

    # Process file requests
    if endpoint.startswith('/file='):
        fp = full_path(endpoint[6:])
        ext = fp.suffix.lower().split('?')[0]

        # Skip preview placeholder images
        if 'card-no-preview.png' in str(fp):
            return False, None

        # Process image files
        if ext in IMAGE_EXTENSIONS:
            should_resize = str(models_dir) in str(fp) or str(embed_dir) in str(fp)
            content = await process_image_async(fp, should_resize)
            if content:
                return True, response_builder(content)

    return False, None

def setup_http_middleware(app: FastAPI):
    """Setup HTTP middleware for standard FastAPI applications"""
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
    """Setup middleware for Forge applications"""
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

def save_image_with_geninfo(image, geninfo, filename, extension=None, existing_pnginfo=None, pnginfo_section_name=None):
    """Save image with generation info, supporting encryption for all formats"""
    try:
        # Determine file extension if not provided
        if extension is None:
            extension = os.path.splitext(filename)[1].lower()

        # Prepare metadata
        parameters = geninfo
        metadata = existing_pnginfo or {}

        # Handle encrypted images
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
                    if v:
                        pnginfo.add_text(k, str(v))
                enc_image.save(filename, format='PNG', pnginfo=pnginfo)
            else:
                # For non-PNG formats, metadata support is limited
                enc_image.save(filename, quality=shared.opts.jpeg_quality)

            enc_image.close()
        else:
            # Standard saving for non-encrypted images
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

def app(_: gr.Blocks, app: FastAPI):
    """Initialize the application when started"""
    # Setup appropriate middleware based on platform
    if not FORGE_AVAILABLE:
        app.middleware_stack = None
        setup_http_middleware(app)
        app.build_middleware_stack()
    else:
        setup_forge_middleware(app)

# Initialize the encryption system
if PILImage.Image.__name__ != 'EncryptedImage':
    # Store original functions
    super_open = PILImage.open
    super_encode_pil_to_base64 = api.encode_pil_to_base64
    super_modules_images_save_image = images.save_image
    super_api_middleware = api.api_middleware

    # Apply encryption if password is provided
    if password is not None:
        PILImage.Image = EncryptedImage
        PILImage.open = open_image
        api.encode_pil_to_base64 = encode_pil_to_base64

        # Replace the original save function
        images.save_image_with_geninfo = save_image_with_geninfo
"""Loopback inference server for requests forwarded by SSH.

The server binds to 127.0.0.1 only. From the edge, local forwarding lands on
this process:

    ssh -N -T -L 127.0.0.1:<edge-port>:127.0.0.1:<this-port> <user>@<cloud-host>

The edge then POSTs an image and a request type tag to the edge port.
OpenSSH delivers that connection to this loopback port.

Request types are face, ocr, and object detection. POST /v1/infer accepts:

    {
      "type": "face",
      "image_base64": "<base64 image bytes>",
      "filename": "frame.jpg",
      "request_id": "optional-token"
    }

Keep this contract in sync with src/Edge/client.py.

Run from the repository root:

    python src/Cloud/server.py --port 8765
"""

import argparse
import base64
import binascii
import collections.abc
import hashlib
import json
import logging
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("faas.cloud.server")

# Keep these names and values in sync with src/Edge/client.py.
SERVICE_NAME = "faas-cloud"
HEALTH_PATH = "/health"
INFER_PATH = "/v1/infer"
DEFAULT_CLOUD_HOST = "127.0.0.1"
DEFAULT_CLOUD_PORT = 8765
DEFAULT_LOCAL_PORT = 18765
REQUEST_TYPES = ("face", "ocr", "object_detection")
TYPE_HELP = "face, ocr, or object detection"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
MAX_BODY_BYTES = 32 * 1024 * 1024
MAX_PIXELS = 25_000_000

Executor = Callable[[str, Any, str, bytes], Any]


class HttpInputError(Exception):
    """A request the server can reject without running a model."""

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


def canonical_request_type(value: Any) -> str | None:
    """Map a request tag to face, ocr, or object_detection.

    Spaces, underscores, and hyphens are treated the same, so
    "object detection" and "object_detection" are one tag.
    """
    if not isinstance(value, str):
        return None
    key = " ".join(value.strip().lower().replace("-", " ").replace("_", " ").split())
    aliases = {
        "face": "face",
        "facial": "face",
        "facial recognition": "face",
        "ocr": "ocr",
        "text": "ocr",
        "object": "object_detection",
        "objects": "object_detection",
        "object detection": "object_detection",
        "yolo": "object_detection",
    }
    return aliases.get(key)


def clean_filename(value: str) -> str:
    """Use a short basename as the expert cache key and log field."""
    name = Path(str(value)).name.replace("\x00", "").strip()
    if name in {"", ".", ".."}:
        return "image"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned[:120] or "image"


def clean_request_id(value: Any) -> str:
    """Keep a caller token when it is a short id, otherwise mint one."""
    if isinstance(value, str):
        token = value.strip()
        if token and len(token) <= 64 and all(ch.isalnum() or ch in "_-" for ch in token):
            return token
    return hashlib.sha256(str(time.time_ns()).encode("ascii")).hexdigest()[:12]


def local_forward_spec(cloud_port: int, local_port: int = DEFAULT_LOCAL_PORT) -> str:
    """Return the ssh -L target: edge loopback to this server's loopback."""
    return f"127.0.0.1:{local_port}:127.0.0.1:{cloud_port}"


def local_forward_command_hint(cloud_port: int, local_port: int = DEFAULT_LOCAL_PORT) -> str:
    """Show the local forward an edge must open to reach this server."""
    return f"ssh -N -T -L {local_forward_spec(cloud_port, local_port)} <user>@<cloud-host>"


def resolve_loopback_host(host: str) -> str:
    """Accept only loopback. The SSH forward is the path onto this port."""
    cleaned = (host or "").strip().lower()
    if cleaned == "localhost":
        return DEFAULT_CLOUD_HOST
    if cleaned == DEFAULT_CLOUD_HOST:
        return cleaned
    raise ValueError(
        f"Refusing to listen on {host!r}. Bind the cloud server to 127.0.0.1 "
        "so clients can reach it only through an SSH local forward."
    )


def decode_image_bytes(image_base64: str) -> bytes:
    """Decode the edge's base64 image and enforce the size limit."""
    if not isinstance(image_base64, str) or not image_base64.strip():
        raise ValueError("image_base64 is required.")
    text = "".join(image_base64.split())
    if text.startswith("data:"):
        comma = text.find(",")
        if comma == -1:
            raise ValueError("image_base64 data URL is missing a comma.")
        text = text[comma + 1 :]
    padding = (-len(text)) % 4
    try:
        raw = base64.b64decode(text + ("=" * padding), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64.") from exc
    if not raw:
        raise ValueError("image_base64 decoded to an empty file.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(raw)} bytes. The limit is {MAX_IMAGE_BYTES} bytes."
        )
    return raw


def decode_image(raw: bytes):
    """Turn image bytes into the BGR array the experts already accept."""
    frame = _decode_bgr(raw)
    if frame is None or getattr(frame, "size", 0) == 0 or getattr(frame, "ndim", 0) < 2:
        raise ValueError("Could not decode the image. Use a PNG, JPEG, or BMP file.")
    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height <= 0 or width <= 0 or height * width > MAX_PIXELS:
        raise ValueError("Image dimensions are not supported.")
    return frame


def _decode_bgr(raw: bytes):
    """Decode with OpenCV when it is installed, otherwise with Pillow."""
    try:
        import cv2 as cv
        import numpy as np
    except ImportError:
        return _decode_bgr_with_pillow(raw)

    array = np.frombuffer(raw, dtype=np.uint8)
    return cv.imdecode(array, cv.IMREAD_COLOR)


def _decode_bgr_with_pillow(raw: bytes):
    import io

    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "opencv-python or Pillow, plus numpy, is required to decode images on the cloud."
        ) from exc
    try:
        with Image.open(io.BytesIO(raw)) as image:
            rgb = np.asarray(image.convert("RGB"))
    except OSError as exc:
        raise ValueError("Could not decode the image. Use a PNG, JPEG, or BMP file.") from exc
    if getattr(rgb, "ndim", 0) != 3 or rgb.shape[2] != 3:
        raise ValueError("Could not decode the image. Use a PNG, JPEG, or BMP file.")
    # Experts expect BGR, which is what cv2.imdecode(IMREAD_COLOR) returns.
    return np.ascontiguousarray(rgb[:, :, ::-1])


def result_to_text(result: Any) -> str | None:
    """Flatten an expert return value into the JSON result string."""
    if result is None:
        return None
    if isinstance(result, str):
        return result.strip()
    # Experts may stream. Numpy arrays are iterable too, so only join iterators.
    if isinstance(result, collections.abc.Iterator) and not isinstance(result, (bytes, bytearray)):
        result = "".join(str(chunk) for chunk in result)
    text = result if isinstance(result, str) else str(result)
    return text.strip()


def handle_infer(payload: Any, executor: Executor) -> tuple[int, dict]:
    """Validate one request and run the matching expert."""
    if not isinstance(payload, dict):
        return 400, {"ok": False, "error": "Request body must be a JSON object."}

    request_id = clean_request_id(payload.get("request_id"))
    task = canonical_request_type(payload.get("type"))
    if task is None:
        return 400, {
            "ok": False,
            "error": f"Request type must be {TYPE_HELP}.",
            "request_id": request_id,
        }

    raw_name = payload.get("filename")
    filename = clean_filename(raw_name if isinstance(raw_name, str) else "image")
    try:
        image_bytes = decode_image_bytes(payload.get("image_base64"))
        frame = decode_image(image_bytes)
    except ValueError as exc:
        return 400, {"ok": False, "error": str(exc), "type": task, "request_id": request_id}
    except RuntimeError as exc:
        logger.error("Image decode is unavailable: %s", exc)
        return 500, {"ok": False, "error": str(exc), "type": task, "request_id": request_id}

    logger.info(
        "request %s type=%s filename=%s bytes=%d",
        request_id,
        task,
        filename,
        len(image_bytes),
    )
    try:
        result = executor(task, frame, filename, image_bytes)
    except Exception as exc:
        logger.exception("Inference failed for request %s.", request_id)
        message = str(exc).strip() or exc.__class__.__name__
        if len(message) > 300:
            message = message[:300] + "..."
        return 500, {
            "ok": False,
            "error": f"Inference failed: {message}",
            "type": task,
            "request_id": request_id,
        }

    text = result_to_text(result)
    if text is None:
        return 500, {
            "ok": False,
            "error": f"No expert is registered for request type '{task}'.",
            "type": task,
            "request_id": request_id,
        }
    return 200, {"ok": True, "type": task, "result": text, "request_id": request_id}


def _load_get_expert_manager():
    """Import ExpertManager without loading a second copy of the package."""
    existing = sys.modules.get("src.ExpertManager")
    if existing is not None and hasattr(existing, "get_expert_manager"):
        return existing.get_expert_manager

    src_dir = Path(__file__).resolve().parent.parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from ExpertManager import get_expert_manager

    return get_expert_manager


def make_expert_executor() -> Executor:
    """Route face, ocr, and object_detection onto the existing experts."""
    manager = _load_get_expert_manager()()

    def execute(task: str, frame, filename: str, image_bytes: bytes):
        digest = hashlib.sha256(image_bytes).hexdigest()[:16]
        input_key = f"{task}:{filename}:{digest}"
        return manager.execute(task, input_key, frame)

    return execute


class _CloudHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True


def make_handler(executor: Executor):
    """Build a request handler closed over the executor and its lock."""
    infer_lock = threading.Lock()

    class InferenceHandler(BaseHTTPRequestHandler):
        server_version = "FaaSCloud/1"
        sys_version = ""

        def version_string(self):
            return "FaaSCloud/1"

        def log_message(self, fmt, *args):
            try:
                message = fmt % args
            except Exception:
                message = fmt
            logger.info("%s - %s", self.address_string(), message)

        def do_GET(self):
            if self._route() != HEALTH_PATH:
                self._write_json(404, {"ok": False, "error": "Not found."})
                return
            self._write_json(
                200,
                {
                    "ok": True,
                    "service": SERVICE_NAME,
                    "request_types": list(REQUEST_TYPES),
                },
            )

        def do_POST(self):
            if self._route() != INFER_PATH:
                self._write_json(404, {"ok": False, "error": "Not found."})
                return
            try:
                payload = self._read_json_object()
            except HttpInputError as exc:
                self._write_json(exc.status, {"ok": False, "error": exc.message})
                return

            with infer_lock:
                try:
                    status, body = handle_infer(payload, executor)
                except Exception:
                    logger.exception("Unhandled error while handling an inference request.")
                    status = 500
                    body = {
                        "ok": False,
                        "error": "Cloud server failed while handling the request.",
                    }
            self._write_json(status, body)

        def _route(self) -> str:
            path = urllib.parse.urlsplit(self.path).path
            if len(path) > 1:
                path = path.rstrip("/")
            return path

        def _read_json_object(self) -> dict:
            length_header = self.headers.get("Content-Length")
            if length_header is None:
                raise HttpInputError(411, "Content-Length is required.")
            try:
                length = int(length_header)
            except ValueError as exc:
                raise HttpInputError(400, "Content-Length must be an integer.") from exc
            if length <= 0:
                raise HttpInputError(400, "Request body is empty.")
            if length > MAX_BODY_BYTES:
                raise HttpInputError(413, "Request body is too large.")

            # Bound the upload. Inference itself can run longer than this read.
            self.connection.settimeout(30)
            try:
                raw = self.rfile.read(length)
            finally:
                self.connection.settimeout(None)
            if len(raw) != length:
                raise HttpInputError(400, "Request body ended early.")
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HttpInputError(400, "Request body must be UTF-8 JSON.") from exc
            if not isinstance(payload, dict):
                raise HttpInputError(400, "Request body must be a JSON object.")
            return payload

        def _write_json(self, status: int, payload: dict):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

    return InferenceHandler


def create_server(host: str, port: int, executor: Executor) -> _CloudHTTPServer:
    """Bind the inference server. `port` 0 asks the OS for an ephemeral port."""
    bind_host = resolve_loopback_host(host)
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer from 0 to 65535.")
    server = _CloudHTTPServer((bind_host, port), make_handler(executor))
    return server


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Listen on 127.0.0.1 for an inference request delivered by an SSH "
            "local forward. The edge sends an image and a request type tag: "
            f"{TYPE_HELP}. This port is not reachable off this machine except "
            "through that forward."
        )
    )
    parser.add_argument(
        "--host",
        default=DEFAULT_CLOUD_HOST,
        help="Loopback address to bind. Must stay 127.0.0.1.",
    )
    parser.add_argument(
        "--port",
        type=_cli_port,
        default=DEFAULT_CLOUD_PORT,
        help="Loopback port the SSH local forward connects to.",
    )
    return parser.parse_args(argv)


def _cli_port(text: str) -> int:
    try:
        port = int(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return port


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        host = resolve_loopback_host(args.host)
    except ValueError as exc:
        logger.error("%s", exc)
        return 2

    try:
        executor = make_expert_executor()
    except Exception:
        logger.exception("Could not load the inference experts.")
        return 1

    try:
        server = create_server(host, args.port, executor)
    except OSError as exc:
        logger.error("Could not bind %s:%s (%s).", host, args.port, exc)
        return 1

    bound_host, bound_port = server.server_address
    logger.info("Cloud inference server listening on http://%s:%s", bound_host, bound_port)
    logger.info("POST %s accepts an image and a request type: %s.", INFER_PATH, TYPE_HELP)
    logger.info("Models load on the first request for each expert.")
    logger.info("SSH local forward from the edge: %s", local_forward_command_hint(bound_port))
    logger.info(
        "Edge script: python src/Edge/client.py --ssh-host <cloud-host> "
        "--ssh-user <user> --image <file> --type face"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        logger.info("Stopping cloud server.")
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

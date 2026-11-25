"""Send an image and a request type to the cloud through an SSH local forward.

The cloud server listens on the cloud machine's loopback. This script opens
the forward with OpenSSH, then posts to the local end of the tunnel:

    ssh -N -T -L 127.0.0.1:<local-port>:127.0.0.1:<cloud-port> <user>@<cloud-host>

Request type tags are face, ocr, and object detection. The JSON body is the
contract implemented by src/Cloud/server.py:

    POST /v1/infer
    {"type": "face", "image_base64": "...", "filename": "frame.jpg", "request_id": "..."}

Run from the repository root after the cloud server is up:

    python src/Edge/client.py --ssh-host 10.0.0.8 --ssh-user faas --image frame.jpg --type face

Authentication is non-interactive. Use an SSH key (ssh-agent or --identity).
Host entries in the normal SSH config are picked up by the ssh program.
"""

import argparse
import base64
import json
import logging
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger("faas.edge.client")

# Keep these names and values in sync with src/Cloud/server.py.
SERVICE_NAME = "faas-cloud"
HEALTH_PATH = "/health"
INFER_PATH = "/v1/infer"
DEFAULT_CLOUD_HOST = "127.0.0.1"
DEFAULT_CLOUD_PORT = 8765
DEFAULT_LOCAL_PORT = 18765
REQUEST_TYPES = ("face", "ocr", "object_detection")
TYPE_HELP = "face, ocr, or object detection"
MAX_IMAGE_BYTES = 20 * 1024 * 1024
STRICT_HOST_KEY_VALUES = ("yes", "no", "accept-new")


class SshForwardError(Exception):
    """The SSH local forward did not come up, or ssh exited."""


class CloudResponseError(Exception):
    """The tunnel was up, but the cloud request did not succeed at the HTTP layer."""


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
    """Use a short basename as the filename sent to the cloud."""
    name = Path(str(value)).name.replace("\x00", "").strip()
    if name in {"", ".", ".."}:
        return "image"
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in name)
    cleaned = cleaned.strip("._")
    return cleaned[:120] or "image"


def _reject_host_text(value: str, label: str) -> str:
    text = (value or "").strip()
    if (
        not text
        or text.startswith("-")
        or any(ch.isspace() for ch in text)
        or any(ch in text for ch in ":@/\\")
    ):
        raise ValueError(f"{label} must be a hostname or IPv4 address.")
    return text


def _validate_ssh_user(value: str) -> str:
    text = (value or "").strip()
    if (
        not text
        or text.startswith("-")
        or any(ch.isspace() for ch in text)
        or any(ch in text for ch in ":@/\\")
    ):
        raise ValueError("SSH user must not contain spaces, ':', or '@'.")
    return text


def _loopback_target(value: str, label: str) -> str:
    """Keep the -L endpoints on IPv4 loopback when the caller says localhost."""
    text = _reject_host_text(value, label)
    if text.lower() == "localhost":
        return DEFAULT_CLOUD_HOST
    return text


def _validate_port(port: int, label: str) -> int:
    if isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535:
        raise ValueError(f"{label} must be an integer from 1 to 65535.")
    return port


@dataclass(frozen=True)
class ForwardSettings:
    """Everything needed to open `ssh -L` and to address the local end."""

    ssh_host: str
    ssh_user: str
    ssh_executable: str
    ssh_port: int = 22
    identity: str | None = None
    local_host: str = DEFAULT_CLOUD_HOST
    local_port: int = DEFAULT_LOCAL_PORT
    forward_host: str = DEFAULT_CLOUD_HOST
    remote_port: int = DEFAULT_CLOUD_PORT
    ready_timeout: float = 20.0
    connect_timeout: int = 15
    strict_host_key_checking: str = "accept-new"


def build_ssh_local_forward_command(settings: ForwardSettings) -> list[str]:
    """Build a non-interactive `ssh -N -L` command.

    The host after -L is resolved on the cloud, so 127.0.0.1 is the cloud
    server's loopback listener. `--` keeps a hostile hostname from being
    parsed as an ssh option.
    """
    ssh_host = _reject_host_text(settings.ssh_host, "SSH host")
    ssh_user = _validate_ssh_user(settings.ssh_user)
    local_host = _loopback_target(settings.local_host, "Local bind address")
    forward_host = _loopback_target(settings.forward_host, "Forward host")
    local_port = _validate_port(settings.local_port, "Local port")
    remote_port = _validate_port(settings.remote_port, "Cloud port")
    ssh_port = _validate_port(settings.ssh_port, "SSH port")
    if settings.strict_host_key_checking not in STRICT_HOST_KEY_VALUES:
        raise ValueError("StrictHostKeyChecking must be yes, no, or accept-new.")
    if (
        isinstance(settings.connect_timeout, bool)
        or not isinstance(settings.connect_timeout, int)
        or settings.connect_timeout <= 0
    ):
        raise ValueError("SSH connect timeout must be a positive integer.")
    if not settings.ssh_executable or settings.ssh_executable.startswith("-"):
        raise ValueError("SSH program path is not valid.")

    forward = f"{local_host}:{local_port}:{forward_host}:{remote_port}"
    # Fail instead of prompting, and exit if the local port cannot be bound.
    command = [
        settings.ssh_executable,
        "-N",
        "-T",
        "-L",
        forward,
        "-p",
        str(ssh_port),
        "-o",
        "ExitOnForwardFailure=yes",
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        f"ConnectTimeout={settings.connect_timeout}",
        "-o",
        "ConnectionAttempts=1",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=3",
        "-o",
        f"StrictHostKeyChecking={settings.strict_host_key_checking}",
    ]
    if settings.identity:
        identity = settings.identity
        if identity.startswith("-") or "\n" in identity or "\r" in identity:
            raise ValueError("SSH identity path is not valid.")
        command.extend(["-i", identity, "-o", "IdentitiesOnly=yes"])
    command.extend(["--", f"{ssh_user}@{ssh_host}"])
    return command


def resolve_ssh_executable(explicit: str | None = None) -> str:
    """Find ssh.exe. Windows ships it as the OpenSSH Client optional feature."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise SshForwardError(f"SSH program not found: {explicit}")
        return str(path)
    found = shutil.which("ssh")
    if not found:
        raise SshForwardError(
            "ssh was not found on PATH. Install the OpenSSH Client, or pass --ssh-bin."
        )
    return found


def read_image(path: Path) -> bytes:
    """Read the image that will be base64-encoded into the request."""
    image_path = path.expanduser()
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    data = image_path.read_bytes()
    if not data:
        raise ValueError(f"Image file is empty: {image_path}")
    if len(data) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(data)} bytes. The limit is {MAX_IMAGE_BYTES} bytes."
        )
    return data


def build_infer_payload(
    image_bytes: bytes,
    filename: str,
    request_type: str,
    request_id: str,
) -> dict:
    """JSON body the cloud server decodes at POST /v1/infer."""
    if request_type not in REQUEST_TYPES:
        raise ValueError(f"Request type must be {TYPE_HELP}.")
    if not image_bytes:
        raise ValueError("Image is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(
            f"Image is {len(image_bytes)} bytes. The limit is {MAX_IMAGE_BYTES} bytes."
        )
    return {
        "type": request_type,
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "filename": clean_filename(filename),
        "request_id": request_id,
    }


def request_json(
    method: str,
    url: str,
    payload: dict | None = None,
    timeout: float = 30,
) -> tuple[int, dict]:
    """HTTP helper aimed at the local end of the forward."""
    data = None
    headers = {"Accept": "application/json", "User-Agent": "FaaS-Edge"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json; charset=utf-8"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.status
            raw = response.read()
    except urllib.error.HTTPError as exc:
        status = exc.code
        raw = exc.read()
    except urllib.error.URLError as exc:
        raise CloudResponseError(
            f"Could not reach the cloud server through the SSH local forward: {exc.reason}"
        ) from exc
    except (TimeoutError, socket.timeout) as exc:
        raise CloudResponseError("The cloud server did not answer before the timeout.") from exc

    try:
        text = raw.decode("utf-8")
        body = json.loads(text) if text else {}
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CloudResponseError(
            f"The cloud server returned a non-JSON response (HTTP {status})."
        ) from exc
    if not isinstance(body, dict):
        raise CloudResponseError(
            f"The cloud server returned HTTP {status} with a JSON value that is not an object."
        )
    return status, body


def tcp_open(host: str, port: int, timeout: float = 0.3) -> bool:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except OSError:
        return False
    finally:
        sock.close()


def wait_for_cloud(base_url: str, deadline: float) -> dict:
    """Poll /health until the forward reaches the cloud server, or time out.

    Connection failures are retried. An HTTP answer from something that is
    not the FaaS cloud server fails immediately.
    """
    last_error = "no response"
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            status, body = request_json(
                "GET",
                base_url + HEALTH_PATH,
                timeout=min(5.0, remaining),
            )
        except CloudResponseError as exc:
            last_error = str(exc)
            time.sleep(0.2)
            continue
        if status == 200 and body.get("ok") is True and body.get("service") == SERVICE_NAME:
            return body
        raise CloudResponseError(
            "The local forward is open, but it did not reach the FaaS cloud server. "
            f"HTTP {status}: {body}"
        )
    raise CloudResponseError(
        "SSH local forward is open, but the cloud server did not answer /health. "
        "Start src/Cloud/server.py on the cloud host and confirm --port matches "
        f"--remote-port. Last error: {last_error}"
    )


def deliver_payload(
    base_url: str,
    payload: dict,
    task: str,
    infer_timeout: float,
    deadline: float,
) -> dict:
    """Health-check the forward, then POST the image and request type."""
    health = wait_for_cloud(base_url, deadline)
    offered = health.get("request_types")
    if isinstance(offered, list) and task not in offered:
        offered_text = ", ".join(str(item) for item in offered)
        raise CloudResponseError(
            f"The cloud server does not accept request type {task!r}. It offers: {offered_text}."
        )
    _status, body = request_json(
        "POST",
        base_url + INFER_PATH,
        payload,
        timeout=infer_timeout,
    )
    return body


class SshLocalForward:
    """Run `ssh -L` and close it when the request is finished."""

    def __init__(self, settings: ForwardSettings, deadline: float):
        self.settings = settings
        self.deadline = deadline
        self.command = build_ssh_local_forward_command(settings)
        self._proc: subprocess.Popen | None = None
        self._stderr_file = None

    @property
    def base_url(self) -> str:
        return f"http://{self.settings.local_host}:{self.settings.local_port}"

    def __enter__(self):
        try:
            self.start()
            self.wait_until_listening()
        except Exception:
            self.close()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def start(self):
        if self._proc is not None:
            raise SshForwardError("SSH local forward is already running.")
        if tcp_open(self.settings.local_host, self.settings.local_port):
            raise SshForwardError(
                f"Local port {self.settings.local_port} is already in use. "
                "Choose another --local-port, or stop the forward that is already open."
            )
        logger.info("Opening SSH local forward: %s", subprocess.list2cmdline(self.command))
        creationflags = 0
        if sys.platform == "win32":
            creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        # A stderr pipe read from another thread deadlocks Popen.poll on Windows.
        # A temp file keeps ssh's error text without that pipe.
        self._stderr_file = tempfile.TemporaryFile()
        self._proc = subprocess.Popen(
            self.command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_file,
            creationflags=creationflags,
        )

    def wait_until_listening(self):
        """Wait until ssh accepts TCP on the local forward, or exits."""
        while time.monotonic() < self.deadline:
            code = self._proc.poll() if self._proc is not None else 1
            if code is not None:
                raise SshForwardError(self._failure_text(code))
            if tcp_open(self.settings.local_host, self.settings.local_port):
                logger.info("SSH local forward is listening on %s", self.base_url)
                return
            time.sleep(0.1)
        raise SshForwardError(self._failure_text(None))

    def close(self):
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("ssh process %s did not exit after kill.", proc.pid)
        if self._stderr_file is not None:
            try:
                self._stderr_file.close()
            except OSError:
                pass
            self._stderr_file = None

    def _stderr_text(self) -> str:
        handle = self._stderr_file
        if handle is None:
            return ""
        try:
            handle.flush()
            handle.seek(0)
            data = handle.read()
        except OSError:
            return ""
        if isinstance(data, bytes):
            text = data.decode("utf-8", errors="replace")
        else:
            text = str(data)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return "\n".join(lines[-15:])

    def _failure_text(self, code: int | None) -> str:
        detail = self._stderr_text() or "ssh produced no error text."
        command = subprocess.list2cmdline(self.command)
        if code is None:
            return (
                "Timed out waiting for the SSH local forward to open "
                f"{self.settings.local_host}:{self.settings.local_port}.\n"
                f"Command: {command}\n{detail}"
            )
        return (
            f"ssh exited with status {code} before the local forward was ready.\n"
            f"Command: {command}\n{detail}"
        )


def deliver_through_forward(
    settings: ForwardSettings,
    payload: dict,
    task: str,
    infer_timeout: float,
) -> dict:
    """Open `ssh -L`, post the request, and always tear the forward down."""
    if settings.ready_timeout <= 0:
        raise ValueError("Tunnel timeout must be positive.")
    deadline = time.monotonic() + settings.ready_timeout
    with SshLocalForward(settings, deadline) as tunnel:
        return deliver_payload(tunnel.base_url, payload, task, infer_timeout, deadline)


def configure_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Send an image and a request type (face, ocr, or object detection) "
            "to the cloud inference server through an SSH local forward. "
            "The cloud script must already be listening on the cloud loopback."
        )
    )
    parser.add_argument("--ssh-host", help="Cloud SSH host name or IPv4 address.")
    parser.add_argument("--ssh-user", help="SSH user on the cloud host.")
    parser.add_argument("--ssh-port", type=_cli_port, default=22, help="SSH server port.")
    parser.add_argument(
        "--identity",
        help="Private key for public-key authentication. ssh-agent is used when omitted.",
    )
    parser.add_argument(
        "--ssh-bin",
        help="Path to ssh. The default is the ssh program on PATH.",
    )
    parser.add_argument(
        "--local-port",
        type=_cli_port,
        default=DEFAULT_LOCAL_PORT,
        help="Port opened on this edge machine. The request is sent here.",
    )
    parser.add_argument(
        "--remote-port",
        type=_cli_port,
        default=DEFAULT_CLOUD_PORT,
        help="Loopback port of src/Cloud/server.py on the cloud host.",
    )
    parser.add_argument(
        "--forward-host",
        default=DEFAULT_CLOUD_HOST,
        help="Address of the cloud server as seen by the SSH server. Keep 127.0.0.1.",
    )
    parser.add_argument("--image", required=True, help="Image file to send.")
    parser.add_argument(
        "--type",
        required=True,
        dest="request_type",
        help=f"Request type tag: {TYPE_HELP}.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Seconds to wait for inference after the forward is up.",
    )
    parser.add_argument(
        "--tunnel-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for ssh to open the local forward and for /health.",
    )
    parser.add_argument(
        "--strict-host-key-checking",
        choices=STRICT_HOST_KEY_VALUES,
        default="accept-new",
        help="Passed to ssh. accept-new stores the host key on the first connection.",
    )
    parser.add_argument(
        "--connect-timeout",
        type=int,
        default=15,
        help="Seconds ssh waits for the TCP connection to the cloud SSH server.",
    )
    parser.add_argument(
        "--no-tunnel",
        action="store_true",
        help=(
            "Do not start ssh. POST to http://127.0.0.1:<local-port> instead. "
            "Use this when a local forward is already open, or when the cloud "
            "server is running on this machine."
        ),
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


def run(args) -> int:
    task = canonical_request_type(args.request_type)
    if task is None:
        logger.error("Request type must be %s. Got %r.", TYPE_HELP, args.request_type)
        return 2
    if args.timeout <= 0 or args.tunnel_timeout <= 0 or args.connect_timeout <= 0:
        logger.error("Timeouts must be positive.")
        return 2

    image_path = Path(args.image).expanduser()
    image_bytes = read_image(image_path)
    request_id = uuid.uuid4().hex[:12]
    payload = build_infer_payload(image_bytes, image_path.name, task, request_id)
    logger.info(
        "Prepared %s request %s from %s (%d bytes).",
        task,
        request_id,
        payload["filename"],
        len(image_bytes),
    )

    if args.no_tunnel:
        base_url = f"http://{DEFAULT_CLOUD_HOST}:{args.local_port}"
        logger.info("Sending without a new SSH session to %s", base_url)
        body = deliver_payload(
            base_url,
            payload,
            task,
            args.timeout,
            time.monotonic() + args.tunnel_timeout,
        )
    else:
        if not args.ssh_host or not args.ssh_user:
            logger.error("SSH local forwarding needs --ssh-host and --ssh-user.")
            return 2
        identity = None
        if args.identity:
            identity_path = Path(args.identity).expanduser()
            if not identity_path.is_file():
                logger.error("SSH identity file not found: %s", identity_path)
                return 2
            identity = str(identity_path)
        settings = ForwardSettings(
            ssh_host=args.ssh_host,
            ssh_user=args.ssh_user,
            ssh_executable=resolve_ssh_executable(args.ssh_bin),
            ssh_port=args.ssh_port,
            identity=identity,
            local_port=args.local_port,
            remote_port=args.remote_port,
            forward_host=args.forward_host,
            ready_timeout=args.tunnel_timeout,
            connect_timeout=args.connect_timeout,
            strict_host_key_checking=args.strict_host_key_checking,
        )
        body = deliver_through_forward(settings, payload, task, args.timeout)

    print(json.dumps(body, indent=2), flush=True)
    if body.get("ok") is True:
        return 0
    return 4


def main(argv=None) -> int:
    configure_logging()
    args = parse_args(argv)
    try:
        return run(args)
    except CloudResponseError as exc:
        logger.error("%s", exc)
        return 4
    except SshForwardError as exc:
        logger.error("%s", exc)
        return 3
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        logger.warning("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

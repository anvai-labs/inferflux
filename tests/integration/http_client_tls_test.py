"""Real TLS identity checks for buffered and streaming HTTP client paths."""
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest

PROBE = sys.argv.pop(1)


class HttpClientTlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="inferflux-tls-")
        cls.root = Path(cls.directory.name)
        # Disposable test CA and server keys remain in this temporary directory.
        def openssl(*args):
            subprocess.run(["openssl", *args], cwd=cls.root, check=True,
                           capture_output=True, timeout=20)
        openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-subj", "/CN=InferFlux Test CA", "-keyout", "ca.key", "-out", "ca.pem")
        openssl("req", "-new", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=unused",
                "-keyout", "server.key", "-out", "server.csr")
        for name, san in (("dns", "DNS:localhost"), ("ip", "IP:127.0.0.1"),
                          ("wrong", "DNS:wrong.invalid,IP:127.0.0.2")):
            (cls.root / "extensions").write_text(
                "subjectAltName=" + san + "\nbasicConstraints=CA:FALSE\n"
                "extendedKeyUsage=serverAuth\n")
            openssl("x509", "-req", "-in", "server.csr", "-CA", "ca.pem",
                    "-CAkey", "ca.key", "-CAcreateserial", "-days", "1",
                    "-extfile", "extensions", "-out", name + ".pem")

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def exercise(self, mode, certificate, host, trusted):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(self.root / (certificate + ".pem"), self.root / "server.key")
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            listener.settimeout(5)
            errors = []
            def serve():
                try:
                    conn, _ = listener.accept()
                    with conn:
                        conn.settimeout(5)
                        with context.wrap_socket(conn, server_side=True) as stream:
                            stream.recv(4096)
                            stream.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n"
                                           b"Connection: close\r\n\r\nOK")
                except ssl.SSLError:
                    pass  # Expected when the client rejects certificate identity/trust.
                except Exception as error:
                    errors.append(error)
            thread = threading.Thread(target=serve)
            thread.start()
            env = os.environ.copy()
            env["SSL_CERT_FILE"] = str(self.root / ("ca.pem" if trusted else "wrong.pem"))
            env["SSL_CERT_DIR"] = str(self.root / "empty-trust-dir")
            try:
                result = subprocess.run(
                    [PROBE, mode, f"https://{host}:{listener.getsockname()[1]}/"],
                    env=env, capture_output=True, text=True, timeout=10)
            finally:
                thread.join(timeout=6)
            self.assertFalse(thread.is_alive(), "TLS fixture failed to stop")
            self.assertEqual(errors, [])
            return result

    def test_identity_and_trust_on_both_request_paths(self):
        for mode in ("buffered", "raw"):
            for cert, host, trusted, accepted in (
                ("dns", "localhost", True, True),
                ("ip", "127.0.0.1", True, True),
                ("wrong", "localhost", True, False),
                ("wrong", "127.0.0.1", True, False),
                ("dns", "127.0.0.1", True, False),
                ("dns", "localhost", False, False),
            ):
                with self.subTest(mode=mode, cert=cert, host=host, trusted=trusted):
                    result = self.exercise(mode, cert, host, trusted)
                    if accepted:
                        self.assertEqual(result.returncode, 0, result.stderr)
                    else:
                        self.assertEqual(result.returncode, 1, result.stderr)
                        self.assertIn("TLS", result.stderr)


if __name__ == "__main__":
    unittest.main()

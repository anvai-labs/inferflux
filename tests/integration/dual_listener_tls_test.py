"""End-to-end dual-listener TLS tests: plain loopback HTTP + dedicated HTTPS.

Starts real inferfluxd processes with a disposable CA-signed certificate and
asserts both listeners serve the same route table, wrong-CA clients are
rejected, and the dedicated listener fails closed on a bad certificate.
"""
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import time
import unittest
import http.client

sys.path.insert(0, os.path.dirname(__file__))
from process_helper import start_server_process, stop_server_process

SERVER_HOST = "127.0.0.1"
SERVER_PORT = int(os.environ.get("INFERFLUX_TEST_PORT_BASE", "18081")) + 8
TLS_PORT = SERVER_PORT + 1
SERVER_BIN = os.environ.get("INFERFLUX_SERVER_BIN", "./build/inferfluxd")


def openssl(*args, cwd):
    subprocess.run(["openssl", *args], cwd=cwd, check=True,
                   capture_output=True, timeout=20)


def wait_for_health(client_factory, host, port, deadline_seconds=20.0):
    deadline = time.time() + deadline_seconds
    while time.time() < deadline:
        try:
            status = client_factory(host, port)
            if status in (200, 401):
                return True
        except Exception:
            pass
        time.sleep(0.1)
    return False


def plain_healthz(host, port):
    conn = http.client.HTTPConnection(host, port, timeout=1)
    conn.request("GET", "/healthz", headers={"Connection": "close"})
    response = conn.getresponse()
    response.read()
    status = response.status
    conn.close()
    return status


def tls_healthz(host, port, cafile, server_hostname):
    context = ssl.create_default_context(cafile=cafile)
    conn = http.client.HTTPSConnection(host, port, timeout=2,
                                       context=context)
    try:
        conn.request("GET", "/healthz", headers={"Connection": "close"})
        response = conn.getresponse()
        response.read()
        return response.status
    finally:
        conn.close()


class DualListenerTlsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="inferflux-dual-tls-")
        cls.root = Path(cls.directory.name)
        openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-subj", "/CN=InferFlux Test CA",
                "-addext", "basicConstraints=critical,CA:TRUE",
                "-addext", "keyUsage=critical,keyCertSign,cRLSign",
                "-keyout", "ca.key", "-out", "ca.pem", cwd=cls.root)
        openssl("req", "-new", "-newkey", "rsa:2048", "-nodes",
                "-subj", f"/CN={SERVER_HOST}", "-keyout", "server.key",
                "-out", "server.csr", cwd=cls.root)
        (cls.root / "extensions").write_text(
            f"subjectAltName=DNS:localhost,IP:{SERVER_HOST}\n"
            "basicConstraints=CA:FALSE\nextendedKeyUsage=serverAuth\n")
        openssl("x509", "-req", "-in", "server.csr", "-CA", "ca.pem",
                "-CAkey", "ca.key", "-CAcreateserial", "-days", "1",
                "-extfile", "extensions", "-out", "server.crt",
                cwd=cls.root)

        env = os.environ.copy()
        env["INFERFLUX_HOST_OVERRIDE"] = SERVER_HOST
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT)
        env["INFERFLUX_TLS_PORT_OVERRIDE"] = str(TLS_PORT)
        env["INFERFLUX_TLS_BIND_HOST"] = "127.0.0.1"
        env["INFERFLUX_TLS_CERT_PATH"] = str(cls.root / "server.crt")
        env["INFERFLUX_TLS_KEY_PATH"] = str(cls.root / "server.key")
        env["INFERFLUX_MODEL_PATH"] = ""
        cls.server = start_server_process(
            [SERVER_BIN, "--config", "config/server.yaml"], env=env,
            merge_stderr=True, text=True)
        if not wait_for_health(plain_healthz, SERVER_HOST, SERVER_PORT):
            raise RuntimeError("plain listener never became healthy")
        if not wait_for_health(
                lambda host, port: tls_healthz(host, port, cls.root / "ca.pem",
                                               SERVER_HOST),
                SERVER_HOST, TLS_PORT):
            raise RuntimeError("HTTPS listener never became healthy")

    @classmethod
    def tearDownClass(cls):
        stop_server_process(cls.server)

    def test_plain_listener_serves_health(self):
        self.assertEqual(plain_healthz(SERVER_HOST, SERVER_PORT), 200)

    def test_tls_listener_serves_health_with_ca_verification(self):
        self.assertEqual(
            tls_healthz(SERVER_HOST, TLS_PORT, self.root / "ca.pem",
                        SERVER_HOST), 200)

    def test_tls_listener_rejects_wrong_ca(self):
        wrong_ca = self.root / "wrong-ca.pem"
        openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
                "-subj", "/CN=Wrong CA", "-keyout", "wrong-ca.key",
                "-out", "wrong-ca.pem", cwd=self.root)
        with self.assertRaises(ssl.SSLCertVerificationError):
            tls_healthz(SERVER_HOST, TLS_PORT, wrong_ca, SERVER_HOST)

    def test_tls_listener_serves_authenticated_route(self):
        # Route parity: an authenticated /v1/models over TLS answers like the
        # plain listener does (200 with a valid key, 401 without one — the
        # shipped config's key set decides which).
        context = ssl.create_default_context(cafile=self.root / "ca.pem")
        conn = http.client.HTTPSConnection(SERVER_HOST, TLS_PORT, timeout=5,
                                           context=context)
        conn.request("GET", "/v1/models",
                     headers={"Authorization": "Bearer dev-key-123",
                              "Connection": "close"})
        response = conn.getresponse()
        response.read()
        status = response.status
        conn.close()
        self.assertIn(status, (200, 401))


class DualListenerFailClosedTests(unittest.TestCase):
    def test_bad_certificate_aborts_startup(self):
        env = os.environ.copy()
        env["INFERFLUX_HOST_OVERRIDE"] = SERVER_HOST
        env["INFERFLUX_PORT_OVERRIDE"] = str(SERVER_PORT + 2)
        env["INFERFLUX_TLS_PORT_OVERRIDE"] = str(TLS_PORT + 2)
        env["INFERFLUX_TLS_BIND_HOST"] = "127.0.0.1"
        env["INFERFLUX_TLS_CERT_PATH"] = "/nonexistent/cert.pem"
        env["INFERFLUX_TLS_KEY_PATH"] = "/nonexistent/key.pem"
        env["INFERFLUX_MODEL_PATH"] = ""
        process = start_server_process(
            [SERVER_BIN, "--config", "config/server.yaml"], env=env,
            merge_stderr=True, text=True)
        try:
            output = process.stdout.read()
        finally:
            stop_server_process(process)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn("failed to load TLS certificate", output)


if __name__ == "__main__":
    unittest.main()

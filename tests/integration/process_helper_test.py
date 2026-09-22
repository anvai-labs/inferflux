#!/usr/bin/env python3
"""An occupied integration port must never be treated as the owned server."""

import socket
import unittest
from unittest import mock

from process_helper import start_server_process


class ProcessHelperTests(unittest.TestCase):
    def test_occupied_port_rejects_before_starting_or_sending_application_data(self):
        for host in ("127.0.0.1", "0.0.0.0"):
            for reuse in (False, True):
                with self.subTest(host=host, reuse=reuse), socket.socket() as occupied:
                    occupied.setsockopt(
                        socket.SOL_SOCKET, socket.SO_REUSEADDR, int(reuse)
                    )
                    occupied.bind((host, 0))
                    occupied.listen()
                    env = {
                        "INFERFLUX_HOST_OVERRIDE": "127.0.0.1",
                        "INFERFLUX_PORT_OVERRIDE": str(occupied.getsockname()[1]),
                    }
                    with mock.patch("process_helper.subprocess.Popen") as spawn:
                        with self.assertRaisesRegex(RuntimeError, "already in use"):
                            start_server_process(["unused-test-binary"], env=env)
                        spawn.assert_not_called()
                    occupied.settimeout(1)
                    connection, _ = occupied.accept()
                    with connection:
                        connection.settimeout(1)
                        self.assertEqual(connection.recv(1), b"")


if __name__ == "__main__":
    unittest.main()

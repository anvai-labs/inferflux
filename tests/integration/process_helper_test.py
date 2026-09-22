#!/usr/bin/env python3
"""An occupied integration port must never be treated as the owned server."""

import socket
import unittest
from unittest import mock

from process_helper import start_server_process


class ProcessHelperTests(unittest.TestCase):
    def test_occupied_port_rejects_before_starting_or_contacting_a_server(self):
        with socket.socket() as occupied:
            occupied.bind(("127.0.0.1", 0))
            occupied.listen()
            env = {
                "INFERFLUX_HOST_OVERRIDE": "127.0.0.1",
                "INFERFLUX_PORT_OVERRIDE": str(occupied.getsockname()[1]),
            }
            with mock.patch("process_helper.subprocess.Popen") as spawn:
                with self.assertRaisesRegex(RuntimeError, "already in use"):
                    start_server_process(["unused-test-binary"], env=env)
                spawn.assert_not_called()
            occupied.setblocking(False)
            with self.assertRaises(BlockingIOError):
                occupied.accept()


if __name__ == "__main__":
    unittest.main()

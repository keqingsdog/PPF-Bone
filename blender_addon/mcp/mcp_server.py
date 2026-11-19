"""Pure Socket-based MCP Server for Blender integration."""

import contextlib
import socket
import threading
import time

from http.server import ThreadingHTTPServer

from ..models.defaults import DEFAULT_MCP_PORT
from .http_handler import MCPRequestHandler
from .server_utils import is_port_available, wait_for_port_release
from .sessions import clear_all_sessions
from .task_system import stop_task_owner


class _OwnedHTTPServer(ThreadingHTTPServer):
    """One listener lifetime, including sockets accepted before its shutdown."""

    def __init__(self, address, handler, shutdown_event):
        self._mcp_shutdown_event = shutdown_event
        self._requests = set()
        self._request_condition = threading.Condition()
        super().__init__(address, handler)

    def process_request(self, request, client_address):
        with self._request_condition:
            if self._mcp_shutdown_event.is_set():
                self.shutdown_request(request)
                return
            self._requests.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            with self._request_condition:
                self._requests.discard(request)
                self._request_condition.notify_all()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self._request_condition:
                self._requests.discard(request)
                self._request_condition.notify_all()

    def stop_requests(self):
        # Use the task lock so shutdown has an unambiguous order relative to
        # admission and claim. Never reset this event on listener restart.
        stop_task_owner(self._mcp_shutdown_event)
        with self._request_condition:
            requests = tuple(self._requests)
        for request in requests:
            with contextlib.suppress(OSError):
                request.shutdown(socket.SHUT_RDWR)

    def wait_requests(self, timeout=2.0):
        with self._request_condition:
            return self._request_condition.wait_for(
                lambda: not self._requests, timeout=timeout
            )


class BlenderMCPServer:
    """Pure socket-based MCP Server for exposing Blender functionality via HTTP."""

    def __init__(self):
        self.server = None
        self.thread = None
        self.running = False
        self.port = DEFAULT_MCP_PORT
        self.shutdown_event = threading.Event()
        self._bind_ready = threading.Event()
        self._bind_error = None

    def is_running(self):
        """Check if the MCP server is currently running."""
        return self.running and self.thread and self.thread.is_alive()

    def start(self, port=DEFAULT_MCP_PORT):
        """Start the MCP server in a separate thread with enhanced cleanup."""
        # First, try to stop any existing instance gracefully
        if self.is_running():
            print("MCP Server: Stopping existing instance...")
            self.stop()
            time.sleep(0.5)  # Wait for graceful shutdown

        self.port = port

        # Check if port is available
        if not is_port_available(port):
            print(f"MCP Server: Port {port} is not available")

            # Wait for port to become available with exponential backoff
            if wait_for_port_release(port, max_attempts=5, initial_delay=0.2):
                print(f"MCP Server: Port {port} is now available")
            else:
                # Try alternative ports
                print("MCP Server: Searching for alternative port...")
                original_port = port
                for alt_port in range(port + 1, port + 10):
                    if is_port_available(alt_port):
                        print(
                            f"MCP Server: Using alternative port {alt_port} (original {original_port} unavailable)"
                        )
                        port = alt_port
                        self.port = port
                        break
                else:
                    raise Exception(
                        f"MCP Server: Could not find available port in range {original_port}-{original_port + 9}"
                    )

        self.running = True
        # A new token cannot revive requests accepted by an older listener.
        shutdown_event = threading.Event()
        self.shutdown_event = shutdown_event

        # Reset bind signaling before launching the thread so start() observes
        # only this run's result (and never a stale set() from a prior start).
        self._bind_error = None
        bind_ready = threading.Event()
        self._bind_ready = bind_ready
        bind_error = None

        def run_server():
            nonlocal bind_error
            server = None
            try:
                server = _OwnedHTTPServer(
                    ("localhost", port), MCPRequestHandler, shutdown_event
                )
                # A timed-out start must not publish its late listener over
                # a newer generation, nor keep serving after start failed.
                if shutdown_event.is_set() or self.shutdown_event is not shutdown_event:
                    server.stop_requests()
                    server.server_close()
                    bind_ready.set()
                    return
                self.server = server

                # Configure server socket for better shutdown behavior
                server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                if hasattr(socket, "SO_REUSEPORT"):
                    with contextlib.suppress(OSError):
                        server.socket.setsockopt(
                            socket.SOL_SOCKET, socket.SO_REUSEPORT, 1
                        )

                print(f"MCP Server: Successfully started on localhost:{port}")

                # Use timeout to make shutdown more responsive
                server.timeout = 1.0

            except Exception as e:
                # Construction/bind failed: report the real error to start()
                # instead of letting it infer success from a port probe.
                bind_error = e
                if self.shutdown_event is shutdown_event:
                    self._bind_error = e
                print(f"MCP Server error: {e}")
                if server is not None:
                    server.stop_requests()
                    server.server_close()
                bind_ready.set()
                if self.shutdown_event is shutdown_event:
                    self.running = False
                shutdown_event.set()
                print("MCP Server: Server thread exiting")
                return

            # Server is fully constructed and bound; signal start() to return.
            bind_ready.set()

            try:
                while not shutdown_event.is_set():
                    try:
                        server.handle_request()
                    except OSError:
                        # Socket was closed, exit gracefully
                        break
            except Exception as e:
                print(f"MCP Server error: {e}")
            finally:
                server.stop_requests()
                server.server_close()
                if self.shutdown_event is shutdown_event:
                    self.running = False
                shutdown_event.set()
                print("MCP Server: Server thread exiting")

        self.thread = threading.Thread(target=run_server, daemon=True)
        self.thread.start()

        # Wait for the thread to actually finish binding (or fail), instead of
        # sleeping a fixed interval and re-probing the port (which races the
        # async bind and cannot tell our listener from a foreign one).
        ready = bind_ready.wait(timeout=5.0)
        if not ready:
            stop_task_owner(shutdown_event)
            if self.shutdown_event is shutdown_event:
                self.running = False
            raise Exception(
                f"MCP Server: Timed out waiting for bind on port {port}"
            )
        if bind_error is not None:
            self.running = False
            raise Exception(
                f"MCP Server: Failed to bind to port {port}: {bind_error}"
            )

        print(f"MCP Server: Confirmed running on port {port}")

    def stop(self):
        """Stop the MCP server with proper coordination and error handling."""
        if not self.running:
            print("MCP Server: Already stopped")
            return

        print("MCP Server: Initiating shutdown...")

        # Step 1: Signal shutdown to prevent new requests
        self.running = False
        if self.server is not None:
            self.server.stop_requests()
        else:
            stop_task_owner(self.shutdown_event)

        # Close any live Streamable HTTP sessions so their SSE threads exit.
        clear_all_sessions()

        # Step 2: Close server socket to stop accepting connections
        if self.server:
            try:
                print("MCP Server: Closing server socket...")
                self.server.server_close()
                if not self.server.wait_requests():
                    print("MCP Server: Warning - accepted requests still finishing")
                print("MCP Server: Server socket closed")
            except Exception as e:
                print(f"MCP Server: Error closing server socket: {e}")

        # Step 3: Wait for server thread to finish with increased timeout
        if self.thread and self.thread.is_alive():
            print("MCP Server: Waiting for server thread to finish...")
            self.thread.join(timeout=5.0)

            if self.thread.is_alive():
                print(
                    "MCP Server: Warning - server thread did not terminate within 5 seconds"
                )
            else:
                print("MCP Server: Server thread terminated successfully")

        # Step 4: Clean up references
        self.server = None
        self.thread = None

        # Step 5: Wait for port to be released
        if hasattr(self, "port"):
            print(f"MCP Server: Waiting for port {self.port} to be released...")

            if wait_for_port_release(self.port, max_attempts=8, initial_delay=0.1):
                print(f"MCP Server: Port {self.port} successfully released")
            else:
                print(f"MCP Server: Warning - port {self.port} may still be occupied")
                print(
                    "MCP Server: This is usually temporary and will resolve automatically"
                )

        print("MCP Server: Shutdown complete")


# Global MCP server instance
mcp_server = BlenderMCPServer()


def get_mcp_server():
    """Get the global MCP server instance."""
    return mcp_server


def is_mcp_running():
    """Check if MCP server is currently running."""
    return mcp_server.is_running()


def start_mcp_server(port=DEFAULT_MCP_PORT):
    """Start the MCP server."""
    return mcp_server.start(port)


def stop_mcp_server():
    """Stop the MCP server."""
    return mcp_server.stop()


def cleanup_mcp_server():
    """Cleanup MCP server during addon unregister."""
    try:
        if mcp_server.is_running():
            print("MCP Server: Cleaning up during addon unregister...")
            stop_mcp_server()
        else:
            # Even if we think it's not running, check the port
            if hasattr(mcp_server, "port") and not is_port_available(mcp_server.port):
                print(
                    f"MCP Server: Found process on port {mcp_server.port}, waiting for release..."
                )
                wait_for_port_release(
                    mcp_server.port, max_attempts=5, initial_delay=0.2
                )
    except Exception as e:
        print(f"MCP Server: Error during cleanup: {e}")
    try:
        from .task_system import clear_task_state
        clear_task_state()
    except Exception:
        pass

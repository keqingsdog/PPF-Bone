"""Public TCMD query contract. Empty means status; actions are never retried."""

import shlex

# Matches ppf-cts-server wire::tcmd_request_to_event. This is the wire command
# vocabulary, not a backend capability declaration.
COMMANDS = frozenset(("build", "cancel_build", "start", "resume", "terminate",
                      "save_and_quit", "delete", "twist_resume_contract"))
PARAMETERS = {"build": frozenset(("preserve_output",)),
              "resume": frozenset(("resume_from", "target_frame", "twist_resume_mode"))}


def validate_query(request):
    if not isinstance(request, dict):
        raise ValueError("Server query must be a dictionary.")
    if not request:
        return
    command = request.get("request")
    if not isinstance(command, str) or command not in COMMANDS:
        raise ValueError("Unsupported server request. Use an empty query for status.")
    allowed = PARAMETERS.get(command, frozenset()) | {"request"}
    if set(request) - allowed:
        raise ValueError("Unsupported parameters for server request: " + command)
    if "twist_resume_mode" in request:
        if request["twist_resume_mode"] != "continuous_from_current":
            raise ValueError("TWIST_RESUME_MODE_INVALID")
        try:
            frame = int(request.get("resume_from", 0))
        except (TypeError, ValueError):
            frame = 0
        if frame <= 0:
            raise ValueError("TWIST_RESUME_MODE_INVALID: positive checkpoint required")
    # The transport serializes TCMD values as whitespace-separated words.
    # Reject values that would introduce another option on the server.
    for value in request.values():
        if not isinstance(value, (str, int)) or isinstance(value, bool):
            raise ValueError("Server query values must be text or integers.")
        value = str(value)
        if not value or any(ch.isspace() for ch in value) or value.startswith("--"):
            raise ValueError("Server query values must be single non-option words.")


def parse_server_script(script):
    words = shlex.split(script)
    if len(words) % 2:
        raise ValueError("Expected --key value pairs; a value is missing.")
    request = {}
    for key, value in zip(words[::2], words[1::2]):
        if not key.startswith("--") or len(key) <= 2 or key[2:] in request:
            raise ValueError("Expected unique --key value pairs.")
        request[key[2:]] = value
    validate_query(request)
    return request

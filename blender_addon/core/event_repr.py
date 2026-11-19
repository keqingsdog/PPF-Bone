"""Bounded diagnostic prefixes for the four built-in binary payload events."""
from dataclasses import fields
from .events import (BuildPipelineRequested, UploadOnlyRequested,
                     SendDataRequested, ReceiveDataComplete)

_BINARY_EVENTS = (BuildPipelineRequested, UploadOnlyRequested,
                  SendDataRequested, ReceiveDataComplete)


def _bytes_prefix(value, limit):
    # CPython bytes repr chooses double quotes only when there is a single
    # quote and no double quote anywhere. Membership scans allocate no copy.
    quote = '"' if b"'" in value and b'"' not in value else "'"
    result = 'b' + quote
    for byte in value:
        if len(result) >= limit:
            return result[:limit]
        if byte == 92 or byte == ord(quote):
            result += '\\' + chr(byte)
        elif byte == 9:
            result += '\\t'
        elif byte == 10:
            result += '\\n'
        elif byte == 13:
            result += '\\r'
        elif 32 <= byte < 127:
            result += chr(byte)
        else:
            result += '\\x%02x' % byte
    return (result + quote)[:limit]


def event_repr_prefix(event, limit=200):
    if type(event) not in _BINARY_EVENTS:
        return repr(event)[:limit]
    result = (type(event).__qualname__ + '(')[:limit]
    first = True
    for field in fields(event):
        if not field.repr:
            continue
        label = ('' if first else ', ') + field.name + '='
        first = False
        result = (result + label)[:limit]
        value = getattr(event, field.name)
        # Still evaluate every ordinary field's repr, even after the prefix
        # is full: custom repr failures and side effects retain their order.
        part = (_bytes_prefix(value, max(0, limit - len(result)))
                if type(value) is bytes else repr(value))
        result = (result + part)[:limit]
    return (result + ')')[:limit]

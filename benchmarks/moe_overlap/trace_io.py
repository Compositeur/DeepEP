"""Read Chrome trace event arrays incrementally without retaining the full file."""

import json


class Reader:
    def __init__(self, stream):
        self.stream = stream
        self.buffer = ""
        self.position = 0
        self.eof = False
        self.bytes_read = 0
        self.decoder = json.JSONDecoder()

    def more(self):
        chunk = self.stream.read(65536)
        self.buffer = self.buffer[self.position :] + chunk
        self.position = 0
        self.eof = not chunk
        self.bytes_read += len(chunk)
        if len(self.buffer) > 16 * 1024 * 1024:
            raise ValueError("Single trace value exceeds 16 MiB")
        return bool(chunk)

    def peek(self):
        while True:
            while self.position < len(self.buffer) and self.buffer[self.position].isspace():
                self.position += 1
            if self.position < len(self.buffer):
                return self.buffer[self.position]
            if not self.more():
                raise ValueError("Truncated trace")

    def take(self, token):
        if self.peek() != token:
            raise ValueError("Expected trace token: " + token)
        self.position += 1

    def value(self):
        self.peek()
        while True:
            try:
                value, end = self.decoder.raw_decode(self.buffer, self.position)
            except json.JSONDecodeError:
                if not self.more():
                    raise ValueError("Truncated or invalid trace value") from None
                continue
            if end == len(self.buffer) and not self.eof:
                self.more()
                continue
            self.position = end
            return value


def iter_trace(path):
    with open(path, encoding="utf-8-sig") as stream:
        reader = Reader(stream)
        if reader.peek() == "{":
            reader.take("{")
            while True:
                if reader.bytes_read > 8 * 1024 * 1024:
                    raise ValueError("traceEvents header not found within 8 MiB")
                key = reader.value()
                if not isinstance(key, str):
                    raise ValueError("Trace object keys must be strings")
                reader.take(":")
                if key == "traceEvents":
                    break
                reader.value()
                reader.take(",")
        reader.take("[")
        if reader.peek() == "]":
            return
        while True:
            event = reader.value()
            if not isinstance(event, dict):
                raise ValueError("Trace events must be objects")
            yield event
            if reader.peek() == "]":
                return
            reader.take(",")

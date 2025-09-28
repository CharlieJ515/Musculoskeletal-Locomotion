from __future__ import annotations
from dataclasses import field
from typing import Iterator, Optional
from contextlib import contextmanager
import torch

class StreamObject:
    _stream: Optional[torch.cuda.Stream] = field(default=None)
    _event: Optional[torch.cuda.Event] = field(default=None)

    def wait(self, device: torch.device, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Ensure `stream` (or current stream on self.device) waits for the event recorded
        by the producer of this object.

        NOTE: This does NOT clear the event, so multiple consumers can safely wait.
        """
        if self._event is None:
            return

        if device.type == "cpu":
            self._event.synchronize()
            return

        s = stream or torch.cuda.current_stream(device)
        s.wait_event(self._event) # type: ignore[arg-type]
        return

    @contextmanager
    def stream(self, device: torch.device) -> Iterator[Optional[torch.cuda.Stream]]:
        """
        Acquire a CUDA stream for enqueueing work on the given device.

        - If there is an existing producer stream on the same device, reuse it.
        - Otherwise create a new stream and make it wait on the last recorded event
          to preserve execution order across devices.

        The returned context manager makes the chosen stream current for the
        duration of the block.  When the block exits, a new event is recorded
        on that stream to mark the completion point for downstream consumers.

        :param device: CUDA device on which to enqueue work.
        :type device: torch.device
        :return: A context manager yielding the CUDA stream.
        :rtype: ContextManager[torch.cuda.Stream]
        :raises ValueError: If ``device`` is not a CUDA device.
        """
        if device.type != "cuda":
            raise ValueError("stream() requires a CUDA device; CPU has no streams.")

        if self._stream is not None and self._stream.device == device:
            s = self._stream
        else:
            s = torch.cuda.Stream(device=device)
            if self._event is not None:
                s.wait_event(self._event) # type: ignore[arg-type]
            self._stream = s

        try:
            with torch.cuda.stream(s):
                yield s
        finally:
            ev = torch.cuda.Event()
            ev.record(s)
            self._event = ev

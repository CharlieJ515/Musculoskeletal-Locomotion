from __future__ import annotations
from dataclasses import field
from typing import Iterator, Optional
from contextlib import contextmanager
import warnings
import torch

# ┌─────────────────────────────────────────────────────────────┐
# │             sync() BEHAVIOR BY SOURCE / TARGET              │
# ├─────────────┬───────────┬───────────┬───────────────────────┤
# │ A._event    │ A.device  │ new_device│  Action in sync()     │
# ├─────────────┼───────────┼───────────┼───────────────────────┤
# │ None        │   CPU     │   CPU     │  no-op                │
# │ None        │   CPU     │   CUDA    │  no-op                │
# │ None        │   CUDA    │   CPU     │  no-op                │
# │ None        │   CUDA    │   CUDA    │  no-op                │
# │ Event       │   CPU     │   CPU     │  event.synchronize()  │
# │ Event       │   CPU     │   CUDA    │  target_stream.wait_event(event) │
# │ Event       │   CUDA    │   CPU     │  event.synchronize()  │
# │ Event       │   CUDA    │   CUDA    │  target_stream.wait_event(event) │
# └─────────────┴───────────┴───────────┴───────────────────────┘
#
# Legend:
# - A is the producing object (self)
# - new_device is the consumer’s device
# - event.synchronize() is a host-side block
# - wait_event is an async CUDA-side fence
# - sync() never clears the event (multi-consumer safe)


# ┌─────────────────────────────────────────────────────────────┐
# │          enqueue() BEHAVIOR BY SOURCE / TARGET              │
# ├─────────────┬───────────┬───────────┬───────────────────────┤
# │ A._event    │ A.device  │ new_device│  Action in enqueue()  │
# ├─────────────┼───────────┼───────────┼───────────────────────┤
# │ None        │   CPU     │   CPU     │  yield nullcontext()  │
# │ None        │   CPU     │   CUDA    │  create stream on new_device → yield  │
# │ None        │   CUDA    │   CPU     │  yield nullcontext()  │
# │ None        │   CUDA    │   CUDA    │  reuse existing stream if same device else create new → yield  │
# │ Event       │   CPU     │   CPU     │  event.synchronize() → yield nullcontext() │
# │ Event       │   CPU     │   CUDA    │  create stream on new_device, wait_event(event) → yield  │
# │ Event       │   CUDA    │   CPU     │  event.synchronize() → yield nullcontext() │
# │ Event       │   CUDA    │   CUDA    │  reuse existing stream if same device else create new, wait_event(event) → yield  │
# └─────────────┴───────────┴───────────┴───────────────────────┘
#
# Legend:
# - A is the producing object (self)
# - new_device is the device for the new work to enqueue
# - nullcontext() is a no-op CPU context but fences if event exists
# - wait_event(event) asynchronously chains CUDA streams
# - exit of enqueue() records a fresh event on the active CUDA stream
#   so the next consumer can chain correctly


class StreamObject:
    _stream: Optional[torch.cuda.Stream]
    _event: Optional[torch.cuda.Event]

    def set_stream(
        self,
        *,
        obj: Optional["StreamObject"] = None,
        stream: Optional[torch.cuda.Stream] = None,
        event: Optional[torch.cuda.Event] = None,
    ) -> None:
        """
        Set or share this object's internal producer pointers.

        Behavior:
          - If ``obj`` is provided, copy its stream/event pair (sharing the same fence).
            Any explicit ``stream`` or ``event`` arguments will be ignored with a warning.
          - Otherwise, both ``stream`` and ``event`` must be provided together.
            This enforces consistency so the object never ends up with only one pointer.

        Notes:
          - This method does not perform synchronization or device checks.
          - Typically, the event represents the true cross-device fence; the stream is
            mainly a reuse hint.

        :param obj: Source object to share pointers from.
        :type obj: Optional[StreamObject]
        :param stream: CUDA stream to set as the producer stream.
        :type stream: Optional[torch.cuda.Stream]
        :param event: CUDA event to set as the last completion fence.
        :type event: Optional[torch.cuda.Event]
        :raises ValueError: if only one of ``stream`` or ``event`` is provided.
        """
        if obj is not None:
            if stream is not None or event is not None:
                warnings.warn(
                    "set_stream(obj=...) ignores explicit 'stream'/'event' arguments.",
                    stacklevel=2,
                )
            self._stream = obj._stream
            self._event = obj._event
            return

        if (stream is None) or (event is None):
            raise ValueError(
                f"Expected both `stream` and `event` to be provided together, "
                f"got stream={stream}, event={event}"
            )

        self._stream = stream
        self._event = event

    def sync(self, *,obj:Optional[StreamObject]=None, device: Optional[torch.device]=None, stream: Optional[torch.cuda.Stream] = None) -> None:
        """
        Wait for the most recently recorded CUDA event (acts as a stack pointer).
        This does **not** clear the event, allowing multiple consumers to fence safely.

        :param device: Device whose execution should be fenced (CPU blocks, CUDA waits on stream).
        :type device: torch.device
        :param stream: CUDA stream to fence; if None, uses the current stream on ``device``.
        :type stream: Optional[torch.cuda.Stream]
        :return: None
        :rtype: None
        """
        if obj is None and device is None and stream is None:
            raise ValueError("sync(): provide at least one of obj/device/stream")

        if self._event is None:
            return

        if obj is not None:
            if obj._stream is None:
                device = torch.device("cpu")
                stream = None
            else:
                device = obj._stream.device
                stream = obj._stream

        stream = stream or torch.cuda.current_stream(device)
        device = device or stream.device

        if device.type == "cpu":
            self._event.synchronize()
            return

        stream.wait_event(self._event)  # type: ignore[arg-type]
        return

    @contextmanager
    def enqueue(self, device: torch.device) -> Iterator[Optional[torch.cuda.Stream]]:
        """
        Acquire a scope for enqueueing work on this object.

        - **CUDA target**: reuse the existing producer stream if it's on the same device;
          otherwise create a new stream and, if an event exists, make it wait on that
          event to preserve ordering. The context makes the stream current for the block
          and records a fresh event on exit.
        - **CPU target**: if an event exists, block the host via ``event.synchronize()``
          to avoid races, then yield a no-op scope (returns ``None``).

        :param device: Target device for new work. If omitted and a CUDA stream already
                       exists, that stream's device is reused; otherwise CPU is assumed.
        :type device: Optional[torch.device]
        :return: The CUDA stream used for enqueueing, or ``None`` for CPU.
        :rtype: Optional[torch.cuda.Stream]
        :raises ValueError: If a non-CUDA, non-CPU device is provided.  # unlikely but kept explicit
        """

        if device.type == "cpu":
            if self._event is not None:
                self._event.synchronize()
            try:
                yield None
            finally:
                self._stream = None
                self._event = None
            return

        # CUDA path: reuse or create stream on target device
        if self._stream is not None and self._stream.device == device:
            s = self._stream
        else:
            s = torch.cuda.Stream(device)
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

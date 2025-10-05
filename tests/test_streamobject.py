# import pytest
# import torch
# from utils.stream import StreamObject
#
# # Skip whole module if CUDA is missing
# pytestmark = pytest.mark.skipif(
#     not torch.cuda.is_available(),
#     reason="CUDA device required for StreamObject tests"
# )
#
# def dev_cpu() -> torch.device:
#     return torch.device("cpu")
#
#
# def dev_cuda(idx: int = 0) -> torch.device:
#     return torch.device("cuda", idx)
#
#
# @pytest.fixture(scope="function")
# def so() -> StreamObject:
#     return StreamObject()
#
#
# # --------------------------
# # Basic state
# # --------------------------
#
# def test_initial_state(so: StreamObject):
#     assert so._event is None
#     assert so._stream is None
#
#
# # --------------------------
# # sync(): non-consuming behavior
# # --------------------------
#
# def test_sync_no_event_cpu(so: StreamObject):
#     so.sync(dev_cpu())
#     assert so._event is None
#     assert so._stream is None
#
#
# def test_sync_no_event_cuda(so: StreamObject):
#     so.sync(dev_cuda())
#     assert so._event is None
#     assert so._stream is None
#
#
# def test_sync_with_event_cpu_is_non_consuming(so: StreamObject):
#     x = torch.zeros((), device=dev_cuda(), dtype=torch.int32)
#
#     with so.enqueue(dev_cuda()):
#         torch.cuda._sleep(int(5e6))
#         x.fill_(7)
#
#     so.sync(dev_cpu())
#     assert so._event is not None
#     assert x.item() == 7
#
#
# def test_sync_with_event_cuda_on_new_stream_is_non_consuming(so: StreamObject):
#     x = torch.zeros((), device=dev_cuda(), dtype=torch.int32)
#
#     with so.enqueue(dev_cuda()):
#         torch.cuda._sleep(int(5e6))
#         x.fill_(9)
#
#     s2 = torch.cuda.Stream(device=dev_cuda())
#     so.sync(dev_cuda(), stream=s2)
#
#     y = torch.empty((), device=dev_cuda(), dtype=torch.int32)
#     with torch.cuda.stream(s2):
#         y.copy_(x)
#
#     s2.synchronize()
#     assert int(y.item()) == 9
#     assert so._event is not None
#
#
# # --------------------------
# # enqueue(): CPU path
# # --------------------------
#
# def test_enqueue_cpu_no_event_yields_none(so: StreamObject):
#     with so.enqueue(dev_cpu()) as s:
#         assert s is None
#     assert so._event is None
#     assert so._stream is None
#
#
# def test_enqueue_cpu_with_event_fences_and_clears_event(so: StreamObject):
#     x = torch.zeros((), device=dev_cuda(), dtype=torch.int32)
#     with so.enqueue(dev_cuda()):
#         torch.cuda._sleep(int(5e6))
#         x.fill_(3)
#
#     assert so._event is not None
#
#     with so.enqueue(dev_cpu()) as s:
#         assert s is None
#         assert x.item() == 3
#
#     assert so._event is None
#     assert isinstance(so._stream, torch.cuda.Stream) or so._stream is None
#
#
# # --------------------------
# # enqueue(): CUDA path
# # --------------------------
#
# def test_enqueue_cuda_creates_stream_and_event(so: StreamObject):
#     assert so._stream is None and so._event is None
#     with so.enqueue(dev_cuda()) as s:
#         assert isinstance(s, torch.cuda.Stream)
#         assert so._stream is s
#     assert so._event is not None
#     assert so._stream is not None
#     assert so._stream.device == dev_cuda()
#
#
# def test_enqueue_cuda_reuses_same_stream(so: StreamObject):
#     with so.enqueue(dev_cuda()) as s1:
#         pass
#     with so.enqueue() as s2:
#         assert s2 is so._stream
#         assert s1 is s2
#
#
# def test_enqueue_cuda_waits_on_prior_event(so: StreamObject):
#     x = torch.zeros((), device=dev_cuda(), dtype=torch.int32)
#
#     with so.enqueue(dev_cuda()) as s1:
#         torch.cuda._sleep(int(5e6))
#         x.fill_(11)
#
#     with so.enqueue(dev_cuda()) as s2:
#         y = torch.empty((), device=dev_cuda(), dtype=torch.int32)
#         y.copy_(x)
#
#     torch.cuda.synchronize()
#     assert int(x.item()) == 11
#
#
# # --------------------------
# # enqueue(): default device inference
# # --------------------------
#
# def test_enqueue_default_without_prior_stream_is_cpu_nullcontext(so: StreamObject):
#     with so.enqueue() as s:
#         assert s is None
#     assert so._event is None
#     assert so._stream is None
#
#
# def test_enqueue_default_with_prior_stream_reuses_that_stream(so: StreamObject):
#     with so.enqueue(dev_cuda()) as s1:
#         pass
#     with so.enqueue() as s2:
#         assert s2 is s1
#         assert s2 is so._stream
#     assert so._event is not None
#
#
# # --------------------------
# # Multiple consumers
# # --------------------------
#
# def test_multiple_sync_calls_remain_non_consuming(so: StreamObject):
#     with so.enqueue(dev_cuda()):
#         pass
#     ev_ref = so._event
#     assert ev_ref is not None
#
#     so.sync(dev_cuda())
#     assert so._event is ev_ref
#     so.sync(dev_cpu())
#     assert so._event is ev_ref
#
#
# # --------------------------
# # Multi-GPU chaining logic (if available)
# # --------------------------
#
# def test_enqueue_multi_device_logic_path(so: StreamObject):
#     d0 = dev_cuda(0)
#     with so.enqueue(d0):
#         a = torch.zeros((), device=d0)
#         a.add_(1)
#
#     target_idx = 0 if torch.cuda.device_count() == 1 else 1
#     dN = dev_cuda(target_idx)
#
#     with so.enqueue(dN) as s_new:
#         b = torch.zeros((), device=dN)
#         b.add_(1)
#
#     torch.cuda.synchronize(device=dN)

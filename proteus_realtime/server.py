"""Single-GPU gRPC server; podexec-compatible inference and local administration."""
import base64
import concurrent.futures
import json
import logging
import os
import signal

import grpc

from .protocol import ADMIN_SERVICE, InferReply, InferRequest
from .runtime import GpuRuntime
from .serving import ProteusExecutor
from .service import BusyError, Executor

RPC_OPTIONS = [("grpc.max_receive_message_length", 32 * 1024 * 1024),
               ("grpc.max_send_message_length", 32 * 1024 * 1024)]


def create_server(executor, address, threads=16):
    server = grpc.server(concurrent.futures.ThreadPoolExecutor(max_workers=threads),
                         options=RPC_OPTIONS)

    def infer(request, context):
        try:
            future = executor.submit(request)
        except ValueError as error:
            return InferReply(rejected=True, reject_kind="bad_request", reason=str(error))
        except BusyError as error:
            return InferReply(rejected=True, reject_kind="saturation", reason=str(error))
        context.add_callback(future.cancel)
        try:
            timeout = context.time_remaining()
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            future.cancel()
            context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "Client RPC deadline exceeded")
        except concurrent.futures.CancelledError:
            context.abort(grpc.StatusCode.CANCELLED, "Request cancelled")
        except Exception as error:
            context.abort(grpc.StatusCode.INTERNAL, f"GPU inference failed: {error}")

    def admin(method):
        def call(payload, context):
            try:
                if method == "Status":
                    return {"ready": True, **executor.state()}
                if method == "Reset":
                    return executor.reset()
                if method == "Profile":
                    image = base64.b64decode(payload.get("image_base64", ""), validate=True)
                    if isinstance(executor, ProteusExecutor):
                        return executor.profile(image, payload.get("trials", 30), payload.get("models"),
                                                timeout_s=context.time_remaining())
                    return executor.profile(image, payload.get("trials", 30))
                if method == "Mode":
                    return executor.set_mode(payload.get("mode"))
                if method == "Evaluate":
                    return executor.evaluate(payload.get("manifest", "manifest.json"))
            except BusyError as error:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
            except (ValueError, TypeError) as error:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
            except Exception as error:
                logging.getLogger("proteus.bootstrap").exception("Admin RPC failed")
                context.abort(grpc.StatusCode.INTERNAL, str(error))
        return grpc.unary_unary_rpc_method_handler(
            call, request_deserializer=lambda data: json.loads(data.decode()),
            response_serializer=lambda value: json.dumps(value, allow_nan=False).encode())

    server.add_generic_rpc_handlers((
        grpc.method_handlers_generic_handler("podexec.Executor", {
            "Infer": grpc.unary_unary_rpc_method_handler(
                infer, request_deserializer=InferRequest.FromString,
                response_serializer=InferReply.SerializeToString)}),
        grpc.method_handlers_generic_handler(ADMIN_SERVICE, {
            name: admin(name) for name in ("Status", "Reset", "Profile", "Mode", "Evaluate")}),
    ))
    port = server.add_insecure_port(address)
    if not port:
        raise RuntimeError(f"Cannot bind gRPC server to {address}")
    return server, port


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [proteus-gpu] %(name)s %(message)s")
    early_drop = os.environ.get("EARLY_DEADLINE_DROP", "1")
    if early_drop not in {"0", "1"}:
        raise ValueError("EARLY_DEADLINE_DROP must be 0 or 1")
    backend = GpuRuntime()
    executor = ProteusExecutor(backend, max_queue=int(os.environ.get("MAX_QUEUE_SIZE", "256")),
        profile_path=os.path.join(os.environ.get("PROFILE_DIR", "/profiles"), "profiles.json"),
        slo=float(os.environ.get("SLO_MS", "200")),
        batches=tuple(int(b) for b in os.environ.get("BATCH_SIZES", "1,2,4,8").split(",")),
        interval=float(os.environ.get("ALLOCATION_INTERVAL", "1")),
        guard_ms=float(os.environ.get("DEADLINE_GUARD_MS", "2")),
        rpc_budget_ms=float(os.environ.get("RPC_BUDGET_MS", "0")),
        early_deadline_drop=early_drop == "1")
    server, _ = create_server(executor, "0.0.0.0:" + os.environ.get("WORKER_PORT", "9000"),
                              int(os.environ.get("RPC_THREADS", "16")))
    server.start()
    logging.info("READY %s", json.dumps(executor.state(), ensure_ascii=False))

    def stop(*_):
        server.stop(5)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.wait_for_termination()
    finally:
        executor.close()


if __name__ == "__main__":
    main()

"""Wire-compatible podexec.Executor messages without a protoc dependency.

Field numbers match the lab INFaaS podexec API. A private descriptor pool keeps
this module compatible with importing INFaaS's generated messages in one process.
The administrative RPC uses JSON and is deliberately a separate service.
"""
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory


def _field(message, name, number, kind, *, repeated=False, type_name=None):
    field = message.field.add(name=name, number=number, type=kind)
    field.label = 3 if repeated else 1
    if type_name:
        field.type_name = type_name


def _map(message, name, number, value_kind):
    entry_name = "".join(part.title() for part in name.split("_")) + "Entry"
    entry = message.nested_type.add(name=entry_name)
    entry.options.map_entry = True
    _field(entry, "key", 1, 9)
    _field(entry, "value", 2, value_kind)
    _field(message, name, number, 11, repeated=True,
           type_name=f".podexec.{message.name}.{entry_name}")


_file = descriptor_pb2.FileDescriptorProto(
    name="proteus_realtime_podexec.proto", package="podexec", syntax="proto3")
_request = _file.message_type.add(name="InferRequest")
for _name, _number, _kind in [("model", 1, 9), ("image", 2, 12),
                             ("slo", 3, 1), ("dispatch", 4, 9)]:
    _field(_request, _name, _number, _kind)
_box = _file.message_type.add(name="Box")
for _name, _number, _kind in [("x1", 1, 1), ("y1", 2, 1), ("x2", 3, 1),
                             ("y2", 4, 1), ("label", 5, 9), ("score", 6, 1)]:
    _field(_box, _name, _number, _kind)
_reply = _file.message_type.add(name="InferReply")
for _name, _number, _kind in [
        ("request_name", 1, 9), ("inference", 2, 9), ("latency", 3, 1),
        ("gpu", 4, 9), ("rejected", 5, 8), ("reason", 6, 9),
        ("img_width", 9, 5), ("img_height", 10, 5), ("reject_kind", 11, 9),
        ("predict_ms", 12, 1), ("worker_ms", 13, 1),
        ("arrive_epoch_ms", 15, 1), ("reply_epoch_ms", 16, 1), ("variant", 17, 9)]:
    _field(_reply, _name, _number, _kind)
_map(_reply, "estimates", 7, 1)
_field(_reply, "detections", 8, 11, repeated=True, type_name=".podexec.Box")
_map(_reply, "timings", 14, 1)
_pool = descriptor_pool.DescriptorPool()
_pool.Add(_file)
InferRequest = message_factory.GetMessageClass(_pool.FindMessageTypeByName("podexec.InferRequest"))
InferReply = message_factory.GetMessageClass(_pool.FindMessageTypeByName("podexec.InferReply"))
Box = message_factory.GetMessageClass(_pool.FindMessageTypeByName("podexec.Box"))

INFER_METHOD = "/podexec.Executor/Infer"
ADMIN_SERVICE = "proteus.bootstrap.Admin"

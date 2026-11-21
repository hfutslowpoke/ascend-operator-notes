#!/usr/bin/env python3
"""Check ACL device initialization and stream creation without launching an operator."""

import argparse
import ctypes as c


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", type=int, default=0)
    args = ap.parse_args()
    acl = c.CDLL("libascendcl.so")
    stream = c.c_void_p()
    for name, types, values in (
        ("aclInit", (c.c_char_p,), (None,)),
        ("aclrtSetDevice", (c.c_int,), (args.device,)),
        ("aclrtCreateStream", (c.POINTER(c.c_void_p),), (c.byref(stream),)),
        ("aclrtDestroyStream", (c.c_void_p,), (stream,)),
    ):
        operation = getattr(acl, name)
        operation.argtypes, operation.restype = types, c.c_int
        print(f"{name} starting", flush=True)
        result = operation(*values)
        print(f"{name}={result}", flush=True)
        if result:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

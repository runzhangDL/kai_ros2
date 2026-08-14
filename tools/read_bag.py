#!/usr/bin/env python3
"""read_bag.py -- read a rosbag2 sqlite3 bag with nothing but the standard library.

``rosbag2_py`` only exists where ROS is installed, which is the robot -- and the
analysis wants to happen on the machine with numpy and the simulator. A rosbag2
``.db3`` is a plain sqlite database whose blobs are CDR, so it can be read
anywhere.

Only the three message types this project records are decoded:
``sensor_msgs/JointState``, ``sensor_msgs/Imu`` and ``std_msgs/String``.

    python3 tools/read_bag.py path/to/bag_dir --list
    python3 tools/read_bag.py path/to/bag_dir --topic /humanoid/joint_states

CDR rules that matter here: a 4-byte encapsulation header comes first and is
NOT counted for alignment; every primitive is then aligned to its own size
relative to the start of the body; strings are a uint32 length including the
trailing NUL, followed by the bytes.
"""

import argparse
import glob
import os
import sqlite3
import struct
import sys


class Cdr:
    """Little-endian CDR reader. Offsets are relative to the body start."""

    def __init__(self, blob: bytes) -> None:
        if len(blob) < 4:
            raise ValueError("truncated CDR payload")
        # blob[0:2] is the encapsulation id; 0x0001 = little-endian CDR.
        self.little = blob[1] in (1, 3)
        self.buf = blob[4:]
        self.at = 0

    def _align(self, size: int) -> None:
        pad = (-self.at) % size
        self.at += pad

    def _take(self, fmt: str, size: int):
        self._align(size)
        end = self.at + size
        if end > len(self.buf):
            raise ValueError("CDR overrun")
        value = struct.unpack_from(("<" if self.little else ">") + fmt,
                                   self.buf, self.at)[0]
        self.at = end
        return value

    def u32(self):
        return self._take("I", 4)

    def i32(self):
        return self._take("i", 4)

    def f64(self):
        return self._take("d", 8)

    def string(self) -> str:
        n = self.u32()
        raw = self.buf[self.at:self.at + n]
        self.at += n
        return raw.split(b"\x00", 1)[0].decode("utf-8", "replace")

    def f64_array(self, n=None):
        count = self.u32() if n is None else n
        return [self.f64() for _ in range(count)]

    def string_array(self):
        return [self.string() for _ in range(self.u32())]

    def header(self):
        sec, nsec = self.i32(), self.u32()
        return sec + nsec * 1e-9, self.string()


def decode_joint_state(blob):
    c = Cdr(blob)
    stamp, _frame = c.header()
    names = c.string_array()
    position = c.f64_array()
    velocity = c.f64_array()
    effort = c.f64_array()
    return {"t": stamp, "name": names, "position": position,
            "velocity": velocity, "effort": effort}


def decode_imu(blob):
    c = Cdr(blob)
    stamp, _frame = c.header()
    quat = [c.f64() for _ in range(4)]           # x, y, z, w
    c.f64_array(9)
    gyro = [c.f64() for _ in range(3)]
    c.f64_array(9)
    accel = [c.f64() for _ in range(3)]
    c.f64_array(9)
    return {"t": stamp, "orientation": quat, "angular_velocity": gyro,
            "linear_acceleration": accel}


def decode_string(blob):
    return {"data": Cdr(blob).string()}


DECODERS = {
    "sensor_msgs/msg/JointState": decode_joint_state,
    "sensor_msgs/msg/Imu": decode_imu,
    "std_msgs/msg/String": decode_string,
}


def open_bag(path):
    if os.path.isdir(path):
        found = sorted(glob.glob(os.path.join(path, "*.db3")))
        if not found:
            raise SystemExit(f"no .db3 inside {path}")
        path = found[0]
    return sqlite3.connect(path)


def topics(conn):
    return {name: (tid, tname) for tid, name, tname
            in conn.execute("SELECT id, name, type FROM topics")}


def messages(conn, topic):
    table = topics(conn)
    if topic not in table:
        raise SystemExit(f"no topic {topic}; have {sorted(table)}")
    tid, tname = table[topic]
    decode = DECODERS.get(tname)
    if decode is None:
        raise SystemExit(f"no decoder for {tname}")
    rows = conn.execute(
        "SELECT timestamp, data FROM messages WHERE topic_id=? ORDER BY timestamp",
        (tid,))
    for stamp, blob in rows:
        out = decode(bytes(blob))
        out["bag_t"] = stamp * 1e-9
        yield out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bag")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--topic")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    conn = open_bag(args.bag)
    if args.list or not args.topic:
        for name, (tid, tname) in sorted(topics(conn).items()):
            n = conn.execute("SELECT COUNT(*) FROM messages WHERE topic_id=?",
                             (tid,)).fetchone()[0]
            print(f"{name:<34}{tname:<30}{n:>7} msgs")
        return 0
    for i, msg in enumerate(messages(conn, args.topic)):
        if i >= args.limit:
            break
        print(msg)
    return 0


if __name__ == "__main__":
    sys.exit(main())

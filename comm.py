from io import BytesIO 
import collections 
import socket 
import struct 

import numpy as np 

TYPES = {} 

def send_socket_message(sock, msg_type_byte, data=None): 
    sock.sendall(encode_message(msg_type_byte, data)) 

def recv_socket_message(sock): 
    size = sock.recv(4) 
    if len(size) != 4: 
        return None, None 
    msg_size = struct.unpack("<I", size)[0] 
    data = bytearray()
    while len(data) != msg_size: 
        try: 
            packet = sock.recv(msg_size - len(data))
            if not packet: 
                return None, None 
            else: 
                data.extend(packet)
        except socket.timeout: 
            pass 
    if len(data) != msg_size: 
        return None, None  
    msg = size + data
    return decode_message(msg) 

def encode_message(msg, data=None): 
    out = struct.pack("<B", msg) 
    out += encode(data) 
    return struct.pack("<I", len(out)) + out 

def decode_message(buf): 
    msg = struct.unpack("<B", buf[4:5])[0]
    return msg, decode(buf[5:])  

def register_type(obj_type, encode, decode): 
    TYPES[obj_type.__name__] = (encode, decode) 

def encode(data): 
    out = encode_buf(data, BytesIO())
    out.seek(0) 
    return out.read() 

def encode_buf(data, out): 
    name = type(data).__name__
    encode_str(name, out)
    TYPES[name][0](data, out) 
    return out 

def decode(buf): 
    i = BytesIO() 
    i.write(buf) 
    i.seek(0) 
    return decode_buf(i) 

def decode_buf(buf): 
    name = decode_str(buf) 
    return TYPES[name][1](buf) 

def encode_str(x, o): 
    s = bytes(x, 'utf-8')
    o.write(struct.pack("<I{}s".format(len(s)), len(s), s))

def decode_str(i): 
    length = struct.unpack("<I", i.read(4))[0]
    return struct.unpack("<{}s".format(length), i.read(length))[0].decode('utf-8')

def encode_int(x, o): 
    o.write(struct.pack("<I", x)) 

def decode_int(i): 
    return struct.unpack("<I", i.read(4))[0] 

def encode_float(x, o): 
    o.write(struct.pack("<d", x)) 

def decode_float(i): 
    return struct.unpack("<d", i.read(8))[0] 

def encode_NoneType(x, o): 
    pass 

def decode_NoneType(i): 
    return None 

def encode_list(x, o): 
    o.write(struct.pack("<I", len(x))) 
    for elem in x: 
        encode_buf(elem, o) 

def decode_list(i): 
    length = struct.unpack("<I", i.read(4))[0]
    out = [] 
    for _ in range(length): 
        out.append(decode_buf(i)) 
    return out 

def encode_dict(x, o): 
    o.write(struct.pack("<I", len(x))) 
    for key in x: 
        encode_buf(key, o) 
        encode_buf(x[key], o) 

def decode_dict(i): 
    length = struct.unpack("<I", i.read(4))[0]
    out = {} 
    for _ in range(length): 
        key = decode_buf(i) 
        value = decode_buf(i) 
        out[key] = value 
    return out 

def encode_tuple(x, o): 
    o.write(struct.pack("<I", len(x))) 
    for elem in x: 
        encode_buf(elem, o) 

def decode_tuple(i): 
    length = struct.unpack("<I", i.read(4))[0]
    out = [] 
    for _ in range(length): 
        out.append(decode_buf(i)) 
    return out 

def encode_bool(x, o): 
    o.write(struct.pack("<?", x))

def decode_bool(i): 
    return struct.unpack("<?", i.read(1)) 

def encode_ndarray(x: np.ndarray, o): 
    np.save(o, x)

def decode_ndarray(i): 
    return np.load(i) 

def register_class(cls): 
    print("Registering {}".format(cls))

    def encode_class(x, o): 
        out = {} 
        for a in dir(x): 
            v = getattr(x, a) 
            if not a.startswith("__") and not isinstance(v, collections.Callable): 
                out[a] = v
        print("Encoding {}".format(out))
        encode_dict(out, o) 

    def decode_class(i): 
        out = decode_dict(i) 
        print("Decoding {}".format(out))
        c = cls.__new__(cls, None, None) 
        for key in out: 
            setattr(c, key, out[key]) 
        return c 

    register_type(cls, encode_class, decode_class)

register_type(type(None), encode_NoneType, decode_NoneType)
register_type(str, encode_str, decode_str)
register_type(int, encode_int, decode_int)
register_type(float, encode_float, decode_float)
register_type(list, encode_list, decode_list)
register_type(dict, encode_dict, decode_dict) 
register_type(tuple, encode_tuple, decode_tuple)
register_type(bool, encode_bool, decode_bool) 
register_type(np.ndarray, encode_ndarray, decode_ndarray) 
register_type(np.float32, encode_ndarray, decode_ndarray) 
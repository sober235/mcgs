import sys, struct, re
import xml.etree.ElementTree as ET

def read_prot(fn):
    with open(fn, "rb") as f:
        magic = f.read(12)
        if magic != b"RAWDATA-V004":
            f.seek(0); magic = b""
        n = struct.unpack("<I", f.read(4))[0]
        xml = f.read(n)
    end = xml.find(b"</UProtocol>")
    xml = xml[: end + len(b"</UProtocol>")]
    return magic, n, xml

def leaves(elem, path=""):
    p = f"{path}/{elem.tag}" if path else elem.tag
    kids = list(elem)
    if not kids:
        yield p, (elem.text or "").strip()
    for k in kids:
        yield from leaves(k, p)

keys = re.compile(r"(Matrix|FOV|Slice|Thick|Dimension|ACS|Accel|Factor|Coil|Channel|Partial|Oversamp|Echo|/TR/|/TE/|Orient|Gap|Phase|Readout|Average|Concat|Interleav|Ros|ETL|TurboFactor|Inversion|TI/|Flip|Distance|Position|Normal|Rotation|Angle|Refer|Calib|Reduc|Reg|Nex|Shot|Segment|Compress|Filter|Fat|Sat|Flair|Bandwidth|Dwell|Sequence|SeqName|ProtName|Voxel|Resolution|Interp|Zero|PE|SPE|RO\b)", re.I)
for fn in sys.argv[1:]:
    magic, n, xml = read_prot(fn)
    print("=" * 100); print(fn); print("magic", magic, "prot bytes", n, "xml bytes", len(xml))
    root = ET.fromstring(xml)
    cnt = 0
    for p, v in leaves(root):
        if v == "" or v is None: continue
        if keys.search(p) and len(v) < 200:
            print(f"{p} = {v}"); cnt += 1
            if cnt > 400: print("... (truncated)"); break

import argparse
import platform
import re
import shlex
import subprocess
import tempfile


def pci_records():
    records = []
    command = shlex.split("lspci -vmm")
    output = subprocess.check_output(command).decode()

    for devices in output.strip().split("\n\n"):
        record = {}
        records.append(record)
        for row in devices.split("\n"):
            key, value = row.split("\t")
            record[key.split(":")[0]] = value

    return records


def generate_xorg_conf(devices, device_num, display):
    xorg_conf = []

    device_section = """
Section "Device"
    Identifier     "Device{device_id}"
    Driver         "nvidia"
    VendorName     "NVIDIA Corporation"
    BusID          "{bus_id}"
EndSection
"""
    server_layout_section = """
Section "ServerLayout"
    Identifier     "Layout0"
    {screen_records}
EndSection
"""
    screen_section = """
Section "Screen"
    Identifier     "Screen{screen_id}"
    Device         "Device{device_id}"
    DefaultDepth    24
    Option         "AllowEmptyInitialConfiguration" "True"
    SubSection     "Display"
        Depth       24
        Virtual 1024 768
    EndSubSection
EndSection
"""
    screen_records = []
    device = devices[int(device_num)]
    bus_id = "PCI:" + ":".join(map(lambda x: str(int(x, 16)), re.split(r"[:\.]", device["Slot"])))
    xorg_conf.append(device_section.format(device_id=display, bus_id=bus_id))
    xorg_conf.append(screen_section.format(device_id=display, screen_id=display))
    screen_records.append('Screen {screen_id} "Screen{screen_id}" 0 0'.format(screen_id=display))

    xorg_conf.append(server_layout_section.format(screen_records="\n    ".join(screen_records)))

    return "\n".join(xorg_conf)


def startx(device_num, display):
    import os

    display = int(display)
    if platform.system() != "Linux":
        raise Exception("Can only run startx on linux")
    records = list(
        filter(
            lambda r: r.get("Vendor", "") == "NVIDIA Corporation"
            and r["Class"] in ["VGA compatible controller", "3D controller"],
            pci_records(),
        )
    )

    if not records:
        raise Exception("no nvidia cards found")

    try:
        fd, path = tempfile.mkstemp()
        with open(path, "w") as f:
            f.write(generate_xorg_conf(records, device_num, display))
            print(generate_xorg_conf(records, device_num, display))
        command = shlex.split(
            "sudo Xorg -noreset +extension GLX +extension RANDR +extension RENDER -config %s :%d" % (path, display)
        )
        subprocess.call(command)
    finally:
        os.close(fd)
        os.unlink(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write a config for startx and start Xorg")
    parser.add_argument("-v", "--device", default=0, type=int, metavar="N", help="gpu id to start Xorg on.")
    parser.add_argument(
        "-d", "--display", default=0, type=int, metavar="N", help="Display index to use. Only change if 0 does not work"
    )
    args = parser.parse_args()
    startx(args.device, args.display)

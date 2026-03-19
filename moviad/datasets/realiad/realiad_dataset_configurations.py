from enum import Enum

class RealIadDefectType(Enum):
    AK = "pit"
    BX = "deformation"
    CH = "abrasion"
    HS = "scratch"
    PS = "damage"
    QS = "missing parts"
    YW = "foreign objects"
    ZW = "contamination"


class RealIadAnomalyClass(Enum):
    OK = "OK"
    NG = "NG"
    AK = "AK"
    BX = "BX"
    CH = "CH"
    HS = "HS"
    PS = "PS"
    QS = "QS"
    YW = "YW"
    ZW = "ZW"

anomaly_class_encoding = {
    RealIadAnomalyClass.OK: 0,
    RealIadAnomalyClass.NG: 1,
    RealIadAnomalyClass.AK: 2,
    RealIadAnomalyClass.BX: 3,
    RealIadAnomalyClass.CH: 4,
    RealIadAnomalyClass.HS: 5,
    RealIadAnomalyClass.PS: 6,
    RealIadAnomalyClass.QS: 7,
    RealIadAnomalyClass.YW: 8,
    RealIadAnomalyClass.ZW: 9,
}

class RealIadCategory:
    value: str
    json_path: str

class RealIadClassEnum(Enum):
    AUDIOJACK = 'audiojack'
    BOTTLE_CAP = 'bottle_cap'
    BUTTON_BATTERY = 'button_battery'
    END_CAP = 'end_cap'
    ERASER = 'eraser'
    FIRE_HOOD = 'fire_hood'
    MINT = 'mint'
    MOUNTS = 'mounts'
    PCB = 'pcb'
    PHONE_BATTERY = 'phone_battery'
    PLASTIC_NUT = 'plastic_nut'
    PLASTIC_PLUG = 'plastic_plug'
    PORCELAIN_DOLL = 'porcelain_doll'
    REGULATOR = 'regulator'
    ROLLED_STRIP_BASE = 'rolled_strip_base'
    SIM_CARD_SET = 'sim_card_set'
    SWITCH = 'switch'
    TAPE = 'tape'
    TERMINALBLOCK = 'terminalblock'
    TOOTHBRUSH = 'toothbrush'
    TOY = 'toy'
    TOY_BRICK = 'toy_brick'
    TRANSISTOR1 = 'transistor1'
    U_BLOCK = 'u_block'
    USB = 'usb'
    USB_ADAPTOR = 'usb_adaptor'
    VCPILL = 'vcpill'
    WOODEN_BEADS = 'wooden_beads'
    WOODSTICK = 'woodstick'
    ZIPPER = 'zipper'
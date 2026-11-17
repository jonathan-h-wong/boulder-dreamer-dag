"""
Pinned upstream commits for git-installed model code. Bump deliberately: stages that
declare a pin (flows/stages.py) fold it into their cache key, so bumping a pin
invalidates exactly those stages and everything downstream of them.
"""

MOGE_REF = "74fbce054ebed49800de42d0ad0e83495065719a"         # MoGe-3 release (#158)
SAM2_REF = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
GVHMR_REF = "ee960bb6e2ea2d381aa97f08e9b71ef320b624b1"
MARIGOLD_V2_REF = "9926d93b0780de5f64bee20895b25a56a98ffbf8"  # huawei-bayerlab/marigold-v2

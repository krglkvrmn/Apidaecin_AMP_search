from typing import Sequence, Literal, NewType

import numpy as np


PredictionMaskStrDecoded = NewType("PredictionMaskStrDecoded", str)
PredictionMaskStrEncoded = NewType("PredictionMaskStrEncoded", str)
PredictionMaskBool = NewType("PredictionMaskBool", np.ndarray)
PredictionMaskStr = PredictionMaskStrDecoded | PredictionMaskStrEncoded
PredictionMaskAny = PredictionMaskStr | PredictionMaskBool

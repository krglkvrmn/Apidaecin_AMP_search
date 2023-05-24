from typing import Sequence, Literal, NewType


PredictionMaskStr = str
PredictionMaskBool = Sequence[bool]
PredictionMaskAny = PredictionMaskStr | PredictionMaskBool

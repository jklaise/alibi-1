from alibi.api.types import Array
from pydantic import BaseModel, conlist, confloat
from typing import Any, List, Optional, Union
from typing_extensions import Literal

import numpy as np

ModelType = Literal['blackbox', 'whitebox', 'tensorflow', 'keras']
ExpType = Literal['local', 'global']


# TODO: add a numeric type like Union[float, int] to replace Any in Array declarations

class AlibiBaseModel(BaseModel):
    """
    We subclass pydantic's BaseModel to override a custom json encoder which handles numpy arrays.
    """

    class Config:
        json_encoders = {
            np.ndarray: lambda x: numpy_encoder(x)
        }


class DefaultMeta(AlibiBaseModel):
    name: Optional[str] = None
    type: List[ModelType] = []
    explanations: List[ExpType] = []
    params: dict = {}


# ALE
class ALEData(AlibiBaseModel):
    ale_values: List[Array[float, (-1, -1)]]  # 2D for classification, for regression 1D empty axis, i.e. (-1, 1)
    constant_value: float
    ale0: List[Array[float, (-1,)]]  # size of array corresponds to the number of targets/classes
    feature_values: List[Array[float, (-1,)]]  # TODO: should be same shape as ale_values
    feature_names: Array[str, (-1,)]  # this is an array for easier post-processing
    target_names: Array[str, (-1,)]  # ditto
    feature_deciles: List[Array[float, (11,)]]  # inclusive of 0th and 10th decile, hence 11 points


# Anchors

class AnchorDataRawTabularExamples(AlibiBaseModel):
    covered_true: Array[Any, (-1, -1)]
    covered_false: Array[Any, (-1, -1)]
    uncovered_true: Array[Any, (-1, -1)]
    uncovered_false: Array[Any, (-1, -1)]


class AnchorDataRawImageExamples(AlibiBaseModel):
    covered_true: Union[List[Array], Array]  # TODO: need to fix this and only return Array
    covered_false: Union[List[Array], Array]
    uncovered_true: Union[List[Array], Array]
    uncovered_false: Union[List[Array], Array]


class AnchorDataRawTextExamples(AlibiBaseModel):
    covered_true: Array[str, (-1,)]
    covered_false: Array[str, (-1,)]
    uncovered_true: Array[str, (-1,)]
    uncovered_false: Array[str, (-1,)]


class AnchorDataRawCommon(AlibiBaseModel):
    feature: List[int]
    mean: List[float]
    precision: List[confloat(ge=0.0, le=1.0)]
    coverage: List[confloat(ge=0.0, le=1.0)]
    all_precision: float
    num_preds: int
    success: bool
    prediction: int


class AnchorDataRawTabular(AnchorDataRawCommon):
    examples: List[AnchorDataRawTabularExamples]  # one for each partial anchor
    names: List[str]
    instance: Array[Any, (-1,)]
    instances: Array[Any, (-1, 1)]


class AnchorDataRawImage(AnchorDataRawCommon):
    examples: List[AnchorDataRawImageExamples]  # one for each partial anchor
    instance: Array
    instances: Array


class AnchorDataRawText(AnchorDataRawCommon):
    examples: List[AnchorDataRawTextExamples]  # one for each partial anchor
    names: List[str]
    instance: str
    instances: List[str]


class AnchorData(AlibiBaseModel):
    anchor: Union[List[str], Array]  # Array for images, List[str] for tabular and text
    precision: confloat(ge=0.0, le=1.0)
    coverage: confloat(ge=-0.0, le=1.0)
    raw: Union[AnchorDataRawTabular, AnchorDataRawText, AnchorDataRawImage]


# CEM
class CEMData(AlibiBaseModel):
    PP: Optional[Array]  # can do some advanced validation to ensure that at least one of these is present
    PN: Optional[Array]
    PP_pred: Optional[int]  # ditto above
    PN_pred: Optional[int]
    grads_graph: Array
    grads_num: Array
    X: Array
    X_pred: int


# CounterFactual

# CFProto

# IntegratedGradients

# KernelShap

# TreeShap

class ExplanationModel(AlibiBaseModel):
    meta: DefaultMeta
    data: Union[AnchorData, ALEData, CEMData]
    # What happens if the data is incorrect? The schema validation will fail for the correct type and
    # pydantic will attempt to check the other types in the Union which will also fail, but this results
    # in a relatively obscure message that the data of e.g. Anchor was not compatible in some fields in
    # e.g. ALE (but without actually specifying the classes tried in this Union). There should be a better
    # way of doing this for developer experience and for changes between Alibi versions. An alternative
    # would be gaving a separate model for each explanation type and use dynamic dispatch in the `Explanation`
    # interface to fetch the correct model given the name of the explainer in the `meta` dictionary.


def numpy_encoder(obj: Any) -> Any:
    if isinstance(
            obj,
            (
                    np.int_,
                    np.intc,
                    np.intp,
                    np.int8,
                    np.int16,
                    np.int32,
                    np.int64,
                    np.uint8,
                    np.uint16,
                    np.uint32,
                    np.uint64,
            ),
    ):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()

import copy
from typing import Any, Callable, Dict, Mapping, Tuple, Union
from typing_extensions import Final

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from alibi.explainers.backend import register_backend


# TODO: TBD: minimize delta or the distance?
def elastic_net_loss(beta: float, delta: tf.Variable) -> tf.Tensor:
    ax_sum = tuple(np.arange(1, len(delta.shape)))
    return beta * tf.norm(delta, ord=1, axis=ax_sum) + tf.square(tf.norm(delta, ord=2, axis=ax_sum))


def get_source_label_score(prediction: tf.Tensor, source_idx: int) -> tf.Tensor:
    return prediction[:, source_idx]


def get_max_nonsource_label_score(prediction: tf.Tensor, source_idx: int) -> tf.Tensor:
    values, indices = tf.math.top_k(prediction, k=2)
    if indices[0][0] == source_idx:  # TODO: doesn't work properly on batch!
        return values[:, 1]
    else:
        return values[:, 0]


def hinge_loss_pn(prediction: tf.Tensor, kappa: float, source_idx: int) -> tf.Tensor:
    source_label_score = get_source_label_score(prediction, source_idx)
    max_nonsource_label_score = get_max_nonsource_label_score(prediction, source_idx)
    return tf.reduce_max(0.0, -max_nonsource_label_score + source_label_score + kappa)


def hinge_loss_pp(prediction: tf.Tensor, kappa: float, source_idx: int) -> tf.Tensor:
    source_label_score = get_source_label_score(prediction, source_idx)
    max_nonsource_label_score = get_max_nonsource_label_score(prediction, source_idx)
    return tf.reduce_max(0.0, max_nonsource_label_score - source_label_score + kappa)


def cem_loss(c: float, pred: tf.Tensor, distance: tf.Tensor, ae: tf.Tensor) -> tf.Tensor:
    return c * pred + distance + ae


CEM_LOSS_SPEC_WHITEBOX = {
    'PP_prediction': {'fcn': None, 'kwargs': {}},
    'PN_prediction': {'fcn': None, 'kwargs': {}},
    'prediction': {},  # either PP or PN determined at runtime
    'distance': {'fcn': elastic_net_loss, 'kwargs': {}},
    'ae': None,
    'loss': {'fcn': cem_loss, 'kwargs': {}},
}

# TODO
CEM_LOSS_SPEC_BLACKBOX = {

}


class TFCEMOptimizer:
    framework = 'tensorflow'  # type: Final

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_spec: Dict[str, Mapping[str, Any]] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs
                 ):
        self.predictor = predictor

    def set_default_optimizer(self) -> None:
        # learning_rate_init = 1e-2 in old CEM
        self.lr_schedule = PolynomialDecay(1e-2, self.max_iter, end_learning_rate=0, power=0.5)
        self.optimizer = SGD(learning_rate=self.lr_schedule)

    # TODO: duplicated in CF
    def set_optimizer(self, optimizer: tf.keras.optimizers.Optimizer, optimizer_opts: Dict[str, Any]) -> None:

        # create a backup if the user does not override
        if optimizer is None:
            self.set_default_optimizer()
            # copy used for resetting optimizer for every c
            self._optimizer_copy = copy.deepcopy(self.optimizer)
            return

        # user passed the initialised object
        if isinstance(optimizer, tf.keras.optimizers.Optimizer):
            self.optimizer = optimizer
        # user passed just the name of the class
        else:
            if optimizer_opts is not None:
                self.optimizer = optimizer(**optimizer_opts)
            else:
                self.optimizer = optimizer()

        self._optimizer_copy = copy.deepcopy(self.optimizer)

    def initialise_variables(self, *args, **kwargs):
        raise NotImplementedError("Concrete implementations should implement variables initialisation!")

    def autograd_loss(self) -> tf.Tensor:
        raise NotImplementedError("Loss should be implemented by sub-class!")

    def update_state(self):
        raise NotImplementedError("Sub-class should implemented method for updating state.")

    # TODO: TBD these static methods should either be a separate mixin, e.g. TFUtils, or just standalone utility functions

    # TODO: TBD make_prediction can also be in a mixin class?
    def make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        return self.predictor(X, training=False)


@register_backend(consumer_class='_CEM', tag='PP')
class TF_CEM_OPTIMIZER_PP(TFCEMOptimizer):

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],  # TODO: white-box shouldn't take Callable?
                 mode: str,
                 loss_spec: Dict[str, Mapping[str, Any]] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):
        self._expected_attributes = set(CEM_LOSS_SPEC_WHITEBOX)

        # pass loss specification to the superclass
        if loss_spec is None:
            loss_spec = CEM_LOSS_SPEC_WHITEBOX
            loss_spec['prediction'] = loss_spec[f'{mode}_prediction']
        super().__init__(predictor, loss_spec, feature_range, **kwargs)

    def reset_optimizer(self):
        self.optimizer = copy.deepcopy(self._optimizer_copy)

    def initialise_variables(self):
        ...

    def initialize_solution(self):
        ...

    def autograd_loss(self):
        ...

    def check_constraint(self):
        ...

    def update_state(self):
        ...


@register_backend(consumer_class='_CEM', tag='PN')
class TF_CEM_OPTIMIZER_PN(TFCEMOptimizer):
    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],  # TODO: white-box shouldn't take Callable?
                 mode: str,
                 loss_spec: Dict[str, Mapping[str, Any]] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs):
        self._expected_attributes = set(CEM_LOSS_SPEC_WHITEBOX)

        # pass loss specification to the superclass
        if loss_spec is None:
            loss_spec = CEM_LOSS_SPEC_WHITEBOX
            loss_spec['prediction'] = loss_spec[f'{mode}_prediction']
        super().__init__(predictor, loss_spec, feature_range, **kwargs)


@register_backend(consumer_class='_CEM', predictor_type='blackbox', tag='PP')
class TF_CEM_Optimizer_BB_PP(TF_CEM_OPTIMIZER_PP):
    pass


@register_backend(consumer_class='_CEM', predictor_type='blackbox', tag='PN')
class TF_CEM_Optimizer_BB_PN(TF_CEM_OPTIMIZER_PN):
    pass

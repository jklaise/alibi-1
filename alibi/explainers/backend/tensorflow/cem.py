import copy
from functools import partial
from typing import Any, Callable, Dict, List, Mapping, Tuple, Union
from typing_extensions import Final

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from alibi.explainers.backend import register_backend
from alibi.explainers.backend.tensorflow.counterfactuals import range_constraint  # TODO: common utility


# TODO: TBD: minimize delta or the distance?
def elastic_net_loss(delta: tf.Variable, beta: float = .1) -> tf.Tensor:
    ax_sum = tuple(np.arange(1, len(delta.shape)))
    l1 = tf.reduce_sum(tf.abs(delta), axis=ax_sum)
    l2 = tf.reduce_sum(tf.square(delta), axis=ax_sum)
    return beta * l1 + l2


def get_source_label_score(prediction: tf.Tensor, source_idx: int) -> tf.Tensor:
    return prediction[:, source_idx]


def get_max_nonsource_label_score(prediction: tf.Tensor, source_idx: int) -> tf.Tensor:
    values, indices = tf.math.top_k(prediction, k=2)
    if indices[0][0] == source_idx:  # TODO: doesn't work properly on batch!
        return values[:, 1]
    else:
        return values[:, 0]


def hinge_loss_pn(prediction: tf.Tensor, source_idx: int, kappa: float = 0.) -> tf.Tensor:
    source_label_score = get_source_label_score(prediction, source_idx)
    max_nonsource_label_score = get_max_nonsource_label_score(prediction, source_idx)
    return tf.maximum(0.0, -max_nonsource_label_score + source_label_score + kappa)


def hinge_loss_pp(prediction: tf.Tensor, source_idx: int, kappa: float = 0.) -> tf.Tensor:
    source_label_score = get_source_label_score(prediction, source_idx)
    max_nonsource_label_score = get_max_nonsource_label_score(prediction, source_idx)
    return tf.maximum(0.0, max_nonsource_label_score - source_label_score + kappa)


def cem_loss(const: float, pred: tf.Tensor, distance: tf.Tensor, ae: tf.Tensor) -> tf.Tensor:
    return const * pred + distance + ae

# TODO: might be easier to use 2 separate losses so as to be able to change so as to use 2 different method_opts
# together in the same call?
CEM_LOSS_SPEC_WHITEBOX = {
    'PP_prediction': {'fcn': hinge_loss_pp, 'kwargs': {'kappa': 0.}},  # TODO: doubly defined
    'PN_prediction': {'fcn': hinge_loss_pn, 'kwargs': {'kappa': 0.}},  # TODO: doubly defined
    'norm': {'fcn': elastic_net_loss, 'kwargs': {'beta': .1}},  # TODO: beta is already defined in method_opts
    'ae': {'fcn': None, 'kwargs': {}},
    'loss': {'fcn': cem_loss, 'kwargs': {}},
}

# TODO
CEM_LOSS_SPEC_BLACKBOX = {

}


@register_backend(consumer_class='_CEM')
class TFCEMOptimizer:
    framework = 'tensorflow'  # type: Final

    def __init__(self,
                 predictor: Union[Callable, tf.keras.Model, 'keras.Model'],
                 loss_spec: Dict[str, Mapping[str, Any]] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs
                 ):
        self.predictor = predictor
        self.mode = kwargs.get('mode')
        self._expected_attributes = set(CEM_LOSS_SPEC_WHITEBOX)

        if loss_spec is None:
            loss_spec = CEM_LOSS_SPEC_WHITEBOX

        # further loss spec properties (for black-box functions or more advanced functionality) are set in sub-classes
        for term in loss_spec:
            # defer setting non-differetiable terms properties to sub-classes
            if 'numerical_diff_scheme' in term:
                continue
            this_term_kwargs = loss_spec[term]['kwargs']
            assert isinstance(this_term_kwargs, dict)
            if this_term_kwargs:
                this_term_fcn = partial(loss_spec[term]['fcn'], **this_term_kwargs)
                self.__setattr__(f"{term}_fcn", this_term_fcn)
            else:
                self.__setattr__(f"{term}_fcn", loss_spec[term]['fcn'])

        # initialisation method and constraints for search
        self.solution_constraint = None
        if feature_range is not None:
            self.solution_constraint = [feature_range[0], feature_range[1]]

        # updated at explain time since user can override. Defines default LR schedule.
        self.max_iter = None

        # updated by the calling context in the optimisation loop
        self.const = None  # type: Union[float, None]

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
            self.optimizer = optimizer(**optimizer_opts)  # user passed just the name of the class
        else:
            if optimizer_opts is not None:
                self.optimizer = optimizer(**optimizer_opts)
            else:
                self.optimizer = optimizer()

        self._optimizer_copy = copy.deepcopy(self.optimizer)

    def reset_optimizer(self) -> None:
        self.optimizer = copy.deepcopy(self._optimizer_copy)

    def initialise_variables(self, instance: np.ndarray, *args, **kwargs) -> None:
        self.instance = tf.identity(instance, name='instance')
        self.instance_class = kwargs.get('instance_class')
        self.instance_proba = kwargs.get('instance_proba')

        self.initialise_solution(instance=instance)
        # raise NotImplementedError("Concrete implementations should implement variables initialisation!")

    def initialise_solution(self, instance: np.ndarray) -> None:
        constraint_fn = None
        if self.solution_constraint is not None:
            constraint_fn = partial(range_constraint(low=self.solution_constraint[0], high=self.solution_constraint[1]))
        self.solution = tf.Variable(
            initial_value=instance,
            trainable=True,
            name=f"{self.mode}",
            constraint=constraint_fn
        )

    # TODO: can inherit with *args, **kwargs
    def step(self) -> None:
        gradients = self.get_autodiff_gradients()
        self.apply_gradients(gradients)

    def get_autodiff_gradients(self) -> List[tf.Tensor]:
        with tf.GradientTape() as tape:
            loss = self.autograd_loss()
        gradients = tape.gradient(loss, [self.solution])

        return gradients

    # TODO: depending on mode create appropriate loss
    def autograd_loss(self) -> tf.Tensor:
        norm_loss = self.norm_fcn(delta=self.solution)
        prediction = self.make_prediction(self.solution)
        # this is where we pick up the relevant PP or PN loss
        pred_fcn = getattr(self, f'{self.mode}_prediction_fcn')
        pred_loss = pred_fcn(prediction=prediction, source_idx=self.instance_class)
        total_loss = self.loss_fcn(const=self.const, pred=pred_loss, distance=norm_loss, ae=0.)

        return total_loss

    def apply_gradients(self, gradients: List[tf.Tensor]) -> None:
        self.optimizer.apply_gradients(zip(gradients, [self.solution]))

    def update_state(self):
        raise NotImplementedError("Sub-class should implemented method for updating state.")

    # TODO: TBD these static methods should either be a separate mixin, e.g. TFUtils, or just standalone utility functions
    @staticmethod
    def to_numpy_arr(X: Union[tf.Tensor, tf.Variable, np.ndarray], **kwargs) -> np.ndarray:
        """
        Casts an array-like object tf.Tensor and tf.Variable objects to a `np.array` object.
        """

        if isinstance(X, tf.Tensor) or isinstance(X, tf.Variable):
            return X.numpy()
        return X

    @staticmethod
    def to_tensor(X: np.ndarray, **kwargs) -> tf.Tensor:
        """
        Casts a numpy array object to tf.Tensor.
        """
        return tf.identity(X)

    # TODO: TBD make_prediction can also be in a mixin class?
    def make_prediction(self, X: Union[np.ndarray, tf.Variable, tf.Tensor]) -> tf.Tensor:
        return self.predictor(X, training=False).numpy()


@register_backend(consumer_class='_CEM', predictor_type='blackbox')
class TFCEMOptimizerBB(TFCEMOptimizer):
    pass

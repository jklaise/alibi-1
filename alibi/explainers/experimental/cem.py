import copy
import logging
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
from typing_extensions import Literal

from alibi.api.interfaces import Explanation, Explainer, FitMixin
from alibi.api.defaults import DEFAULT_META_CEM
from alibi.explainers.base.counterfactuals import CounterfactualBase, logger
from alibi.explainers.exceptions import CEMError
from alibi.utils.wrappers import get_blackbox_wrapper
from alibi.utils.logging import DEFAULT_LOGGING_OPTS

CEM_CONST_OPTS_DEFAULT = {
    'const_init': 1000000.,
    'const_lower_bound': 0.,
    'const_upper_bound': 1e10,
    'max_const_steps': 10,
}

CEM_SEARCH_OPTS_DEFAULT = {
    'max_iter': 100
}

CEM_METHOD_OPTS = {
    'kappa': 0.,
    'beta': .1,
    'gamma': 0.,
    'no_info_val': None,
    'search_opts': CEM_SEARCH_OPTS_DEFAULT,
    'const_opts': CEM_CONST_OPTS_DEFAULT
}

CEM_VALID_NO_INFO_TYPES = ['mean', 'median']
CEM_VALID_EXPLAIN_MODES = ['PP', 'PN', 'both']

CEM_LOGGING_OPTS_DEFAULT = copy.deepcopy(DEFAULT_LOGGING_OPTS)


def _validate_cem_loss_spec(loss_spec: dict, predictor_type: str) -> None:
    if not loss_spec:
        return


# TODO: duplicate from experimental/counterfactuals.py
def _convert_to_label(Y: np.ndarray, threshold: float = 0.5) -> int:
    # TODO: ALEX: TBD: Parametrise `threshold` in explain or do we assume people always assign labels on this rule?

    if Y.shape[1] == 1:
        return int(Y > threshold)
    else:
        return np.argmax(Y)


class CEM(Explainer, FitMixin):

    def __init__(self,
                 predictor: Union[Callable, 'tf.keras.Model', 'keras.Model'],
                 predictor_type: Literal['blackbox', 'whitebox'] = 'blackbox',
                 loss_spec: Optional[dict] = None,
                 method_opts: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 framework: Literal['pytorch', 'tensorflow'] = 'tensorflow',
                 **kwargs) -> None:
        # TODO: TBD should we compute both PN and PP by default or allow the user to choose (like now)?
        super().__init__(meta=copy.deepcopy(DEFAULT_META_CEM))

        self._explainer_type = _CEM

        explainer_args = (predictor,)
        explainer_kwargs = {
            'predictor_type': predictor_type,
            'loss_spec': loss_spec,
            'method_opts': method_opts,
            'feature_range': feature_range,
            'framework': framework,
        }  # type: Dict
        self._explainer = self._explainer_type(*explainer_args, **explainer_kwargs, **kwargs)  # type: ignore

    def fit(self,
            X: Optional[np.ndarray] = None,
            no_info_type: Literal['mean', 'median'] = 'median') -> "CEM":
        self._explainer.fit(X=X, no_info_type=no_info_type)
        self._update_metadata(self._explainer.params, params=True)

    def explain(self,
                X: np.ndarray,
                mode: Literal['PP', 'PN', 'both'] = 'both',
                Y: Optional[np.ndarray] = None,  # TODO: TBD do we still want to support this?
                logging_opts: Optional[dict] = None,
                optimizer: Optional['tf.keras.optimizers.Optimizer'] = None,  # TODO: there are 2 optimizers...
                optimizer_opts: Optional[dict] = None,
                method_opts: Optional[dict] = None) -> "Explanation":

        # self._validate_mode(mode)

        # TODO: the following boilerplate seems common to many methods
        # override default method settings with user input
        if method_opts:
            for key in method_opts:
                if isinstance(method_opts[key], Dict):
                    self._explainer.set_expected_attributes(method_opts[key])
                else:
                    self._explainer.set_expected_attributes({key: method_opts[key]})

        if logging_opts:
            self._explainer.logging_opts.update(logging_opts)

        # TODO: do both calls? sequentially or parallel? or allow user to specify in explain?
        result = self._explainer.cem(
            X,
            mode=mode,
            optimizer=optimizer,
            optimizer_opts=optimizer_opts
        )

        self._update_metadata({'mode': mode}, params=True)

        return self._build_explanation(X, result)

    def _build_explanation(self, X: np.ndarray, result: dict) -> Explanation:
        result['instance'] = X
        explanation = Explanation(meta=copy.deepcopy(self.meta), data=result)

        return explanation


# TODO: rename CounterfactualBase to be more general
class _CEM(CounterfactualBase):

    def __init__(self,
                 predictor: Union[Callable, 'tf.keras.Model', 'keras.Model'],
                 framework: Literal['pytorch', 'tensorflow'] = 'tensorflow',
                 predictor_type: Literal['blackbox', 'whitebox'] = 'blackbox',
                 loss_spec: Optional[dict] = None,
                 method_opts: Optional[dict] = None,
                 feature_range: Union[Tuple[Union[float, np.ndarray], Union[float, np.ndarray]], None] = None,
                 **kwargs
                 ):

        _validate_cem_loss_spec(loss_spec, predictor_type)
        blackbox_wrapper = get_blackbox_wrapper(framework) if predictor_type == 'blackbox' else None
        self.fitted = False

        super().__init__(
            predictor,
            framework,
            loss_spec,
            CEM_METHOD_OPTS,
            feature_range,
            predictor_type=predictor_type,
            # can pass additional kwargs for backend initialization like this
            backend_kwargs={'blackbox_wrapper': blackbox_wrapper},
            predictor_device=kwargs.get("predictor_device", None)
        )

        self.backend.device = None

        # override defaults with user input
        self.set_expected_attributes(method_opts)

        # set default options for logging (updated from the API class)
        self.logging_opts = copy.deepcopy(CEM_LOGGING_OPTS_DEFAULT)
        self.log_traces = self.logging_opts['log_traces']

    def fit(self,
            X: Optional[np.ndarray] = None,
            no_info_type: Literal['mean', 'median'] = 'median'
            ) -> "_CEM":
        self._check_no_info_type(no_info_type)

        # TODO: figure our what to do with self.shape and shape in general
        shape = (1, *X.shape[1:])

        if self.no_info_type:
            # flatten training data
            train_flat = X.reshape((X.shape[0], -1))

            # calculate no info values by feature and reshape to original shape
            if self.no_info_type == 'median':
                self.no_info_val = np.median(train_flat, axis=0).reshape(shape)
            elif self.no_info_type == 'mean':
                self.no_info_val = np.mean(train_flat, axis=0).reshape(shape)

            self.fitted = True

        self.params = {
            'no_info_type': self.no_info_type,
            'fitted': self.fitted
        }

        return self

    def _check_no_info_type(self, no_info_type: Literal['mean', 'median']) -> None:
        no_info_type_ = None
        if no_info_type not in CEM_VALID_NO_INFO_TYPES:
            logger.warning(f"Received unrecognized option {no_info_type} for no_info_type. No")
        else:
            no_info_type_ = no_info_type
        self.no_info_type = no_info_type_

    def _validate_mode(self, mode: str):
        if mode not in CEM_VALID_EXPLAIN_MODES:
            raise CEMError(f"Unknown mode {mode}. Valid explanations modes are {CEM_VALID_EXPLAIN_MODES}.")
        self.mode = mode

    def cem(self,
            instance: np.ndarray,
            mode: str,
            optimizer: Optional['tf.keras.optimizers.Optimizer'] = None,
            optimizer_opts: Optional[Dict] = None
            ):
        self._validate_mode(mode)

        # check inputs
        if instance.shape[0] != 1:
            raise CEMError(
                f"Only single instance explanations supported (leading dim = 1). Got leading dim = {instance.shape[0]}",
            )

        y = self.backend.make_prediction(self.backend.to_tensor(instance))
        y = self.backend.to_numpy_arr(y)
        instance_class = _convert_to_label(y)
        instance_proba = y[:, instance_class].item()

        self.initialise_variables(
            instance,
            instance_class,
            instance_proba,
            self.mode
        )

        self.backend.set_optimizer(optimizer, optimizer_opts)
        self.setup_tensorboard()
        if self.logging_opts['verbose']:
            logging.basicConfig(level=logging.DEBUG)

        # TODO: where to parallelize?
        result = self.search(initial=instance)

        result['instance_class'] = instance_class
        result['instance_proba'] = instance_proba
        self.backend.reset_optimizer()
        self.reset_step()

        return result

    def initialise_variables(self,
                             instance: np.ndarray,
                             instance_class: int,
                             instance_proba: float,
                             mode: str,
                             *args,
                             **kwargs) -> None:
        self.backend.initialise_variables(instance=instance, mode=mode, instance_class=instance_class,
                                          instance_proba=instance_proba)
        self.instance = instance
        self.instance_class = instance_class
        self.instance_proba = instance_proba

    def search(self, *, initial: np.ndarray) -> dict:

        # set the lower and upper bounds for the constant 'c' to scale the attack loss term
        # these bounds are updated for each c_step iteration
        const_lb = self.const_lower_bound
        cont_ub = self.const_upper_bound
        const = self.const_init

        # initial values for the best instance

        # iterate over the number of updates for 'const'
        for const_step in range(self.max_const_steps):
            self.const = const
            self.backend.const = const

            # reset learning rate
            self.backend.reset_optimizer()

            for gd_step in range(self.max_iter):
                self.backend.step(mode=self.mode)
                self.step += 1

        return {'solution': self.backend.solution}

    def bisect_const(self):
        ...

    def compare(self, x: Union[float, int, np.ndarray], y: int) -> bool:
        if not isinstance(x, (float, int, np.int64)):
            x = np.copy(x)
            if self.mode == "PP":
                x[y] -= self.kappa
            elif self.mode == "PN":
                x[y] += self.kappa
            x = np.argmax(x)
        if self.mode == "PP":
            return x == y
        else:
            return x != y

    def reset_step(self):
        ...

import copy
import numpy as np
from typing import Callable, Dict, Optional, Tuple, Union
from typing_extensions import Literal

from alibi.api.interfaces import Explanation, Explainer, FitMixin
from alibi.api.defaults import DEFAULT_META_CEM
from alibi.explainers.base.counterfactuals import CounterfactualBase, logger
from alibi.explainers.exceptions import CEMError
from alibi.utils.logging import DEFAULT_LOGGING_OPTS

CEM_CONST_OPTS_DEFAULT = {
    'c_init': 10.,
    'c_steps': 10,
}

CEM_SEARCH_OPTS_DEFAULT = {
    'max_iter': 1000
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

        self._check_mode(mode)

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

        return self._build_explanation(X, result)

    def _check_mode(self, mode: str):
        if mode not in CEM_VALID_EXPLAIN_MODES:
            raise CEMError(f"Unknown mode {mode}. Valid explanations modes are {CEM_VALID_EXPLAIN_MODES}.")

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
        ...

        self.fitted = False

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

    def cem(self):

    def PP(self):
        ...

    def PN(self):
        ...

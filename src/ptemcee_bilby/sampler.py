import copy
import datetime
import logging
import os
import time
from collections import namedtuple

import numpy as np

from bilby.core.likelihood import _safe_likelihood_call
from bilby.core.utils import (
    check_directory_exists_and_if_not_mkdir,
    logger,
)
from bilby.core.sampler.base_sampler import (
    MCMCSampler,
    SamplerError,
    signal_wrapper,
)

try:
    from bilby.core.sampler.base_sampler import LikePriorEvaluator
except ImportError:
    try:
        from bilby.core.sampler.ptemcee import LikePriorEvaluator
    except ImportError:
        raise RuntimeError("Could not import `LikePriorEvaluator`")

from .utils import (
    check_iteration,
    compute_evidence,
    get_minimum_stable_itertion,
    plot_mean_log_posterior,
    plot_tau,
    plot_walkers,
    checkpoint,
    do_nothing_function,
)

ConvergenceInputs = namedtuple(
    "ConvergenceInputs",
    [
        "autocorr_c",
        "autocorr_tol",
        "autocorr_tau",
        "gradient_tau",
        "gradient_mean_log_posterior",
        "Q_tol",
        "safety",
        "burn_in_nact",
        "burn_in_fixed_discard",
        "mean_logl_frac",
        "thin_by_nact",
        "nsamples",
        "ignore_keys_for_tau",
        "min_tau",
        "niterations_per_check",
    ],
)


class Ptemcee(MCMCSampler):
    """bilby wrapper ptemcee (https://github.com/willvousden/ptemcee)

    All positional and keyword arguments (i.e., the args and kwargs) passed to
    `run_sampler` will be propagated to `ptemcee.Sampler`, see
    documentation for that class for further help. Under Other Parameters, we
    list commonly used kwargs and the bilby defaults.

    Parameters
    ----------
    nsamples: int, (5000)
        The requested number of samples. Note, in cases where the
        autocorrelation parameter is difficult to measure, it is possible to
        end up with more than nsamples.
    burn_in_nact, thin_by_nact: int, (50, 1)
        The number of burn-in autocorrelation times to discard and the thin-by
        factor. Increasing burn_in_nact increases the time required for burn-in.
        Increasing thin_by_nact increases the time required to obtain nsamples.
    burn_in_fixed_discard: int (0)
        A fixed number of samples to discard for burn-in
    mean_logl_frac: float, (0.0.1)
        The maximum fractional change the mean log-likelihood to accept
    autocorr_tol: int, (50)
        The minimum number of autocorrelation times needed to trust the
        estimate of the autocorrelation time.
    autocorr_c: int, (5)
        The step size for the window search used by emcee.autocorr.integrated_time
    safety: int, (1)
        A multiplicative factor for the estimated autocorrelation. Useful for
        cases where non-convergence can be observed by eye but the automated
        tools are failing.
    autocorr_tau: int, (1)
        The number of autocorrelation times to use in assessing if the
        autocorrelation time is stable.
    gradient_tau: float, (0.1)
        The maximum (smoothed) local gradient of the ACT estimate to allow.
        This ensures the ACT estimate is stable before finishing sampling.
    gradient_mean_log_posterior: float, (0.1)
        The maximum (smoothed) local gradient of the logliklilhood to allow.
        This ensures the ACT estimate is stable before finishing sampling.
    Q_tol: float (1.01)
        The maximum between-chain to within-chain tolerance allowed (akin to
        the Gelman-Rubin statistic).
    min_tau: int, (1)
        A minimum tau (autocorrelation time) to accept.
    check_point_delta_t: float, (600)
        The period with which to checkpoint (in seconds).
    threads: int, (1)
        If threads > 1, a MultiPool object is setup and used.
    exit_code: int, (77)
        The code on which the sampler exits.
    store_walkers: bool (False)
        If true, store the unthinned, unburnt chains in the result. Note, this
        is not recommended for cases where tau is large.
    ignore_keys_for_tau: str
        A pattern used to ignore keys in estimating the autocorrelation time.
    pos0: str, list, np.ndarray, dict
        If a string, one of "prior" or "minimize". For "prior", the initial
        positions of the sampler are drawn from the sampler. If "minimize",
        a scipy.optimize step is applied to all parameters a number of times.
        The walkers are then initialized from the range of values obtained.
        If a list, for the keys in the list the optimization step is applied,
        otherwise the initial points are drawn from the prior.
        If a :code:`numpy` array the shape should be
        :code:`(ntemps, nwalkers, ndim)`.
        If a :code:`dict`, this should be a dictionary with keys matching the
        :code:`search_parameter_keys`. Each entry should be an array with
        shape :code:`(ntemps, nwalkers)`.

    niterations_per_check: int (5)
        The number of iteration steps to take before checking ACT. This
        effectively pre-thins the chains. Larger values reduce the per-eval
        timing due to improved efficiency. But, if it is made too large the
        pre-thinning may be overly aggressive effectively wasting compute-time.
        If you see tau=1, then niterations_per_check is likely too large.


    Other Parameters
    ----------------
    nwalkers: int, (200)
        The number of walkers
    nsteps: int, (100)
        The number of steps to take
    ntemps: int (10)
        The number of temperatures used by ptemcee
    Tmax: float
        The maximum temperature

    """

    sampler_name = "ptemcee"
    # Arguments used by ptemcee
    default_kwargs = dict(
        ntemps=10,
        nwalkers=100,
        Tmax=None,
        betas=None,
        a=2.0,
        adaptation_lag=10000,
        adaptation_time=100,
        random=None,
        adapt=False,
        swap_ratios=False,
    )

    def __init__(
        self,
        likelihood,
        priors,
        outdir="outdir",
        label="label",
        use_ratio=False,
        check_point_plot=True,
        skip_import_verification=False,
        resume=True,
        nsamples=5000,
        burn_in_nact=50,
        burn_in_fixed_discard=0,
        mean_logl_frac=0.01,
        thin_by_nact=0.5,
        autocorr_tol=50,
        autocorr_c=5,
        safety=1,
        autocorr_tau=1,
        gradient_tau=0.1,
        gradient_mean_log_posterior=0.1,
        Q_tol=1.02,
        min_tau=1,
        check_point_delta_t=600,
        threads=1,
        exit_code=77,
        plot=False,
        store_walkers=False,
        ignore_keys_for_tau=None,
        pos0="prior",
        niterations_per_check=5,
        log10beta_min=None,
        verbose=True,
        **kwargs,
    ):
        super(Ptemcee, self).__init__(
            likelihood=likelihood,
            priors=priors,
            outdir=outdir,
            label=label,
            use_ratio=use_ratio,
            plot=plot,
            skip_import_verification=skip_import_verification,
            exit_code=exit_code,
            **kwargs,
        )

        self.nwalkers = self.sampler_init_kwargs["nwalkers"]
        self.ntemps = self.sampler_init_kwargs["ntemps"]
        self.max_steps = 500

        # Checkpointing inputs
        self.resume = resume
        self.check_point_delta_t = check_point_delta_t
        self.check_point_plot = check_point_plot
        self.resume_file = f"{self.outdir}/{self.label}_checkpoint_resume.pickle"

        # Store convergence checking inputs in a named tuple
        convergence_inputs_dict = dict(
            autocorr_c=autocorr_c,
            autocorr_tol=autocorr_tol,
            autocorr_tau=autocorr_tau,
            safety=safety,
            burn_in_nact=burn_in_nact,
            burn_in_fixed_discard=burn_in_fixed_discard,
            mean_logl_frac=mean_logl_frac,
            thin_by_nact=thin_by_nact,
            gradient_tau=gradient_tau,
            gradient_mean_log_posterior=gradient_mean_log_posterior,
            Q_tol=Q_tol,
            nsamples=nsamples,
            ignore_keys_for_tau=ignore_keys_for_tau,
            min_tau=min_tau,
            niterations_per_check=niterations_per_check,
        )
        self.convergence_inputs = ConvergenceInputs(**convergence_inputs_dict)
        logger.info(f"Using convergence inputs: {self.convergence_inputs}")

        # Check if threads was given as an equivalent arg
        if threads == 1:
            for equiv in self.npool_equiv_kwargs:
                if equiv in kwargs:
                    threads = kwargs.pop(equiv)

        # Store threads
        self.threads = threads

        # Misc inputs
        self.store_walkers = store_walkers
        self.pos0 = pos0

        self._periodic = [
            self.priors[key].boundary == "periodic"
            for key in self.search_parameter_keys
        ]
        self.priors.sample()
        self._minima = np.array(
            [self.priors[key].minimum for key in self.search_parameter_keys]
        )
        self._range = (
            np.array([self.priors[key].maximum for key in self.search_parameter_keys])
            - self._minima
        )

        self.log10beta_min = log10beta_min
        if self.log10beta_min is not None:
            betas = np.logspace(0, self.log10beta_min, self.ntemps)
            logger.warning(f"Using betas {betas}")
            self.kwargs["betas"] = betas
        self.verbose = verbose

        self.iteration = 0
        self.chain_array = self.get_zero_chain_array()
        self.log_likelihood_array = self.get_zero_array()
        self.log_posterior_array = self.get_zero_array()
        self.beta_list = list()
        self.tau_list = list()
        self.tau_list_n = list()
        self.Q_list = list()
        self.time_per_check = list()

        self.nburn = np.nan
        self.thin = np.nan
        self.tau_int = np.nan
        self.nsamples_effective = 0
        self.discard = 0

    @property
    def sampler_function_kwargs(self):
        """Kwargs passed to samper.sampler()"""
        keys = ["adapt", "swap_ratios"]
        return {key: self.kwargs[key] for key in keys}

    @property
    def sampler_init_kwargs(self):
        """Kwargs passed to initialize ptemcee.Sampler()"""
        return {
            key: value
            for key, value in self.kwargs.items()
            if key not in self.sampler_function_kwargs
        }

    def _translate_kwargs(self, kwargs):
        """Translate kwargs"""
        kwargs = super()._translate_kwargs(kwargs)
        if "nwalkers" not in kwargs:
            for equiv in self.nwalkers_equiv_kwargs:
                if equiv in kwargs:
                    kwargs["nwalkers"] = kwargs.pop(equiv)

    def get_pos0_from_prior(self):
        """Draw the initial positions from the prior

        Returns
        -------
        pos0: list
            The initial postitions of the walkers, with shape (ntemps, nwalkers, ndim)

        """
        logger.info("Generating pos0 samples")
        return np.array(
            [
                [self.get_random_draw_from_prior() for _ in range(self.nwalkers)]
                for _ in range(self.kwargs["ntemps"])
            ]
        )

    def get_pos0_from_minimize(self, minimize_list=None):
        """Draw the initial positions using an initial minimization step

        See pos0 in the class initialization for details.

        Returns
        -------
        pos0: list
            The initial postitions of the walkers, with shape (ntemps, nwalkers, ndim)

        """

        from scipy.optimize import minimize

        from bilby.core.utils import random

        # Set up the minimize list: keys not in this list will have initial
        # positions drawn from the prior
        if minimize_list is None:
            minimize_list = self.search_parameter_keys
            pos0 = np.zeros((self.kwargs["ntemps"], self.kwargs["nwalkers"], self.ndim))
        else:
            pos0 = np.array(self.get_pos0_from_prior())

        logger.info(f"Attempting to set pos0 for {minimize_list} from minimize")

        likelihood_copy = copy.copy(self.likelihood)

        def neg_log_like(params):
            """Internal function to minimize"""
            try:
                parameters = {key: val for key, val in zip(minimize_list, params)}
                return -_safe_likelihood_call(likelihood_copy, parameters)
            except RuntimeError:
                return +np.inf

        # Bounds used in the minimization
        bounds = [
            (self.priors[key].minimum, self.priors[key].maximum)
            for key in minimize_list
        ]

        # Run the minimization step several times to get a range of values
        trials = 0
        success = []
        while True:
            draw = self.priors.sample()
            likelihood_copy.parameters.update(draw)
            x0 = [draw[key] for key in minimize_list]
            res = minimize(
                neg_log_like, x0, bounds=bounds, method="L-BFGS-B", tol=1e-15
            )
            if res.success:
                success.append(res.x)
            if trials > 100:
                raise SamplerError("Unable to set pos0 from minimize")
            if len(success) >= 10:
                break

        # Initialize positions from the range of values
        success = np.array(success)
        for i, key in enumerate(minimize_list):
            pos0_min = np.min(success[:, i])
            pos0_max = np.max(success[:, i])
            logger.info(f"Initialize {key} walkers from {pos0_min}->{pos0_max}")
            j = self.search_parameter_keys.index(key)
            pos0[:, :, j] = random.rng.uniform(
                pos0_min,
                pos0_max,
                size=(self.kwargs["ntemps"], self.kwargs["nwalkers"]),
            )
        return pos0

    def get_pos0_from_array(self):
        if self.pos0.shape != (self.ntemps, self.nwalkers, self.ndim):
            raise ValueError(
                "Shape of starting array should be (ntemps, nwalkers, ndim). "
                f"In this case that is ({self.ntemps}, {self.nwalkers}, "
                f"{self.ndim}), got {self.pos0.shape}"
            )
        else:
            return self.pos0

    def get_pos0_from_dict(self):
        """
        Initialize the starting points from a passed dictionary.

        The :code:`pos0` passed to the :code:`Sampler` should be a dictionary
        with keys matching the :code:`search_parameter_keys`.
        Each entry should have shape :code:`(ntemps, nwalkers)`.
        """
        pos0 = np.array([self.pos0[key] for key in self.search_parameter_keys])
        self.pos0 = np.moveaxis(pos0, 0, -1)
        return self.get_pos0_from_array()

    def setup_sampler(self):
        """Either initialize the sampler or read in the resume file"""
        import ptemcee

        if ptemcee.__version__ == "1.0.0":
            # This is a very ugly hack to support numpy>=1.24
            ptemcee.sampler.np.float = float

        if (
            os.path.isfile(self.resume_file)
            and os.path.getsize(self.resume_file)
            and self.resume is True
        ):
            import dill

            logger.info(f"Resume data {self.resume_file} found")
            with open(self.resume_file, "rb") as file:
                data = dill.load(file)

            # Extract the check-point data
            self.sampler = data["sampler"]
            self.iteration = data["iteration"]
            self.chain_array = data["chain_array"]
            self.log_likelihood_array = data["log_likelihood_array"]
            self.log_posterior_array = data["log_posterior_array"]
            self.pos0 = data["pos0"]
            self.beta_list = data["beta_list"]
            self.sampler._betas = np.array(self.beta_list[-1])
            self.tau_list = data["tau_list"]
            self.tau_list_n = data["tau_list_n"]
            self.Q_list = data["Q_list"]
            self.time_per_check = data["time_per_check"]

            # Initialize the pool
            self.sampler.pool = self.pool
            self.sampler.threads = self.threads

            logger.info(f"Resuming from previous run with time={self.iteration}")

        else:
            # Initialize the PTSampler
            if self.threads == 1:
                self.sampler = ptemcee.Sampler(
                    dim=self.ndim,
                    logl=self.log_likelihood,
                    logp=self.log_prior,
                    **self.sampler_init_kwargs,
                )
            else:
                self.sampler = ptemcee.Sampler(
                    dim=self.ndim,
                    logl=do_nothing_function,
                    logp=do_nothing_function,
                    threads=self.threads,
                    **self.sampler_init_kwargs,
                )

            self.sampler._likeprior = LikePriorEvaluator()

            # Initialize storing results
            self.iteration = 0
            self.chain_array = self.get_zero_chain_array()
            self.log_likelihood_array = self.get_zero_array()
            self.log_posterior_array = self.get_zero_array()
            self.beta_list = list()
            self.tau_list = list()
            self.tau_list_n = list()
            self.Q_list = list()
            self.time_per_check = list()
            self.pos0 = self.get_pos0()

        return self.sampler

    def get_zero_chain_array(self):
        return np.zeros((self.nwalkers, self.max_steps, self.ndim))

    def get_zero_array(self):
        return np.zeros((self.ntemps, self.nwalkers, self.max_steps))

    def get_pos0(self):
        """Master logic for setting pos0"""
        if isinstance(self.pos0, str) and self.pos0.lower() == "prior":
            return self.get_pos0_from_prior()
        elif isinstance(self.pos0, str) and self.pos0.lower() == "minimize":
            return self.get_pos0_from_minimize()
        elif isinstance(self.pos0, list):
            return self.get_pos0_from_minimize(minimize_list=self.pos0)
        elif isinstance(self.pos0, np.ndarray):
            return self.get_pos0_from_array()
        elif isinstance(self.pos0, dict):
            return self.get_pos0_from_dict()
        else:
            raise SamplerError(f"pos0={self.pos0} not implemented")

    def _close_pool(self):
        if getattr(self.sampler, "pool", None) is not None:
            self.sampler.pool = None
        if "pool" in self.result.sampler_kwargs:
            del self.result.sampler_kwargs["pool"]
        super(Ptemcee, self)._close_pool()

    @signal_wrapper
    def run_sampler(self):
        self._setup_pool()
        sampler = self.setup_sampler()

        t0 = datetime.datetime.now()
        logger.info("Starting to sample")

        while True:
            for pos0, log_posterior, log_likelihood in sampler.sample(
                self.pos0,
                storechain=False,
                iterations=self.convergence_inputs.niterations_per_check,
                **self.sampler_function_kwargs,
            ):
                pos0[:, :, self._periodic] = (
                    np.mod(
                        pos0[:, :, self._periodic] - self._minima[self._periodic],
                        self._range[self._periodic],
                    )
                    + self._minima[self._periodic]
                )

            if self.iteration == self.chain_array.shape[1]:
                self.chain_array = np.concatenate(
                    (self.chain_array, self.get_zero_chain_array()), axis=1
                )
                self.log_likelihood_array = np.concatenate(
                    (self.log_likelihood_array, self.get_zero_array()), axis=2
                )
                self.log_posterior_array = np.concatenate(
                    (self.log_posterior_array, self.get_zero_array()), axis=2
                )

            self.pos0 = pos0

            self.chain_array[:, self.iteration, :] = pos0[0, :, :]
            self.log_likelihood_array[:, :, self.iteration] = log_likelihood
            self.log_posterior_array[:, :, self.iteration] = log_posterior
            self.mean_log_posterior = np.mean(
                self.log_posterior_array[:, :, : self.iteration], axis=1
            )

            # (nwalkers, ntemps, iterations)
            # so mean_log_posterior is shaped (nwalkers, iterations)

            # Calculate time per iteration
            self.time_per_check.append((datetime.datetime.now() - t0).total_seconds())
            t0 = datetime.datetime.now()

            self.iteration += 1

            # Calculate minimum iteration step to discard
            minimum_iteration = get_minimum_stable_itertion(
                self.mean_log_posterior, frac=self.convergence_inputs.mean_logl_frac
            )
            logger.debug(f"Minimum iteration = {minimum_iteration}")

            # Calculate the maximum discard number
            discard_max = np.max(
                [self.convergence_inputs.burn_in_fixed_discard, minimum_iteration]
            )

            if self.iteration > discard_max + self.nwalkers:
                # If we have taken more than nwalkers steps after the discard
                # then set the discard
                self.discard = discard_max
            else:
                # If haven't discard everything (avoid initialisation bias)
                logger.debug("Too few steps to calculate convergence")
                self.discard = self.iteration

            (
                stop,
                self.nburn,
                self.thin,
                self.tau_int,
                self.nsamples_effective,
            ) = check_iteration(
                self.iteration,
                self.chain_array[:, self.discard : self.iteration, :],
                sampler,
                self.convergence_inputs,
                self.search_parameter_keys,
                self.time_per_check,
                self.beta_list,
                self.tau_list,
                self.tau_list_n,
                self.Q_list,
                self.mean_log_posterior,
                verbose=self.verbose,
            )

            if stop:
                logger.info("Finished sampling")
                break

            # If a checkpoint is due, checkpoint
            if os.path.isfile(self.resume_file):
                last_checkpoint_s = time.time() - os.path.getmtime(self.resume_file)
            else:
                last_checkpoint_s = np.sum(self.time_per_check)

            if last_checkpoint_s > self.check_point_delta_t:
                self.write_current_state(plot=self.check_point_plot)

        # Run a final checkpoint to update the plots and samples
        self.write_current_state(plot=self.check_point_plot)

        # Get 0-likelihood samples and store in the result
        self.result.samples = self.chain_array[
            :, self.discard + self.nburn : self.iteration : self.thin, :
        ].reshape((-1, self.ndim))
        loglikelihood = self.log_likelihood_array[
            0, :, self.discard + self.nburn : self.iteration : self.thin
        ]  # nwalkers, nsteps
        self.result.log_likelihood_evaluations = loglikelihood.reshape((-1))

        if self.store_walkers:
            self.result.walkers = self.sampler.chain
        self.result.nburn = self.nburn
        self.result.discard = self.discard

        log_evidence, log_evidence_err = compute_evidence(
            sampler,
            self.log_likelihood_array,
            self.outdir,
            self.label,
            self.discard,
            self.nburn,
            self.thin,
            self.iteration,
        )
        self.result.log_evidence = log_evidence
        self.result.log_evidence_err = log_evidence_err

        self.result.sampling_time = datetime.timedelta(
            seconds=np.sum(self.time_per_check)
        )

        self._close_pool()

        return self.result

    def write_current_state(self, plot=True):
        check_directory_exists_and_if_not_mkdir(self.outdir)
        checkpoint(
            self.iteration,
            self.outdir,
            self.label,
            self.nsamples_effective,
            self.sampler,
            self.discard,
            self.nburn,
            self.thin,
            self.search_parameter_keys,
            self.resume_file,
            self.log_likelihood_array,
            self.log_posterior_array,
            self.chain_array,
            self.pos0,
            self.beta_list,
            self.tau_list,
            self.tau_list_n,
            self.Q_list,
            self.time_per_check,
        )

        if plot:
            try:
                # Generate the walkers plot diagnostic
                plot_walkers(
                    self.chain_array[:, : self.iteration, :],
                    self.nburn,
                    self.thin,
                    self.search_parameter_keys,
                    self.outdir,
                    self.label,
                    self.discard,
                )
            except Exception as e:
                logger.info(f"Walkers plot failed with exception {e}")

            try:
                # Generate the tau plot diagnostic if DEBUG
                if logger.level < logging.INFO:
                    plot_tau(
                        self.tau_list_n,
                        self.tau_list,
                        self.search_parameter_keys,
                        self.outdir,
                        self.label,
                        self.tau_int,
                        self.convergence_inputs.autocorr_tau,
                    )
            except Exception as e:
                logger.info(f"tau plot failed with exception {e}")

            try:
                plot_mean_log_posterior(
                    self.mean_log_posterior,
                    self.outdir,
                    self.label,
                )
            except Exception as e:
                logger.info(f"mean_logl plot failed with exception {e}")

    @classmethod
    def get_expected_outputs(cls, outdir=None, label=None):
        """Get lists of the expected outputs directories and files.

        These are used by :code:`bilby_pipe` when transferring files via HTCondor.

        Parameters
        ----------
        outdir : str
            The output directory.
        label : str
            The label for the run.

        Returns
        -------
        list
            List of file names.
        list
            List of directory names. Will always be empty for ptemcee.
        """
        filenames = [f"{outdir}/{label}_checkpoint_resume.pickle"]
        return filenames, []

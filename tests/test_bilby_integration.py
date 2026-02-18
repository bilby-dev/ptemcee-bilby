import threading
import time
import signal

import bilby
import pytest


@pytest.fixture(scope="session")
def sampler():
    return "ptemcee"


@pytest.fixture(scope="session")
def sampler_kwargs():
    return dict(nsamples=100, nwalkers=50, ntemps=1, burn_in_act=1, frac_threshold=0.5)


@pytest.fixture
def outdir(tmp_path):
    return tmp_path / "outdir"


@pytest.fixture
def conversion_function():
    def _conversion_function(parameters, likelihood, prior):
        converted = parameters.copy()
        if "derived" not in converted:
            converted["derived"] = converted["x"] * converted["y"]
        return converted

    return _conversion_function


def run_sampler(
    likelihood, priors, outdir, conversion_function, sampler, npool=None, **kwargs
):
    result = bilby.run_sampler(
        likelihood=likelihood,
        priors=priors,
        sampler=sampler,
        outdir=str(outdir),
        save="hdf5",
        npool=npool,
        conversion_function=conversion_function,
        **kwargs,
    )
    return result


def test_run_sampler(
    bilby_gaussian_likelihood_and_priors,
    outdir,
    conversion_function,
    npool,
    sampler,
    sampler_kwargs,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors
    result = run_sampler(
        likelihood,
        priors,
        outdir,
        conversion_function,
        sampler,
        npool,
        **sampler_kwargs,
    )
    assert "derived" in result.posterior


def test_interrupt_sampler(
    bilby_gaussian_likelihood_and_priors,
    outdir,
    conversion_function,
    sampler,
    npool,
):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    started = threading.Event()
    calls = 0

    def trigger_signal():
        if started.wait(timeout=10):
            signal.raise_signal(signal.SIGINT)
        else:
            # if we never started, don't hang the test forever
            pytest.fail("Sampler never began likelihood evaluations")

    thread = threading.Thread(target=trigger_signal, daemon=True)
    thread.start()

    original_log_likelihood = likelihood.log_likelihood

    def slow_log_likelihood(parameters=None):
        nonlocal calls
        calls += 1
        # Bilby tests the likelihood before starting sampling
        if calls > 500:
            started.set()
        time.sleep(0.01)
        return original_log_likelihood(parameters)

    likelihood.log_likelihood = slow_log_likelihood

    label = "test_interrupt"

    with pytest.raises((SystemExit, KeyboardInterrupt)) as exc:
        run_sampler(
            likelihood,
            priors,
            outdir,
            conversion_function,
            sampler,
            exit_code=5,
            resume=True,
            label=label,
            npool=npool,
        )

    if isinstance(exc.value, SystemExit):
        assert exc.value.code == 5

    assert (outdir / f"{label}_checkpoint_resume.pickle").exists()

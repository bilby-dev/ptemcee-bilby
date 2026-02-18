import bilby
import numpy as np
import os
import pytest
from ptemcee_bilby import Ptemcee


@pytest.fixture()
def SamplerClass():
    return Ptemcee


@pytest.fixture()
def create_sampler(SamplerClass, bilby_gaussian_likelihood_and_priors, tmp_path):
    likelihood, priors = bilby_gaussian_likelihood_and_priors

    def create_fn(**kwargs):
        return SamplerClass(
            likelihood,
            priors,
            outdir=tmp_path / "outdir",
            label="test",
            use_ratio=False,
            **kwargs,
        )

    return create_fn


@pytest.fixture
def sampler(create_sampler):
    return create_sampler()


@pytest.fixture
def expected_default_kwargs():
    return dict(
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


def test_default_kwargs(sampler, expected_default_kwargs):
    assert sampler.kwargs == expected_default_kwargs


@pytest.mark.parametrize(
    "equiv", bilby.core.sampler.base_sampler.MCMCSampler.nwalkers_equiv_kwargs
)
def test_translate_kwargs(create_sampler, equiv, expected_default_kwargs):
    sampler = create_sampler(**{equiv: 123})
    expected_kwargs = expected_default_kwargs.copy()
    expected_kwargs["nwalkers"] = 123
    assert sampler.kwargs == expected_kwargs


def test_get_expected_outputs():
    label = "par0"
    outdir = os.path.join("some", "bilby_pipe", "dir")
    filenames, directories = Ptemcee.get_expected_outputs(outdir=outdir, label=label)
    assert len(filenames) == 1
    assert len(directories) == 0
    assert os.path.join(outdir, f"{label}_checkpoint_resume.pickle") in filenames


def test_set_pos0_using_array(sampler, create_sampler):
    pos0 = sampler.get_pos0()
    new_sampler = create_sampler(pos0=pos0)
    new_pos0 = new_sampler.get_pos0()
    assert np.array_equal(pos0, new_pos0)


def test_set_pos0_using_dict(sampler, create_sampler):
    old = np.array(sampler.get_pos0())
    pos0 = np.moveaxis(old, -1, 0)
    pos0 = {key: points for key, points in zip(sampler.search_parameter_keys, pos0)}
    new_sampler = create_sampler(pos0=pos0)
    new = new_sampler.get_pos0()
    assert np.array_equal(new, old)


def test_set_pos0_from_minimize(sampler, create_sampler):
    old = sampler.get_pos0().shape
    new_sampler = create_sampler(pos0="minimize")
    new = new_sampler.get_pos0().shape
    assert old == new

import numpyro
from numpyro import distributions as dist
from numpyro.infer import NUTS, MCMC
import jax
from jax import numpy as jnp, random
import typing


def get_input_data(county_election):
    num_voters_list = []
    votes_list = []
    for _, election in county_election.items():
        head_to_head_election = election.sort_values(by="votes", ascending=False).iloc[
            :2
        ]
        num_voters_list.append(head_to_head_election["votes"].sum())
        votes_list.append(head_to_head_election.iloc[0]["votes"])
    num_voters_array = jnp.array(num_voters_list)
    votes_array = jnp.array(votes_list)
    return num_voters_array, votes_array


def model_C2_batched(num_voters_array=None, votes_array=None):
    p_sigma = numpyro.sample("p_sigma", dist.Exponential(rate=1.0))
    p_mu = numpyro.sample("p_mu", dist.Beta(4, 4))
    with numpyro.plate("election", len(num_voters_array)) as ind:
        p = numpyro.sample("p", dist.BetaProportion(p_mu, p_sigma))
        votes = votes_array[ind] if votes_array is not None else None
        numpyro.sample(
            "observed_votes",
            dist.Binomial(total_count=num_voters_array[ind], probs=p),
            obs=votes,
        )


def get_C2_samples(
    rng: jax.Array, num_voters_array: jnp.ndarray, votes_array: jnp.ndarray, verbose: bool = False
) -> typing.Dict[str, jnp.ndarray]:
    kernel = NUTS(model_C2_batched)
    num_samples = 2_000
    mcmc = MCMC(kernel, num_warmup=1000, num_samples=num_samples, progress_bar=verbose)
    mcmc.run(
        rng,
        num_voters_array,
        votes_array,
    )
    return mcmc.get_samples()


def get_pivot_odds_from_C2_samples(
    rng: jax.Array,
    c2_samples: typing.Dict[str, jnp.ndarray],
    num_voters_array: jnp.ndarray,
) -> float:
    p_mu = c2_samples["p_mu"]
    p_sigma = c2_samples["p_sigma"]
    p_05 = jax.scipy.stats.gaussian_kde(dist.BetaProportion(p_mu, p_sigma).sample(rng))(
        0.5
    )[0].item()
    return (p_05 / (jnp.mean(num_voters_array) + 1)).item()


def evaluate_kde(points: jnp.ndarray, value: float):
    kde = jax.scipy.stats.gaussian_kde(points)
    return kde(value)


def get_pivot_odds(p_samples, num_voters_array):
    return jax.vmap(evaluate_kde, in_axes=(0, None))(p_samples.T, 0.5)[:, 0] / (
        num_voters_array + 1
    )

def get_pivot_odds_from_elections_c2(rng, county_elections):
    rngs = random.split(rng, 2)
    num_voters_array, votes_array = get_input_data(county_elections)
    c2_samples = get_C2_samples(rngs[0], num_voters_array, votes_array)
    return get_pivot_odds_from_C2_samples(rngs[1], c2_samples, num_voters_array)
## How are your odds?

The goal of this project is to meaningfully estimate the relative importance of
your vote on the local, state, and national levels.

## Methodology

### Model Choice

We use the independent binomial model to estimate the probability of a "pivot
election." This is mostly a matter of convenience in order to be able to
practically compute the odds of a pivot election in a large number of elections.
We justify this choice with the empirical analysis of [4], but it should be
noted that they do not directly assert that the independent binomial model is
applicable for smaller elections and seem to imply that it there is less
evidence of it being a good model for smaller elections. A better model would
more carefully interpret the interests of the voters.

Technically, our model will not be binomial, but multinomial, since many local
elections feature more than two parties. The model relies on a prior
distribution for the expected margin, so we will use partial pooling across
elections with a dirichlet prior to estimate this prior distribution.

### Other considerations

In [2], the authors use simulated elections to estimate the distribution of
margins, from which they estimate a Student _t_-distribution with 4 degrees of
freedom. Apparently this use of distribution matches choices in the literature
to allow for greater outlier events. This distribution is used to estimate the
odds of a tie, since the data is relatively sparse.

This project uses data from: https://electionlab.mit.edu/data

References:

[1] Chamberlain, Rothschild. 1981

[2] Gelman, Silver, Edlin. 2008

[3] MIT Election Data and Science Lab, 2022, "Local Precinct-Level Returns
2018", https://doi.org/10.7910/DVN/CHYXUP, Harvard Dataverse, V1

[4] Mulligan, Hunter. 2003

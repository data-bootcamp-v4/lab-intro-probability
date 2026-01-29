import math
from scipy.stats import binom, poisson, norm

# =========================
# Challenge 1 — Ironhack Airlines (Binomial)
# =========================
# 450 seats, 460 tickets sold
# Each passenger has 3% chance of missing flight => show-up prob p = 0.97
# Let X = number of passengers who show up ~ Binomial(n=460, p=0.97)
# Need P(X <= 450)

n = 460
p_show = 0.97
seats = 450

prob_have_seats = binom.cdf(seats, n, p_show)
print("Challenge 1 - P(have seats for all who show):", prob_have_seats)


# =========================
# Challenge 2 — Call Center (Geometric idea)
# =========================
# Success on an attempt = 0.3
# "needs to make at least three attempts before successfully resolving"
# => first two attempts must fail
p_success = 0.3
p_fail = 1 - p_success

prob_at_least_3_attempts = p_fail**2
print("Challenge 2 - P(need at least 3 attempts):", prob_at_least_3_attempts)


# =========================
# Challenge 3 — Website Traffic (Poisson)
# =========================
# average 500 visits per hour => X ~ Poisson(lam=500)
# server handles up to 550 per hour => overwhelmed if X > 550
lam = 500
capacity = 550

prob_overwhelmed_hour = poisson.sf(capacity, lam)  # P(X > capacity)
print("Challenge 3a - P(overwhelmed in one hour):", prob_overwhelmed_hour)

# Probability overwhelmed at some point during a day (24 hours)
# Assume independence across hours (approx)
prob_overwhelmed_day = 1 - (1 - prob_overwhelmed_hour)**24
print("Challenge 3b - P(overwhelmed at least once in 24h):", prob_overwhelmed_day)


# =========================
# Challenge 4 — Helpdesk (Exponential inter-arrival time)
# =========================
# On average a customer arrives every 10 minutes
# Exponential waiting time with mean 10 => rate λ = 1/10 per minute
rate = 1/10  # per minute

# Probability next customer arrives within next 5 minutes:
# P(T <= 5) = 1 - exp(-λ*5)
prob_within_5 = 1 - math.exp(-rate * 5)
print("Challenge 4a - P(next arrival within 5 min):", prob_within_5)

# Probability employee can take a break (no customer for 15 minutes):
# P(T > 15) = exp(-λ*15)
prob_no_customer_15 = math.exp(-rate * 15)
print("Challenge 4b - P(no customer for 15 min):", prob_no_customer_15)


# =========================
# Challenge 5 — Bird weights (Normal)
# =========================
# X ~ Normal(mean=150, sd=10)
# P(140 <= X <= 160) => between -1 and +1 standard deviations
mu = 150
sigma = 10

prob_between = norm.cdf(160, mu, sigma) - norm.cdf(140, mu, sigma)
print("Challenge 5 - P(140 to 160 grams):", prob_between)


# =========================
# Challenge 6 — Exponential lifetime
# =========================
# mean lifetime = 50 hours => rate λ = 1/50 per hour
# P(fails within first 30 hours): P(T <= 30) = 1 - exp(-λ*30)
rate_h = 1/50
prob_fail_30 = 1 - math.exp(-rate_h * 30)
print("Challenge 6 - P(fail within 30 hours):", prob_fail_30)

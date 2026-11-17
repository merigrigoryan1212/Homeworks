"""A/B testing with multi-armed bandits: Epsilon-Greedy vs. Thompson Sampling.

Four advertisement options (bandits) with true mean rewards [1, 2, 3, 4] are
tested over 20000 trials. Observed rewards are drawn from a Normal
distribution centred on the arm's true mean (known precision).

"""
############################### LOGGER
import os
from abc import ABC, abstractmethod

import matplotlib

matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

############################### CONFIG
BANDIT_REWARDS = [1, 2, 3, 4]
NUMBER_OF_TRIALS = 20000
REWARD_PRECISION = 1.0  # known precision (1 / variance) of the reward noise
PRIOR_PRECISION = 0.01  # Thompson Sampling: precision of the N(0, 1/tau0) prior
SEED = 42
OUTPUT_DIR = "output"


class Bandit(ABC):

    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # print average reward (using f strings to make it informative)
        # print average regret (using f strings to make it informative)
        pass

    # ---- additions ---- #
    def _init_common(self, p, precision, seed):
        """Initialise the state shared by every algorithm.

        Parameters
        ----------
        p : list of float
            True mean reward of each bandit (arm).
        precision : float
            Known precision (1 / variance) of the reward noise.
        seed : int
            Seed of the random generator (reproducibility).
        """
        self.p = np.asarray(p, dtype=float)
        self.k = len(self.p)
        self.precision = precision
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(self.k, dtype=int)
        self.estimates = np.zeros(self.k)
        self.chosen = []      # arm chosen at every trial
        self.rewards = []     # reward received at every trial
        self.regrets = []     # per-trial regret (best mean - chosen mean)

    def _draw_reward(self, arm):
        """Sample a noisy reward from the chosen arm.

        Parameters
        ----------
        arm : int
            Index of the arm that is pulled.

        Returns
        -------
        float
            Reward drawn from N(p[arm], 1 / precision).
        """
        return self.rng.normal(self.p[arm], 1.0 / np.sqrt(self.precision))

    def _log_trial(self, arm, reward):
        """Store the outcome of a single trial."""
        self.chosen.append(arm)
        self.rewards.append(reward)
        self.regrets.append(self.p.max() - self.p[arm])

    def to_dataframe(self):
        """Return the experiment data as a DataFrame.

        Returns
        -------
        pandas.DataFrame
            Columns ``Bandit``, ``Reward``, ``Algorithm``.
        """
        return pd.DataFrame(
            {
                "Bandit": [int(self.p[a]) if float(self.p[a]).is_integer() else self.p[a]
                           for a in self.chosen],
                "Reward": self.rewards,
                "Algorithm": self.name,
            }
        )

    def report(self):
        """Store the data in a CSV file and print the summary statistics.

        The CSV has the columns ``Bandit``, ``Reward``, ``Algorithm``.
        Cumulative reward, cumulative regret, average reward and average
        regret are printed with the logger.
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        path = os.path.join(OUTPUT_DIR, f"{self.name}_rewards.csv")
        self.to_dataframe().to_csv(path, index=False)
        logger.info(f"[{self.name}] data stored in {path}")

        n = len(self.rewards)
        cum_reward = float(np.sum(self.rewards))
        cum_regret = float(np.sum(self.regrets))
        logger.info(f"[{self.name}] cumulative reward: {cum_reward:.2f}")
        logger.info(f"[{self.name}] cumulative regret: {cum_regret:.2f}")
        logger.info(f"[{self.name}] average reward over {n} trials: {cum_reward / n:.4f}")
        logger.info(f"[{self.name}] average regret over {n} trials: {cum_regret / n:.4f}")
        logger.info(f"[{self.name}] pulls per bandit: "
                    f"{dict(zip(self.p.tolist(), self.counts.tolist()))}")


#--------------------------------------#


class Visualization():
    """Plots for the learning process and the algorithm comparison."""

    def plot1(self, algorithms):
        """Visualize the performance (learning process) of each algorithm.

        For every algorithm two panels are drawn: the running average reward
        on a linear scale and on a log scale (x axis), together with the
        optimal mean reward.

        Parameters
        ----------
        algorithms : list of Bandit
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        for algo in algorithms:
            r = np.asarray(algo.rewards)
            avg = np.cumsum(r) / np.arange(1, len(r) + 1)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for ax, scale in zip(axes, ["linear", "log"]):
                ax.plot(avg, label=f"{algo.name} average reward")
                ax.axhline(algo.p.max(), color="red", ls="--", label="optimal mean")
                ax.set_xscale(scale)
                ax.set_title(f"{algo.name}: learning process ({scale} scale)")
                ax.set_xlabel("Trial")
                ax.set_ylabel("Average reward")
                ax.legend()
            fig.tight_layout()
            path = os.path.join(OUTPUT_DIR, f"plot1_{algo.name}.png")
            fig.savefig(path, dpi=150)
            plt.close(fig)
            logger.info(f"plot1 saved to {path}")

    def plot2(self, eg, ts):
        """Compare Epsilon-Greedy and Thompson Sampling.

        Left: cumulative rewards. Right: cumulative regrets.

        Parameters
        ----------
        eg : EpsilonGreedy
            Finished Epsilon-Greedy experiment.
        ts : ThompsonSampling
            Finished Thompson Sampling experiment.
        """
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for algo in (eg, ts):
            axes[0].plot(np.cumsum(algo.rewards), label=algo.name)
            axes[1].plot(np.cumsum(algo.regrets), label=algo.name)
        axes[0].set_title("Cumulative reward")
        axes[1].set_title("Cumulative regret")
        for ax in axes:
            ax.set_xlabel("Trial")
            ax.legend()
        axes[0].set_ylabel("Cumulative reward")
        axes[1].set_ylabel("Cumulative regret")
        fig.tight_layout()
        path = os.path.join(OUTPUT_DIR, "plot2_comparison.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info(f"plot2 saved to {path}")


#--------------------------------------#


class EpsilonGreedy(Bandit):
    """Epsilon-Greedy with epsilon decaying as 1/t."""

    name = "EpsilonGreedy"

    def __init__(self, p, precision=REWARD_PRECISION, seed=SEED):
        """Create the algorithm.

        Parameters
        ----------
        p : list of float - True mean reward of each bandit.
        precision : float, optional - Known precision of the reward noise.
        seed : int, optional - Random seed.
        """
        self._init_common(p, precision, seed)
        self.t = 0

    def __repr__(self):
        return (f"EpsilonGreedy(arms={self.p.tolist()}, "
                f"epsilon=1/t, trials={self.t})")

    def pull(self):
        """Choose an arm and pull it.

        With probability ``epsilon = 1/t`` a random arm is explored, otherwise
        the arm with the highest estimated mean is exploited.

        Returns - tuple of (int, float) - The chosen arm and the reward received.
        """
        self.t += 1
        epsilon = 1.0 / self.t
        if self.rng.random() < epsilon:
            arm = int(self.rng.integers(self.k))
        else:
            arm = int(np.argmax(self.estimates))
        return arm, self._draw_reward(arm)

    def update(self, arm, reward):
        """Update the running mean of the pulled arm.

        Parameters - arm, reward
        """
        self.counts[arm] += 1
        self.estimates[arm] += (reward - self.estimates[arm]) / self.counts[arm]
        self._log_trial(arm, reward)

    def experiment(self, n_trials=NUMBER_OF_TRIALS):
        """Run the experiment.

        Parameters - n_trials (default 20000)
        """
        logger.info(f"Running {self.name} for {n_trials} trials")
        for _ in range(n_trials):
            arm, reward = self.pull()
            self.update(arm, reward)


#--------------------------------------#


class ThompsonSampling(Bandit):
    """Thompson Sampling with Normal rewards and known precision.

    Each arm has a Normal posterior over its mean: with prior N(0, 1/tau0)
    and known reward precision tau, after n pulls with reward sum S the
    posterior precision is ``tau0 + n*tau`` and the posterior mean is
    ``tau * S / (tau0 + n*tau)``.
    """

    name = "ThompsonSampling"

    def __init__(self, p, precision=REWARD_PRECISION,
                 prior_precision=PRIOR_PRECISION, seed=SEED):
        """Create the algorithm.

        Parameters
        ----------
        p : list of float - True mean reward of each bandit.
        precision : float, optional - Known precision of the reward noise.
        prior_precision : float, optional - Precision of the prior over each arm's mean.
        seed : int, optional - Random seed.
        """
        self._init_common(p, precision, seed)
        self.prior_precision = prior_precision
        self.post_precision = np.full(self.k, prior_precision, dtype=float)
        self.reward_sums = np.zeros(self.k)

    def __repr__(self):
        return (f"ThompsonSampling(arms={self.p.tolist()}, "
                f"known_precision={self.precision}, "
                f"prior_precision={self.prior_precision})")

    def pull(self):
        """Sample a mean from each posterior and pull the best sample.

        Returns - tuple of (int, float) - The chosen arm and the reward received.
        """
        samples = self.rng.normal(self.estimates, 1.0 / np.sqrt(self.post_precision))
        arm = int(np.argmax(samples))
        return arm, self._draw_reward(arm)

    def update(self, arm, reward):
        """Update the posterior of the pulled arm (known precision).

        Parameters - arm, reward
        """
        self.counts[arm] += 1
        self.reward_sums[arm] += reward
        self.post_precision[arm] = self.prior_precision + self.counts[arm] * self.precision
        self.estimates[arm] = self.precision * self.reward_sums[arm] / self.post_precision[arm]
        self._log_trial(arm, reward)

    def experiment(self, n_trials=NUMBER_OF_TRIALS):
        """Run the experiment.

        Parameters - n_trials (default 20000)
        """
        logger.info(f"Running {self.name} for {n_trials} trials")
        for _ in range(n_trials):
            arm, reward = self.pull()
            self.update(arm, reward)


def comparison(eg, ts):
    """Compare the two algorithms visually and numerically.

    saves the combined CSV (``Bandit, Reward, Algorithm``), the comparison
    plots, and logs which algorithm achieved the higher cumulative reward.

    Parameters - eg : EpsilonGreedy, ts : ThompsonSampling
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    combined = pd.concat([eg.to_dataframe(), ts.to_dataframe()], ignore_index=True)
    path = os.path.join(OUTPUT_DIR, "rewards.csv")
    combined.to_csv(path, index=False)
    logger.info(f"Combined rewards stored in {path}")

    viz = Visualization()
    viz.plot1([eg, ts])
    viz.plot2(eg, ts)

    for algo in (eg, ts):
        logger.info(f"{algo.name}: cumulative reward = {np.sum(algo.rewards):.2f}, "
                    f"cumulative regret = {np.sum(algo.regrets):.2f}")
    winner = ts if np.sum(ts.regrets) < np.sum(eg.regrets) else eg
    logger.info(f"Lower cumulative regret: {winner.name}")


if __name__=='__main__':

    eg = EpsilonGreedy(BANDIT_REWARDS)
    ts = ThompsonSampling(BANDIT_REWARDS)
    logger.info(repr(eg))
    logger.info(repr(ts))

    eg.experiment()
    ts.experiment()

    eg.report()
    ts.report()

    comparison(eg, ts)

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Implementations of the bandit algorithms
class DiscountedUCB:
    def __init__(self, n_arms, gamma=0.99):
        self.n_arms = n_arms
        self.gamma = gamma
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
        self.discounted_rewards = np.zeros(n_arms)
        self.discounted_counts = np.zeros(n_arms)
    
    def select_arm(self):
        if np.min(self.counts) == 0:
            return np.argmin(self.counts)
        
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.discounted_counts[arm] > 0:
                mean_reward = self.discounted_rewards[arm] / self.discounted_counts[arm]
            else:
                mean_reward = 0
            
            exploration = np.sqrt((2 * np.log(np.sum(self.discounted_counts))) / self.discounted_counts[arm])
            ucb_values[arm] = mean_reward + exploration
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.t += 1
        self.counts[arm] += 1
        
        self.discounted_counts *= self.gamma
        self.discounted_rewards *= self.gamma
        
        self.discounted_counts[arm] += 1
        self.discounted_rewards[arm] += reward
        
        if self.discounted_counts[arm] > 0:
            self.values[arm] = self.discounted_rewards[arm] / self.discounted_counts[arm]


class SlidingWindowUCB:
    def __init__(self, n_arms, window_size=50):
        self.n_arms = n_arms
        self.window_size = window_size
        self.rewards = [[] for _ in range(n_arms)]
        self.t = 0
    
    def select_arm(self):
        for arm in range(self.n_arms):
            if len(self.rewards[arm]) == 0:
                return arm
        
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if len(self.rewards[arm]) > 0:
                mean_reward = np.mean(self.rewards[arm])
            else:
                mean_reward = 0
            
            n_samples = len(self.rewards[arm])
            exploration = np.sqrt((2 * np.log(self.t)) / n_samples)
            ucb_values[arm] = mean_reward + exploration
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.t += 1
        self.rewards[arm].append(reward)
        
        if len(self.rewards[arm]) > self.window_size:
            self.rewards[arm].pop(0)


# Traditional UCB1 for comparison
class UCB1:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.counts = np.zeros(n_arms)
        self.values = np.zeros(n_arms)
        self.t = 0
    
    def select_arm(self):
        if np.min(self.counts) == 0:
            return np.argmin(self.counts)
        
        ucb_values = np.zeros(self.n_arms)
        for arm in range(self.n_arms):
            if self.counts[arm] > 0:
                mean_reward = self.values[arm]
                exploration = np.sqrt((2 * np.log(self.t)) / self.counts[arm])
                ucb_values[arm] = mean_reward + exploration
            else:
                ucb_values[arm] = float('inf')
        
        return np.argmax(ucb_values)
    
    def update(self, arm, reward):
        self.t += 1
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n-1) / n) * value + (1 / n) * reward


# Enhanced Non-stationary environment with configurable settings
class EnhancedBanditEnvironment:
    def __init__(self, n_arms, seed=None, env_type="irregular_change"):
        """
        Initialize the enhanced bandit environment
        
        Parameters:
        -----------
        n_arms : int
            Initial number of arms
        seed : int
            Random seed for reproducibility
        env_type : str
            Type of environment: "irregular_change" or "increasing_arms"
        """
        self.n_arms = n_arms
        self.rng = np.random.RandomState(seed)
        self.env_type = env_type
        self.current_means = self.rng.rand(n_arms)  # Initial means
        self.history_means = [self.current_means.copy()]
        
        # For "increasing_arms" environment, prepare for arm additions
        if env_type == "increasing_arms":
            # Store initial number of arms
            self.initial_n_arms = n_arms 
            # Maximum number of arms that can be added
            self.max_additional_arms = 20
    
    def pull(self, arm):
        # Return reward based on current means
        return self.rng.normal(self.current_means[arm], 0.1)
    
    def introduce_change_point(self, shift_magnitude=0.3):
        # Always shift existing arm means regardless of environment type
        shift = self.rng.uniform(-shift_magnitude, shift_magnitude, self.n_arms)
        self.current_means += shift
        self.current_means = np.clip(self.current_means, 0, 1)
        
        if self.env_type == "increasing_arms":
            # Add new arms with random means
            n_new_arms = 0  # Add 1-3 new arms
            
            # Check if we've reached maximum arms
            if self.n_arms + n_new_arms > self.initial_n_arms + self.max_additional_arms:
                n_new_arms = self.initial_n_arms + self.max_additional_arms - self.n_arms
                if n_new_arms <= 0:
                    self.history_means.append(self.current_means.copy())
                    return
            
            # Generate means for new arms
            new_means = self.rng.rand(n_new_arms)
            
            # Add new arms
            self.current_means = np.concatenate([self.current_means, new_means])
            self.n_arms += n_new_arms
        
        self.history_means.append(self.current_means.copy())
    
    def get_optimal_reward(self):
        # Return the expected reward of the best arm
        return np.max(self.current_means)
    
    def get_optimal_arm(self):
        # Return the index of the best arm
        return np.argmax(self.current_means)


# Enhanced simulation function to handle different environment types
def run_simulation(n_arms=5, n_steps=10000, seed=42, env_type="irregular_change", include_parameter_variants=True):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create environment
    env = EnhancedBanditEnvironment(n_arms, seed=seed, env_type=env_type)
    
    # Create algorithms
    algorithms = {
        'UCB1': UCB1(n_arms),
    }
    
    # Add parameter variants if requested
    if include_parameter_variants:
        algorithms.update({
            'Discounted UCB (γ=0.99)': DiscountedUCB(n_arms, gamma=0.99),
            'Discounted UCB (γ=0.95)': DiscountedUCB(n_arms, gamma=0.95),
            'Sliding Window UCB (τ=50)': SlidingWindowUCB(n_arms, window_size=50),
            'Sliding Window UCB (τ=200)': SlidingWindowUCB(n_arms, window_size=200)
        })
    
    # Track rewards and optimal arm choices
    rewards = {name: np.zeros(n_steps) for name in algorithms.keys()}
    optimal_choices = {name: np.zeros(n_steps) for name in algorithms.keys()}
    regrets = {name: np.zeros(n_steps) for name in algorithms.keys()}
    
    # Generate change points
    change_points = []
    if env_type == "increasing_arms":
        # Regular intervals of 1000 steps
        interval = 1000
        change_points = list(range(interval, n_steps, interval))
    else:
        # Random interval between 500 and 1500 steps for irregular changes
        t = 0
        while t < n_steps:
            interval = np.random.randint(500, 1500)
            t += interval
            if t < n_steps:
                change_points.append(t)
    
    # Run simulation
    for t in tqdm(range(n_steps), desc=f"Simulating {env_type} environment"):
        # Check if we need to introduce a change point
        if t in change_points:
            env.introduce_change_point()
            
            # If arms were added, update algorithms
            if env_type == "increasing_arms":
                for name, algorithm in algorithms.items():
                    # Create new algorithm instance with updated number of arms
                    if name.startswith('Discounted UCB'):
                        gamma = float(name.split('=')[1][:-1])
                        algorithms[name] = DiscountedUCB(env.n_arms, gamma=gamma)
                    elif name.startswith('Sliding Window UCB'):
                        window_size = int(name.split('=')[1][:-1])
                        algorithms[name] = SlidingWindowUCB(env.n_arms, window_size=window_size)
                    else:  # UCB1
                        algorithms[name] = UCB1(env.n_arms)
        
        # Get optimal reward for this timestep
        optimal_reward = env.get_optimal_reward()
        
        # For each algorithm, select an arm and update
        for name, algorithm in algorithms.items():
            arm = algorithm.select_arm()
            reward = env.pull(arm)
            algorithm.update(arm, reward)
            
            # Record reward
            rewards[name][t] = reward
            
            # Record regret (difference between optimal and received reward)
            regrets[name][t] = optimal_reward - reward
            
            # Record if we chose the optimal arm
            optimal_arm = env.get_optimal_arm()
            optimal_choices[name][t] = 1 if arm == optimal_arm else 0
    
    return rewards, optimal_choices, regrets, change_points, env.history_means


# Analysis and visualization function extended to handle increasing arms
def analyze_results(rewards, optimal_choices, regrets, change_points, history_means, window_size=100, env_type=""):
    # Calculate cumulative average reward
    cum_rewards = {name: np.cumsum(rewards[name]) / np.arange(1, len(rewards[name]) + 1)
                  for name in rewards.keys()}
    
    # Calculate cumulative regret
    cum_regrets = {name: np.cumsum(regrets[name]) for name in regrets.keys()}
    
    # Calculate percentage of optimal arm choices (windowed)
    windowed_optimal = {}
    for name in optimal_choices.keys():
        windowed = np.zeros(len(optimal_choices[name]) - window_size + 1)
        for i in range(len(windowed)):
            windowed[i] = np.mean(optimal_choices[name][i:i+window_size])
        windowed_optimal[name] = windowed
    
    # Create a figure with multiple subplots
    fig, axes = plt.subplots(4, 1, figsize=(14, 20), sharex=True)
    
    # Plot 1: Mean rewards over time for each arm
    ax = axes[0]
    for i, means in enumerate(history_means):
        for arm in range(len(means)):
            if i < len(history_means) - 1:
                next_point = change_points[i] if i < len(change_points) else len(rewards[list(rewards.keys())[0]])
                ax.plot([change_points[i-1] if i > 0 else 0, next_point], [means[arm], means[arm]], 
                        label=f'Arm {arm}' if i == 0 else "")
    
    for cp in change_points:
        ax.axvline(x=cp, color='r', linestyle='--', alpha=0.3)
    
    ax.set_ylabel('True Mean Reward')
    ax.set_title('True Mean Rewards with Change Points and New Arms')
    ax.legend(loc='upper right')
    
    # Plot 2: Cumulative average reward
    ax = axes[1]
    for name, cum_reward in cum_rewards.items():
        ax.plot(cum_reward, label=name)
    
    for cp in change_points:
        ax.axvline(x=cp, color='r', linestyle='--', alpha=0.3)
    
    ax.set_ylabel('Cumulative Avg Reward')
    ax.set_title('Algorithm Performance: Cumulative Average Reward')
    ax.legend(loc='lower right')
    
    # Plot 3: Cumulative regret
    ax = axes[2]
    for name, cum_regret in cum_regrets.items():
        ax.plot(cum_regret, label=name)
    
    for cp in change_points:
        ax.axvline(x=cp, color='r', linestyle='--', alpha=0.3)
    
    ax.set_ylabel('Cumulative Regret')
    ax.set_title('Algorithm Performance: Cumulative Regret')
    ax.legend(loc='upper left')
    
    # Plot 4: Percentage of optimal arm choices (windowed)
    ax = axes[3]
    for name, opt_choices in windowed_optimal.items():
        ax.plot(np.arange(window_size-1, len(optimal_choices[name])), opt_choices, label=name)
    
    for cp in change_points:
        if cp >= window_size:
            ax.axvline(x=cp, color='r', linestyle='--', alpha=0.3)
    
    ax.set_xlabel('Time steps')
    ax.set_ylabel(f'% Optimal Arms (Window={window_size})')
    ax.set_title(f'Algorithm Performance: Percentage of Optimal Arm Choices (Window Size={window_size})')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


# Enhanced main function to run simulations with different settings
def main(n_arms=5, n_steps=10000, seed=42, analysis_window=100, env_type="irregular_change):
    """
    Main function to run simulation and analysis
    
    Parameters:
    -----------
    n_arms : int
        Initial number of arms
    n_steps : int
        Number of time steps to simulate
    seed : int
        Random seed for reproducibility
    analysis_window : int
        Window size for moving average analysis
    env_type : str
        Type of environment: "irregular_change" or "increasing_arms"
    """
    print(f"Running simulation with {n_arms} initial arms in {env_type} environment for {n_steps} steps")
    
    rewards, optimal_choices, regrets, change_points, history_means = run_simulation(
        n_arms=n_arms, n_steps=n_steps, seed=seed, env_type=env_type
    )
    
    fig = analyze_results(
        rewards, optimal_choices, regrets, change_points, history_means, 
        window_size=analysis_window, env_type=env_type
    )
    
    # Print summary statistics
    print("Change points occurred at time steps:", change_points)
    print("\nAverage rewards over the entire simulation:")
    for name in rewards.keys():
        print(f"{name}: {np.mean(rewards[name]):.4f}")
    
    print("\nAverage percentage of optimal arm choices:")
    for name in optimal_choices.keys():
        print(f"{name}: {np.mean(optimal_choices[name]) * 100:.2f}%")
    
    print("\nFinal cumulative regret:")
    for name in regrets.keys():
        print(f"{name}: {np.sum(regrets[name]):.1f}")
    
    plt.show()
    return rewards, optimal_choices, regrets, change_points, history_means


# Enhanced parameter sensitivity analysis for both environment types
def parameter_sensitivity_analysis(n_arms=5, n_steps=5000, seed=42, env_type="irregular_change"):
    """
    Run multiple simulations with different parameter settings to analyze sensitivity
    
    Parameters:
    -----------
    n_arms : int
        Initial number of arms
    n_steps : int
        Number of time steps to simulate
    seed : int
        Random seed for reproducibility
    env_type : str
        Type of environment: "irregular_change" or "increasing_arms"
    """
    print(f"Running parameter sensitivity analysis in {env_type} environment with {n_arms} initial arms")
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create environment with fixed change points for fair comparison
    env = EnhancedBanditEnvironment(n_arms, seed=seed, env_type=env_type)
    
    # Generate change points
    change_points = []
    if env_type == "increasing_arms":
        # Regular intervals of 1000 steps
        interval = 1000
        change_points = list(range(interval, n_steps, interval))
    else:
        # Random interval between 500 and 1500 steps for irregular changes
        t = 0
        while t < n_steps:
            interval = np.random.randint(500, 1500)
            t += interval
            if t < n_steps:
                change_points.append(t)
    
    # Different parameter settings to test
    gamma_values = [0.9, 0.95, 0.98, 0.99, 0.995]
    window_sizes = [20, 50, 100, 200, 500]
    
    # Create all algorithms
    algorithms = {'UCB1': UCB1(n_arms)}
    
    # Add Discounted UCB with different gamma values
    for gamma in gamma_values:
        algorithms[f'D-UCB (γ={gamma})'] = DiscountedUCB(n_arms, gamma=gamma)
    
    # Add Sliding Window UCB with different window sizes
    for window in window_sizes:
        algorithms[f'SW-UCB (τ={window})'] = SlidingWindowUCB(n_arms, window_size=window)
    
    # Track cumulative regret for each algorithm
    total_regrets = {name: 0 for name in algorithms.keys()}
    
    # Run simulation
    for t in tqdm(range(n_steps), desc=f"Analyzing parameter sensitivity in {env_type} environment"):
        # Check if we need to introduce a change point
        if t in change_points:
            env.introduce_change_point()
            
            # If arms were added, update algorithms
            if env_type == "increasing_arms":
                for name in list(algorithms.keys()):
                    # Create new algorithm instance with updated number of arms
                    if name.startswith('D-UCB'):
                        gamma = float(name.split('=')[1][:-1])
                        algorithms[name] = DiscountedUCB(env.n_arms, gamma=gamma)
                    elif name.startswith('SW-UCB'):
                        window = int(name.split('=')[1][:-1])
                        algorithms[name] = SlidingWindowUCB(env.n_arms, window_size=window)
                    else:  # UCB1
                        algorithms[name] = UCB1(env.n_arms)
        
        # Get optimal reward for this timestep
        optimal_reward = env.get_optimal_reward()
        
        # For each algorithm, select an arm and update
        for name, algorithm in algorithms.items():
            arm = algorithm.select_arm()
            reward = env.pull(arm)
            algorithm.update(arm, reward)
            
            # Accumulate regret
            total_regrets[name] += optimal_reward - reward
    
    # Plot results by algorithm type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Discounted UCB with different gamma values
    ducb_results = {k: v for k, v in total_regrets.items() if 'D-UCB' in k}
    gamma_values = [float(k.split('=')[1][:-1]) for k in ducb_results.keys()]
    regret_values = list(ducb_results.values())
    
    # Sort by gamma value
    sorted_indices = np.argsort(gamma_values)
    gamma_values = [gamma_values[i] for i in sorted_indices]
    regret_values = [regret_values[i] for i in sorted_indices]
    
    ax1.plot(gamma_values, regret_values, 'o-')
    ax1.set_xlabel('Gamma (γ) Value')
    ax1.set_ylabel('Total Regret')
    ax1.set_title('Parameter Sensitivity: Discounted UCB')
    ax1.grid(True)
    
    # Plot 2: Sliding Window UCB with different window sizes
    swucb_results = {k: v for k, v in total_regrets.items() if 'SW-UCB' in k}
    window_values = [int(k.split('=')[1][:-1]) for k in swucb_results.keys()]
    regret_values = list(swucb_results.values())
    
    # Sort by window size
    sorted_indices = np.argsort(window_values)
    window_values = [window_values[i] for i in sorted_indices]
    regret_values = [regret_values[i] for i in sorted_indices]
    
    ax2.plot(window_values, regret_values, 'o-')
    ax2.set_xlabel('Window Size (τ)')
    ax2.set_ylabel('Total Regret')
    ax2.set_title('Parameter Sensitivity: Sliding Window UCB')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Also print the UCB1 baseline
    print(f"UCB1 baseline regret: {total_regrets['UCB1']:.1f}")
    
    # Return the results
    return fig, total_regrets, change_points


# Compare different initial numbers of arms
def compare_arm_counts(n_steps=5000, seed=42):
    """Compare performance with different initial numbers of arms"""
    # Different initial arm counts to test
    initial_arm_counts = [3, 5, 10, 20, 50]
    
    # Track results
    results = {}
    
    # Run simulations for each initial arm count
    for n_arms in initial_arm_counts:
        print(f"Testing with {n_arms} initial arms...")
        rewards, _, regrets, _, _ = run_simulation(
            n_arms=n_arms, 
            n_steps=n_steps, 
            seed=seed, 
            env_type="increasing_arms",
            include_parameter_variants=False  # Only use UCB1 for comparison
        )
        
        # Store average reward and cumulative regret
        results[n_arms] = {
            'avg_reward': np.mean(rewards['UCB1']),
            'cum_regret': np.sum(regrets['UCB1'])
        }
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot average rewards
    ax1.plot([str(n) for n in initial_arm_counts], 
            [results[n]['avg_reward'] for n in initial_arm_counts], 'o-')
    ax1.set_xlabel('Initial Number of Arms')
    ax1.set_ylabel('Average Reward')
    ax1.set_title('Effect of Initial Number of Arms on Average Reward')
    ax1.grid(True)
    
    # Plot cumulative regret
    ax2.plot([str(n) for n in initial_arm_counts], 
            [results[n]['cum_regret'] for n in initial_arm_counts], 'o-')
    ax2.set_xlabel('Initial Number of Arms')
    ax2.set_ylabel('Cumulative Regret')
    ax2.set_title('Effect of Initial Number of Arms on Cumulative Regret')
    ax2.grid(True)
    
    plt.tight_layout()
    return fig, results


# If running as a script
if __name__ == "__main__":
    # Configuration settings - modify these values to change the simulation
    CONFIG = {
        "env_type": "irregular_change",  # Set to "increasing_arms" or "irregular_change"
        "n_arms": 10,                    # Initial number of arms
        "n_steps": 10000,                # Number of simulation steps
        "seed": 42,                      # Random seed for reproducibility
        "compare_arm_counts": False      # Set to True to compare different numbers of initial arms
    }
    
    print(f"Running simulation with configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    if CONFIG["compare_arm_counts"]:
        # Run comparison of different initial arm counts
        compare_arm_counts(n_steps=CONFIG["n_steps"], seed=CONFIG["seed"])
    else:
        # Run main simulation
        main(
            n_arms=CONFIG["n_arms"], 
            n_steps=CONFIG["n_steps"], 
            seed=CONFIG["seed"], 
            analysis_window=100, 
            env_type=CONFIG["env_type"]
        )
        
        # Run parameter sensitivity analysis
        parameter_sensitivity_analysis(
            n_arms=CONFIG["n_arms"], 
            n_steps=CONFIG["n_steps"]//2,  # Use fewer steps for sensitivity analysis
            seed=CONFIG["seed"], 
            env_type=CONFIG["env_type"]
        )
    
    plt.show()
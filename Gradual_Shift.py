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


# Gradually Shifting Non-stationary environment
class GraduallyShiftingEnvironment:
    def __init__(self, n_arms, seed=None, env_type="increased_prices", drift_rate=0.0002):
        """
        Initialize a gradually shifting bandit environment
        
        Parameters:
        -----------
        n_arms : int
            Number of arms (price options)
        seed : int
            Random seed for reproducibility
        env_type : str
            Type of environment: "continuous_drift" or "increased_prices"
        drift_rate : float
            Rate of gradual shift per timestep (default: 0.0002)
        """
        self.n_arms = n_arms
        self.rng = np.random.RandomState(seed)
        self.env_type = env_type
        self.drift_rate = drift_rate
        
        # Initialize means
        self.current_means = self.rng.rand(n_arms)  # Initial means
        self.target_means = self.current_means.copy()  # Target means to drift toward
        
        # Historical data for visualization
        self.history_means = []
        self.record_means(0) # Record initial state
        
        # For price sensitivity environment
        if env_type == "increased_prices":
            # Each arm represents a different price point
            self.base_rewards = self.rng.uniform(0.5, 1.5, n_arms)
            # Update means based on base rewards
            self._update_reward_based_means()
    
    def _update_reward_based_means(self):
        """Update means based on base rewards for increased_prices environment"""
        if self.env_type == "increased_prices":
            # Calculate expected reward for each price point
            for arm in range(self.n_arms):
                # Simple reward calculation based on base reward
                self.current_means[arm] = self.base_rewards[arm]
            
            # Set new target means 
            self.target_means = self.current_means.copy()
    
    def pull(self, arm):
        # Return reward based on current means
        return self.rng.normal(self.current_means[arm], 0.1)
    
    def update(self, t):
        """Update environment at each timestep with gradual drift"""
        # For continuous drift, update means toward target
        if t % 100 == 0:  # Record mean values periodically for visualization
            self.record_means(t)
        
        # Check if we should set new target means
        if self.env_type == "continuous_drift":
            # Periodically generate completely new target means
            if t % 2000 == 0 and t > 0:
                # Generate new target means to drift toward
                self.target_means = self.rng.rand(self.n_arms)
                # Ensure targets stay within [0, 1]
                self.target_means = np.clip(self.target_means, 0, 1)
        
        elif self.env_type == "increased_prices":
            # Periodically change customer preferences
            if t % 2000 == 0 and t > 0:
                # Gradually change base rewards
                self.base_rewards += self.rng.uniform(-0.2, 0.2, self.n_arms)
                self.base_rewards = np.clip(self.base_rewards, 0.3, 1.7)
                # Update target means based on new base rewards
                self._update_reward_based_means()
        
        # Move current means toward target means
        drift_magnitude = self.drift_rate * np.abs(self.target_means - self.current_means)
        direction = np.sign(self.target_means - self.current_means)
        self.current_means += direction * drift_magnitude
    
    def record_means(self, t):
        """Record current means for visualization"""
        self.history_means.append((t, self.current_means.copy()))
    
    def get_optimal_reward(self):
        # Return the expected reward of the best arm
        return np.max(self.current_means)
    
    def get_optimal_arm(self):
        # Return the index of the best arm
        return np.argmax(self.current_means)


# Enhanced simulation function to handle gradual shifts
def run_simulation(n_arms=5, n_steps=10000, seed=42, env_type="increased_prices", 
                   drift_rate=0.0002, include_parameter_variants=True):
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Create environment
    env = GraduallyShiftingEnvironment(n_arms, seed=seed, env_type=env_type, drift_rate=drift_rate)
    
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
    
    # Change points (for visualization)
    change_points = list(range(0, n_steps, 2000))[1:]  # Every 2000 steps, but exclude 0
    
    # Run simulation
    for t in tqdm(range(n_steps), desc=f"Simulating {env_type} environment with gradual shifts"):
        # Update environment (gradual drift)
        env.update(t)
        
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
    
    # Add final record point
    env.record_means(n_steps-1)
    
    return rewards, optimal_choices, regrets, change_points, env.history_means


# Analysis and visualization function for gradual shift environment
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
    
    # Plot 1: Mean rewards over time for each arm (gradual changes)
    ax = axes[0]
    
    # Extract timesteps and means arrays from history
    timesteps = [t for t, _ in history_means]
    means_history = [means for _, means in history_means]
    
    # Plot gradual changes for each arm
    for arm in range(len(means_history[0])):
        arm_means = [means[arm] for means in means_history]
        ax.plot(timesteps, arm_means, label=f'{"Price " if env_type=="increased_prices" else "Arm "}{arm}')
    
    # Mark major change points with vertical lines
    for cp in change_points:
        ax.axvline(x=cp, color='r', linestyle='--', alpha=0.3)
    
    ax.set_ylabel('True Mean Reward')
    if env_type == "increased_prices":
        ax.set_title('Expected Revenue for Different Price Points with Gradual Shifts')
    else:
        ax.set_title('True Mean Rewards with Gradual Shifts')
    ax.legend(loc='upper right')
    
    # Plot 2: Cumulative average reward
    ax = axes[1]
    for name, cum_reward in cum_rewards.items():
        ax.plot(cum_reward, label=name)
    
    for cp in change_points:
        ax.axvline(x=cp, color='r', linestyle='--', alpha=0.3)
    
    ax.set_ylabel('Cumulative Avg Reward')
    if env_type == "increased_prices":
        ax.set_title('Algorithm Performance: Cumulative Average Revenue')
    else:
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
    ax.set_ylabel(f'% Optimal {"Price" if env_type=="increased_prices" else "Arms"} (Window={window_size})')
    ax.set_title(f'Algorithm Performance: Percentage of Optimal {"Price" if env_type=="increased_prices" else "Arm"} Choices (Window Size={window_size})')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return fig


# Enhanced main function
def main(n_arms=5, n_steps=10000, seed=42, analysis_window=100, env_type="increased_prices", drift_rate=0.0002):
    """
    Main function to run simulation and analysis with gradual shifts
    
    Parameters:
    -----------
    n_arms : int
        Number of arms (price options)
    n_steps : int
        Number of time steps to simulate
    seed : int
        Random seed for reproducibility
    analysis_window : int
        Window size for moving average analysis
    env_type : str
        Type of environment: "continuous_drift" or "increased_prices"
    drift_rate : float
        Rate of gradual drift per timestep
    """
    print(f"Running simulation with {n_arms} arms in {env_type} environment for {n_steps} steps")
    print(f"Using drift rate of {drift_rate} per timestep")
    
    rewards, optimal_choices, regrets, change_points, history_means = run_simulation(
        n_arms=n_arms, n_steps=n_steps, seed=seed, env_type=env_type, drift_rate=drift_rate
    )
    
    fig = analyze_results(
        rewards, optimal_choices, regrets, change_points, history_means, 
        window_size=analysis_window, env_type=env_type
    )
    
    # Print summary statistics
    print("Major change direction points occurred at time steps:", change_points)
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


# Parameter sensitivity analysis for drift rates
def drift_rate_sensitivity_analysis(n_arms=5, n_steps=5000, seed=42, env_type="increased_prices"):
    """
    Analyze the impact of different drift rates on algorithm performance
    
    Parameters:
    -----------
    n_arms : int
        Number of arms (price options)
    n_steps : int
        Number of time steps to simulate
    seed : int
        Random seed for reproducibility
    env_type : str
        Type of environment: "continuous_drift" or "increased_prices"
    """
    print(f"Running drift rate sensitivity analysis in {env_type} environment")
    
    # Different drift rates to test
    drift_rates = [0.0001, 0.0002, 0.0005, 0.001, 0.002]
    
    # Store results for each drift rate
    results = {}
    
    # Run simulations for each drift rate
    for drift_rate in drift_rates:
        print(f"Testing drift rate: {drift_rate}")
        
        rewards, _, regrets, _, _ = run_simulation(
            n_arms=n_arms, 
            n_steps=n_steps, 
            seed=seed, 
            env_type=env_type,
            drift_rate=drift_rate,
            include_parameter_variants=True
        )
        
        # Store results
        results[drift_rate] = {
            name: {
                'avg_reward': np.mean(rewards[name]),
                'cum_regret': np.sum(regrets[name])
            }
            for name in rewards.keys()
        }
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Get algorithm names
    alg_names = list(results[drift_rates[0]].keys())
    
    # Plot for each algorithm type
    for i, alg_name in enumerate(['UCB1', 'Discounted UCB (γ=0.99)', 'Sliding Window UCB (τ=50)']):
        if alg_name not in alg_names:
            continue
            
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Plot average reward vs drift rate
        ax.plot(drift_rates, 
                [results[rate][alg_name]['avg_reward'] for rate in drift_rates], 
                'o-', label='Avg Reward')
        
        # Create second y-axis for regret
        ax2 = ax.twinx()
        ax2.plot(drift_rates, 
                [results[rate][alg_name]['cum_regret'] for rate in drift_rates], 
                'r^-', label='Cum Regret')
        
        ax.set_xlabel('Drift Rate')
        ax.set_ylabel('Average Reward', color='blue')
        ax2.set_ylabel('Cumulative Regret', color='red')
        
        ax.set_title(f'Impact of Drift Rate on {alg_name}')
        ax.grid(True)
    
    # Empty subplot (if odd number of algorithms)
    if len(alg_names) < 4:
        row, col = 1, 1
        axes[row, col].axis('off')
    
    plt.tight_layout()
    return fig, results


# If running as a script
if __name__ == "__main__":
    # Configuration settings - modify these values to change the simulation
    CONFIG = {
        "env_type": "increased_prices",  # "continuous_drift" or "increased_prices"
        "n_arms": 10,                    # Number of price options
        "n_steps": 10000,                # Number of simulation steps
        "seed": 50,                      # Random seed for reproducibility
        "drift_rate": 0.0002,            # Rate of gradual change (0.0001 to 0.002)
        "run_sensitivity_analysis": True # Whether to run drift rate sensitivity analysis
    }
    
    print(f"Running simulation with configuration:")
    for key, value in CONFIG.items():
        print(f"  {key}: {value}")
    
    # Run main simulation
    main(
        n_arms=CONFIG["n_arms"], 
        n_steps=CONFIG["n_steps"], 
        seed=CONFIG["seed"], 
        analysis_window=100, 
        env_type=CONFIG["env_type"],
        drift_rate=CONFIG["drift_rate"]
    )
    
    # Run drift rate sensitivity analysis if requested
    if CONFIG["run_sensitivity_analysis"]:
        drift_rate_sensitivity_analysis(
            n_arms=CONFIG["n_arms"], 
            n_steps=CONFIG["n_steps"]//2,  # Use fewer steps for sensitivity analysis
            seed=CONFIG["seed"], 
            env_type=CONFIG["env_type"]
        )
    
    plt.show()

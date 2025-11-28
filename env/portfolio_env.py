import gym
import pandas as pd
import torch
from gym import spaces
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv

class StockPortfolioEnv(gym.Env):
    def __init__(self, args, corr=None, ts_features=None, features=None,
                 ind=None, pos=None, neg=None, returns=None, pyg_data=None,
                 benchmark_return=None, mode="train", reward_net=None, device='cuda:0',
                 ind_yn=False, pos_yn=False, neg_yn=False, risk_profile=None):
        super(StockPortfolioEnv, self).__init__()
        self.current_step = 0
        self.max_step = returns.shape[0] - 1
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.peak_value = 1.0  # Track peak wealth for drawdown calculation
        self.net_value_s = [1.0]
        self.daily_return_s = [0.0]
        self.num_stocks = returns.shape[-1]
        self.benchmark_return = benchmark_return
        
        # Track portfolio weights history
        self.weights_history = []  # List of weight vectors at each step
        self.dates = []  # List of dates/step indices

        self.corr_tensor = corr
        self.ts_features_tensor = ts_features
        self.features_tensor = features
        self.ind_tensor = ind
        self.pos_tensor = pos
        self.neg_tensor = neg
        self.pyg_data_batch = pyg_data
        self.ror_batch = returns
        self.ind_yn = ind_yn
        self.pos_yn = pos_yn
        self.neg_yn = neg_yn
        self.risk_profile = risk_profile or {}
        self.risk_score = float(self.risk_profile.get('risk_score', getattr(args, 'risk_score', 0.5)))
        self.max_weight_cap = self.risk_profile.get('max_weight', None)
        self.min_weight_floor = self.risk_profile.get('min_weight', 0.0)
        # Conservative users (low score) get higher temperature to spread allocations
        self.action_temperature = 0.2 + 2.3 * (1.0 - self.risk_score)

        # Action space: continuous weights for each stock
        # Output raw scores, will be normalized via softmax to sum to 1
        # SB3 requires finite bounds, use large range [-10, 10] which is sufficient for softmax
        self.action_space = spaces.Box(
            low=-10.0,  # Scores before softmax
            high=10.0,  # After softmax, will become valid probabilities in [0, 1]
            shape=(self.num_stocks,),
            dtype=np.float32
        )

        # Optional: store for compatibility if needed
        self.top_k = max(1, int(0.1 * self.num_stocks))  # Can be used for constraints later

        # Partially observable setting
        # Observation space: stock features plus relation graphs
        # HGAT expects flattened input: [ind_matrix, pos_matrix, neg_matrix, features]
        obs_len = 0
        if self.ind_yn:
            obs_len += self.num_stocks * self.num_stocks  # Flattened industry matrix
        else:
            obs_len += self.num_stocks * self.num_stocks  # Zeros placeholder
        if self.pos_yn:
            obs_len += self.num_stocks * self.num_stocks  # Flattened momentum matrix
        else:
            obs_len += self.num_stocks * self.num_stocks  # Zeros placeholder
        if self.neg_yn:
            obs_len += self.num_stocks * self.num_stocks  # Flattened reversal matrix
        else:
            obs_len += self.num_stocks * self.num_stocks  # Zeros placeholder
        obs_len += self.num_stocks * args.input_dim  # Flattened features
        
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(obs_len,),  # 1D flattened observation
                                            dtype=np.float32)
        self.mode = mode
        self.reward_net = reward_net  # Inject IRL reward network
        self.device = device

    def load_observation(self, ts_yn=False, ind_yn=False, pos_yn=False, neg_yn=False):
        # Stable-Baselines3's DummyVecEnv expects NumPy observations
        if torch.isnan(self.features_tensor).any():
            print("Detected NaNs in feature tensor!")
        features = self.features_tensor[self.current_step].cpu().numpy()  # [num_stocks, 6]
        corr_matrix = self.corr_tensor[self.current_step].cpu().numpy()
        ind_matrix = self.ind_tensor[self.current_step].cpu().numpy()  # [num_stocks, num_stocks]
        pos_matrix = self.pos_tensor[self.current_step].cpu().numpy()  # [num_stocks, num_stocks]
        neg_matrix = self.neg_tensor[self.current_step].cpu().numpy()  # [num_stocks, num_stocks]
        
        # HGAT expects: [ind_matrix, pos_matrix, neg_matrix, features] all flattened and concatenated
        # Reshape matrices to [num_stocks, num_stocks] and features to [num_stocks, 6]
        # Then flatten in the correct order for HGAT
        obs_parts = []
        if ind_yn:
            obs_parts.append(ind_matrix.flatten())  # [num_stocks * num_stocks]
        else:
            obs_parts.append(np.zeros(self.num_stocks * self.num_stocks))
            
        if pos_yn:
            obs_parts.append(pos_matrix.flatten())  # [num_stocks * num_stocks]
        else:
            obs_parts.append(np.zeros(self.num_stocks * self.num_stocks))
            
        if neg_yn:
            obs_parts.append(neg_matrix.flatten())  # [num_stocks * num_stocks]
        else:
            obs_parts.append(np.zeros(self.num_stocks * self.num_stocks))
        
        obs_parts.append(features.flatten())  # [num_stocks * 6]
        
        # Concatenate all parts into single vector
        obs = np.concatenate(obs_parts)  # Total: 3*num_stocks^2 + num_stocks*6
        
        self.observation = obs
        self.ror = self.ror_batch[self.current_step].cpu()


    def reset(self):
        self.current_step = 0
        self.done = False
        self.reward = 0.0
        self.net_value = 1.0
        self.peak_value = 1.0  # Reset peak value
        self.net_value_s = [1.0]
        self.daily_return_s = [0.0]
        self.weights_history = []  # Reset weights history
        self.dates = []  # Reset dates
        self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)
        return self.observation

    def seed(self, seed):
        return np.random.seed(seed)

    def step(self, actions):
        self.done = self.current_step == self.max_step
        if self.done:
            if self.mode == "test":
                print("=================================")
                print(f"net_values:{self.net_value_s}")
                metrics, benchmark_metrics = self.evaluate()

                print("ARR: ", metrics.get("arr"))
                print("AVOL: ", metrics.get("avol"))
                print("Sharpe: ", metrics.get("sharpe"))
                print("MDD: ", metrics.get("mdd"))
                print("CR: ", metrics.get("cr"))
                print("IR: ", metrics.get("ir"))
                if benchmark_metrics:
                    print("Benchmark ARR: ", benchmark_metrics.get("arr"))
                    print("Benchmark AVOL: ", benchmark_metrics.get("avol"))
                    print("Benchmark Sharpe: ", benchmark_metrics.get("sharpe"))
                    print("Benchmark MDD: ", benchmark_metrics.get("mdd"))
                    print("Benchmark CR: ", benchmark_metrics.get("cr"))
                print("=================================")
        else:
            # load s'
            self.current_step += 1
            self.load_observation(ind_yn=self.ind_yn, pos_yn=self.pos_yn, neg_yn=self.neg_yn)
            
            # Continuous action space: normalize to portfolio weights
            # actions: [num_stocks] raw scores from policy
            # Apply softmax to ensure weights sum to 1 and are non-negative
            action_scores = np.array(actions).flatten()
            
            # Softmax normalization: w_i = exp(a_i) / sum(exp(a_j))
            exp_actions = np.exp((action_scores - np.max(action_scores)) / self.action_temperature)  # Temperature controls concentration
            weights = exp_actions / exp_actions.sum()
            weights = self._apply_risk_constraints(weights)
            
            # Store weights for this step
            self.weights_history.append(weights.copy())
            self.dates.append(self.current_step)
            
            # Optional: Apply concentration penalty or top-K constraint
            # For now, allow full continuous allocation
            
            if self.mode == "test":
                print(f"Step {self.current_step}")
                print(f"Weights (top 5): {sorted(zip(range(len(weights)), weights), key=lambda x: x[1], reverse=True)[:5]}")
                print(f"Weight sum: {weights.sum():.6f}")
                print(f"Non-zero allocations: {(weights > 0.01).sum()}/{len(weights)}")

            # Use the IRL reward network when available
            if self.reward_net is not None:
                # self.observation is already flattened [obs_len]
                state_tensor = torch.FloatTensor(self.observation).to(self.device)  # Current state
                # Pass actual weights (continuous) instead of multi-hot
                action_tensor = torch.FloatTensor(weights).to(self.device)  # Action (weight vector)
                
                # Pass wealth information for drawdown calculation
                wealth_info = torch.FloatTensor([self.net_value, self.peak_value]).to(self.device)
                
                with torch.no_grad():
                    self.reward = self.reward_net(
                        state_tensor,
                        action_tensor,
                        wealth_info,
                        risk_score=self.risk_score
                    ).mean().cpu().item()
            else:
                # Portfolio return: weighted sum of individual stock returns
                self.reward = np.dot(weights, np.array(self.ror))

            self.net_value *= (1 + self.reward)
            self.peak_value = max(self.peak_value, self.net_value)  # Update peak value
            self.daily_return_s.append(self.reward)
            self.net_value_s.append(self.net_value)

        return self.observation, self.reward, self.done, {}

    def _apply_risk_constraints(self, weights):
        """Apply simple weight caps/floors derived from the risk profile."""
        if self.max_weight_cap is not None and self.max_weight_cap > 0:
            clipped = np.minimum(weights, self.max_weight_cap)
            # Optional floor to avoid zeroing everything for aggressive users
            if self.min_weight_floor > 0:
                clipped = np.maximum(clipped, self.min_weight_floor)
            total = clipped.sum()
            if total > 1e-8:
                return clipped / total
            return np.ones_like(weights) / len(weights)
        return weights

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs

    def get_df_net_value(self):
        df_net_value = pd.DataFrame(self.net_value_s)
        df_net_value.columns = ["net_value"]
        return df_net_value

    def get_df_daily_return(self):
        df_daily_return = pd.DataFrame(self.daily_return_s)
        df_daily_return.columns = ["daily_return"]
        return df_daily_return
    
    def get_weights_history(self):
        """Return portfolio weights history as numpy array.
        
        Returns:
            np.ndarray: Shape [num_steps, num_stocks] containing portfolio weights
        """
        if not self.weights_history:
            return np.array([])
        return np.array(self.weights_history)
    
    def get_weights_dataframe(self, tickers=None):
        """Return portfolio weights as a DataFrame.
        
        Args:
            tickers: Optional list of ticker symbols. If None, uses stock indices.
            
        Returns:
            pd.DataFrame: Columns are tickers/indices, rows are time steps
        """
        weights_array = self.get_weights_history()
        if weights_array.size == 0:
            return pd.DataFrame()
        
        if tickers is not None:
            if len(tickers) != weights_array.shape[1]:
                raise ValueError(f"Number of tickers ({len(tickers)}) doesn't match "
                               f"number of stocks ({weights_array.shape[1]})")
            columns = tickers
        else:
            columns = [f"stock_{i}" for i in range(weights_array.shape[1])]
        
        df = pd.DataFrame(weights_array, columns=columns)
        df.insert(0, 'step', self.dates)
        return df

    def _compute_metrics(self, series: pd.Series) -> dict:
        metrics = {"arr": 0.0, "avol": 0.0, "sharpe": 0.0, "mdd": 0.0, "cr": 0.0}
        if series is None or len(series) == 0:
            return metrics

        metrics["arr"] = (1 + series.mean()) ** 252 - 1
        metrics["avol"] = series.std() * (252 ** 0.5)
        metrics["sharpe"] = (252 ** 0.5) * series.mean() / series.std() if series.std() != 0 else 0.0

        cumulative = (1 + series).cumprod()
        running_max = cumulative.cummax()
        drawdown = cumulative / running_max - 1
        metrics["mdd"] = drawdown.min() if not drawdown.empty else 0.0
        metrics["cr"] = metrics["arr"] / abs(metrics["mdd"]) if metrics["mdd"] != 0 else 0.0
        return metrics

    def evaluate(self):
        df_daily_return = self.get_df_daily_return()
        portfolio_returns = df_daily_return["daily_return"].reset_index(drop=True)

        # Drop the initial zero placeholder to align with benchmark length when present
        if len(portfolio_returns) > 0 and portfolio_returns.iloc[0] == 0:
            portfolio_returns = portfolio_returns.iloc[1:].reset_index(drop=True)

        benchmark_series = None
        if self.benchmark_return is not None:
            if isinstance(self.benchmark_return, pd.DataFrame):
                if "daily_return" in self.benchmark_return.columns:
                    benchmark_series = self.benchmark_return["daily_return"]
                elif "return" in self.benchmark_return.columns:
                    benchmark_series = self.benchmark_return["return"]
            elif isinstance(self.benchmark_return, pd.Series):
                benchmark_series = self.benchmark_return
            else:
                benchmark_series = pd.Series(self.benchmark_return)

            if benchmark_series is not None:
                benchmark_series = pd.Series(benchmark_series).dropna().reset_index(drop=True)

        portfolio_metrics = self._compute_metrics(portfolio_returns)
        benchmark_metrics = {}

        if benchmark_series is not None and not benchmark_series.empty and not portfolio_returns.empty:
            aligned_len = min(len(portfolio_returns), len(benchmark_series))
            portfolio_aligned = portfolio_returns.iloc[:aligned_len]
            benchmark_aligned = benchmark_series.iloc[:aligned_len]

            benchmark_metrics = self._compute_metrics(benchmark_aligned)

            excess_return = portfolio_aligned - benchmark_aligned
            if excess_return.std() != 0:
                portfolio_metrics["ir"] = excess_return.mean() / excess_return.std() * (252 ** 0.5)
            else:
                portfolio_metrics["ir"] = 0.0
        else:
            portfolio_metrics["ir"] = 0.0

        return portfolio_metrics, benchmark_metrics

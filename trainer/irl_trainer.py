import pandas as pd
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from torch_geometric.data import DataLoader

from env.portfolio_env import StockPortfolioEnv
from trainer.evaluation_utils import (
    aggregate_metric_records,
    apply_promotion_gate,
    create_metric_record,
    persist_metrics,
)
from utils.ticker_mapping import (
    get_ticker_mapping_for_period,
)
from gen_data.gen_expert_ensemble import (
    generate_expert_trajectories,
    load_expert_trajectories,
    save_expert_trajectories,
)


class RewardNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(RewardNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action, wealth_info=None, risk_score=None):
        if state.dim() > 2:
            state = state.view(state.size(0), -1)
        state = state.squeeze()
        action = action.unsqueeze(1)
        x = torch.cat([state, action], dim=1)
        return self.fc(x)


class MaxEntIRL:
    def __init__(self, reward_net, expert_data, lr=1e-3, risk_score: float = 0.5):
        self.reward_net = reward_net
        self.expert_data = expert_data
        self.optimizer = torch.optim.Adam(reward_net.parameters(), lr=lr)
        self.risk_score = float(risk_score)

    def train(self, agent_env, model, num_epochs=50, batch_size=32, device='cuda:0'):
        for epoch in range(num_epochs):
            agent_trajectories = self._sample_trajectories(agent_env, model, batch_size=batch_size, device=device)
            expert_rewards = self._calculate_rewards(self.expert_data, device)
            agent_rewards = self._calculate_rewards(agent_trajectories, device)

            expert_mean = expert_rewards.mean()
            agent_logsumexp = torch.logsumexp(agent_rewards, dim=0)
            loss = -(expert_mean - agent_logsumexp)
            reward_gap = expert_mean - agent_logsumexp

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            print(f"Train IRL Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}, "
                  f"Expert: {expert_mean.item():.4f}, Agent: {agent_logsumexp.item():.4f}, "
                  f"Gap: {reward_gap.item():.4f}")

    def _sample_trajectories(self, env, model, batch_size, device):
        trajectories = []
        obs = env.reset()
        num_stocks = env.action_space.shape[0]

        for _ in range(batch_size):
            action, _ = model.predict(obs)
            next_obs, reward, done, _ = env.step(action)

            action_scores = action[0]
            exp_actions = np.exp(action_scores - np.max(action_scores))
            action_weights = exp_actions / exp_actions.sum()

            trajectories.append((obs.copy(), action_weights))
            obs = next_obs
            if done:
                obs = env.reset()
        return trajectories

    def _calculate_rewards(self, trajectories, device):
        rewards = []
        for state, action in trajectories:
            if isinstance(state, np.ndarray) and state.ndim > 1 and state.shape[0] == 1:
                state = state.squeeze(0)
            state_tensor = torch.FloatTensor(state).to(device)
            action_tensor = torch.FloatTensor(action).to(device)
            reward = self.reward_net(
                state_tensor.unsqueeze(0),
                action_tensor.unsqueeze(0),
                wealth_info=None,
                risk_score=self.risk_score,
            )
            rewards.append(reward)
        return torch.cat(rewards)


class MultiRewardNetwork(nn.Module):
    def __init__(self, input_dim, num_stocks, hidden_dim=64,
                 ind_yn=False, pos_yn=False, neg_yn=False,
                 use_drawdown=True, dd_kappa=0.05, dd_tau=0.01,
                 dd_base_weight=1.0, dd_risk_factor=1.0,
                 lookback=30):
        super().__init__()
        self.num_stocks = num_stocks
        self.input_dim = input_dim
        self.use_drawdown = use_drawdown
        self.dd_base_weight = dd_base_weight
        self.dd_risk_factor = dd_risk_factor
        self.dd_tau = dd_tau
        self.dd_kappa_min = 0.02
        self.dd_kappa_max = 0.10

        adj_size = num_stocks * num_stocks
        self.feature_dims = {
            'ind': adj_size if ind_yn else 0,
            'pos': adj_size if pos_yn else 0,
            'neg': adj_size if neg_yn else 0,
            'base': num_stocks * lookback * input_dim
        }

        self.encoders = nn.ModuleDict()
        for feat, dim in self.feature_dims.items():
            if dim > 0:
                self.encoders[feat] = nn.Sequential(
                    nn.Linear(dim + num_stocks, hidden_dim),
                    nn.ReLU()
                )

        if self.use_drawdown:
            self.dd_encoder = nn.Sequential(
                nn.Linear(2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )

        active_feats = [k for k, v in self.feature_dims.items() if v > 0]
        self.num_rewards = len(active_feats)
        if self.use_drawdown:
            self.num_rewards += 1
        self.weights = nn.Parameter(torch.ones(self.num_rewards))

    def forward(self, state, action, wealth_info=None, risk_score=None):
        batch_size = state.shape[0] if state.dim() > 1 else 1
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if wealth_info is not None and wealth_info.dim() == 1:
            wealth_info = wealth_info.unsqueeze(0)

        if risk_score is None:
            risk_score = torch.ones(batch_size, 1, device=state.device) * 0.5
        elif isinstance(risk_score, (int, float)):
            risk_score = torch.ones(batch_size, 1, device=state.device) * risk_score
        elif risk_score.dim() == 0:
            risk_score = risk_score.view(1, 1).expand(batch_size, 1)
        elif risk_score.dim() == 1:
            risk_score = risk_score.unsqueeze(1)

        # Dynamically infer base feature length from state to avoid shape mismatches
        adj_size = self.num_stocks * self.num_stocks
        expected_adj_total = 3 * adj_size
        total_len = state.shape[-1]
        if total_len < expected_adj_total:
            raise ValueError(f"State length {total_len} is smaller than expected adjacencies {expected_adj_total}")
        base_dim_actual = total_len - expected_adj_total

        feature_dims_runtime = {
            'ind': self.feature_dims['ind'],
            'pos': self.feature_dims['pos'],
            'neg': self.feature_dims['neg'],
            'base': base_dim_actual,
        }

        ptr = 0
        features = {}
        for feat, dim in feature_dims_runtime.items():
            if dim > 0:
                features[feat] = state[..., ptr:ptr + dim]
                ptr += dim

        irl_rewards = []
        active_weights = []
        for feat, data in features.items():
            # Rebuild encoder on-the-fly if runtime dim differs from configured dim
            encoder = self.encoders[feat] if feat in self.encoders else None
            if encoder is not None and encoder[0].in_features != data.shape[-1] + self.num_stocks:
                self.encoders[feat] = nn.Sequential(
                    nn.Linear(data.shape[-1] + self.num_stocks, encoder[0].out_features),
                    nn.ReLU()
                ).to(data.device)
                encoder = self.encoders[feat]
            fused = torch.cat([data, action], dim=-1)
            encoded = encoder(fused) if encoder is not None else fused.mean(dim=-1, keepdim=True)
            irl_rewards.append(encoded.mean(dim=-1, keepdim=True))
            active_weights.append(self.weights[len(active_weights)])

        if self.use_drawdown and wealth_info is not None:
            W_current = wealth_info[:, 0:1]
            W_peak = wealth_info[:, 1:2]
            epsilon = 1e-8
            dd = (W_peak - W_current) / torch.clamp(W_peak, min=epsilon)
            kappa_adaptive = self.dd_kappa_min + (self.dd_kappa_max - self.dd_kappa_min) * risk_score
            dd_scaled = (dd - kappa_adaptive) / self.dd_tau
            R_dd = -F.softplus(dd_scaled)
            active_weights.append(self.weights[len(active_weights)])
            irl_rewards.append(R_dd)

        weights_softmax = F.softmax(torch.stack(active_weights), dim=0)
        R_total = sum(w * r for w, r in zip(weights_softmax, irl_rewards))
        return R_total


def process_data(data_dict, device="cuda:0"):
    corr = data_dict['corr'].to(device).squeeze()
    ts_features = data_dict['ts_features'].to(device).squeeze()
    features = data_dict['features'].to(device).squeeze()
    industry_matrix = data_dict['industry_matrix'].to(device).squeeze()
    pos_matrix = data_dict['pos_matrix'].to(device).squeeze()
    neg_matrix = data_dict['neg_matrix'].to(device).squeeze()
    pyg_data = data_dict['pyg_data'].to(device)
    labels = data_dict['labels'].to(device).squeeze()
    mask = data_dict['mask']
    return corr, ts_features, features,\
           industry_matrix, pos_matrix, neg_matrix,\
           labels, pyg_data, mask


def resolve_expert_cache_path(args, risk_score):
    cache_setting = getattr(args, "expert_cache_path", None)
    if not cache_setting:
        return None
    cache_setting = os.path.expanduser(cache_setting)

    def _default_filename():
        return (
            f"experts_{args.market}_{args.train_start_date}_{args.train_end_date}_"
            f"{args.horizon}_{args.relation_type}_risk{risk_score:.2f}.pkl"
        )

    is_dir_hint = cache_setting.endswith(os.sep) or os.path.splitext(cache_setting)[1] == ""
    if os.path.isdir(cache_setting) or is_dir_hint:
        base_dir = cache_setting.rstrip(os.sep)
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, _default_filename())

    base_dir = os.path.dirname(cache_setting)
    if base_dir:
        os.makedirs(base_dir, exist_ok=True)
    return cache_setting


def create_env_init(args, dataset=None, data_loader=None):
    if data_loader is None:
        data_loader = DataLoader(dataset, batch_size=len(dataset), pin_memory=True, collate_fn=lambda x: x,
                                 drop_last=True)
    for batch_idx, data in enumerate(data_loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
        env = StockPortfolioEnv(args=args, corr=corr, ts_features=ts_features, features=features,
                                ind=ind, pos=pos, neg=neg,
                                returns=labels, pyg_data=pyg_data, device=args.device,
                                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn,
                                risk_profile=getattr(args, 'risk_profile', None))
        env.seed(seed=args.seed)
        env, _ = env.get_sb_env()
        print("Environment created")
        return env


PPO_PARAMS = {
    "n_steps": 2048,
    "ent_coef": 0.005,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "gamma": 0.99,
    "tensorboard_log": "./logs",
    "clip_range": 0.2,
    "gae_lambda": 0.95,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
}


def model_predict(args, model, test_loader, split: str = "test"):
    df_benchmark = pd.read_csv(f"./dataset_default/index_data/{args.market}_index.csv")
    df_benchmark = df_benchmark[(df_benchmark['datetime'] >= args.test_start_date) &
                                (df_benchmark['datetime'] <= args.test_end_date)]
    benchmark_return = df_benchmark['daily_return']

    ticker_map = get_ticker_mapping_for_period(
        args.market,
        args.test_start_date,
        args.test_end_date,
        base_dir="dataset_default"
    )

    records = []
    env_snapshots = []
    all_weights_data = []

    for batch_idx, data in enumerate(test_loader):
        corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
        env_test = StockPortfolioEnv(
            args=args,
            corr=corr,
            ts_features=ts_features,
            features=features,
            ind=ind,
            pos=pos,
            neg=neg,
            returns=labels,
            pyg_data=pyg_data,
            benchmark_return=benchmark_return,
            mode="test",
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            risk_profile=getattr(args, 'risk_profile', None),
        )

        obs_test = env_test.reset()
        max_step = len(labels)

        for _ in range(max_step):
            action, _states = model.predict(obs_test, deterministic=False)
            obs_test, reward, done, info = env_test.step(action)
            if done:
                break

        metrics, benchmark_metrics = env_test.evaluate()
        metrics_record = {
            "arr": metrics.get("arr"),
            "avol": metrics.get("avol"),
            "sharpe": metrics.get("sharpe"),
            "mdd": metrics.get("mdd"),
            "cr": metrics.get("cr"),
            "ir": metrics.get("ir"),
        }
        if benchmark_metrics:
            metrics_record.update({f"benchmark_{k}": v for k, v in benchmark_metrics.items()})
        record = create_metric_record(args, split, metrics_record, batch_idx)
        records.append(record)
        env_snapshots.append((env_test, record["run_id"]))

        weights_array = env_test.get_weights_history()
        if weights_array.size > 0:
            num_steps, num_stocks = weights_array.shape
            date_keys = list(ticker_map.keys()) if ticker_map else []
            step_labels = getattr(env_test, "dates", list(range(num_steps)))

            for step_idx in range(num_steps):
                weights = weights_array[step_idx]
                date_label = date_keys[step_idx] if step_idx < len(date_keys) else None

                tickers = None
                if date_label:
                    candidate = ticker_map.get(date_label)
                    if candidate and len(candidate) == num_stocks:
                        tickers = candidate

                if tickers is None:
                    tickers = [f"stock_{i}" for i in range(num_stocks)]
                    if date_label is None:
                        date_val = step_labels[step_idx] if step_idx < len(step_labels) else step_idx
                        date_label = f"step_{date_val}"

                step_value = step_labels[step_idx] if step_idx < len(step_labels) else step_idx

                for ticker, weight in zip(tickers, weights):
                    if weight > 0.0001:
                        all_weights_data.append({
                            'run_id': record["run_id"],
                            'batch': batch_idx,
                            'date': date_label,
                            'step': step_value,
                            'ticker': ticker,
                            'weight': weight,
                            'weight_pct': weight * 100
                        })

    log_info = persist_metrics(records, env_snapshots, args, split)
    summary = aggregate_metric_records(records)

    if all_weights_data:
        timestamp_label = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        log_dir = None
        if log_info and 'log_dir' in log_info:
            log_dir = log_info['log_dir']
        else:
            log_dir = os.path.join("logs", "weights")
            os.makedirs(log_dir, exist_ok=True)

        weights_csv_path = os.path.join(log_dir, f"{split}_weights_{timestamp_label}.csv")
        df_weights = pd.DataFrame(all_weights_data)
        df_weights.to_csv(weights_csv_path, index=False)
        if log_info is not None:
            log_info['weights_csv'] = weights_csv_path

        summary_weights = df_weights.groupby('ticker').agg({
            'weight': ['mean', 'std', 'min', 'max', 'count']
        }).round(6)
        summary_weights.columns = ['avg_weight', 'std_weight', 'min_weight', 'max_weight', 'num_days']
        summary_weights = summary_weights.sort_values('avg_weight', ascending=False)

        summary_path = os.path.join(log_dir, f"{split}_weights_summary_{timestamp_label}.csv")
        summary_weights.to_csv(summary_path)
        if log_info is not None:
            log_info['weights_summary'] = summary_path

    if summary:
        print(f"Evaluation summary for split='{split}': {summary}")

    return {
        "records": records,
        "summary": summary,
        "log": log_info,
    }


def train_model_and_predict(model, args, train_loader, val_loader, test_loader):
    print("\n" + "=" * 70)
    print("Using Hybrid Ensemble Expert Generation")
    print("=" * 70)
    profile = getattr(args, "risk_profile", None)
    risk_score_value = profile.get("risk_score", getattr(args, "risk_score", 0.5)) if profile else getattr(args, "risk_score", 0.5)
    cache_file = resolve_expert_cache_path(args, risk_score_value)
    expert_trajectories = None
    if cache_file and os.path.exists(cache_file):
        try:
            print(f"Loading cached expert trajectories from {cache_file}")
            expert_trajectories = load_expert_trajectories(cache_file)
        except Exception as exc:
            print(f"Warning: failed to load cached expert trajectories ({exc}); regenerating.")

    if expert_trajectories is None:
        expert_trajectories = generate_expert_trajectories(
            args,
            train_loader.dataset,
            num_trajectories=getattr(args, "num_expert_trajectories", 100),
            risk_profile=profile,
        )
        if cache_file:
            try:
                save_expert_trajectories(expert_trajectories, cache_file)
            except Exception as exc:
                print(f"Warning: could not save expert trajectories to {cache_file}: {exc}")

    num_stocks = args.num_stocks
    lookback = getattr(args, 'lookback', 30)
    input_dim = getattr(args, 'input_dim', 6)
    graph_size = num_stocks * num_stocks
    ts_size = num_stocks * lookback * input_dim
    obs_len = graph_size * 3 + ts_size + num_stocks  # +prev_weights

    if not args.multi_reward:
        reward_net = RewardNetwork(input_dim=obs_len + 1).to(args.device)
    else:
        reward_net = MultiRewardNetwork(
            input_dim=input_dim,
            num_stocks=num_stocks,
            ind_yn=args.ind_yn,
            pos_yn=args.pos_yn,
            neg_yn=args.neg_yn,
            dd_base_weight=getattr(args, 'dd_base_weight', 1.0),
            dd_risk_factor=getattr(args, 'dd_risk_factor', 1.0),
            lookback=lookback
        ).to(args.device)

    if getattr(args, 'reward_net_path', None) and os.path.exists(args.reward_net_path):
        try:
            print(f"Loading reward network from {args.reward_net_path}")
            state = torch.load(args.reward_net_path, map_location=args.device)
            reward_net.load_state_dict(state)
        except Exception as e:
            print(f"Warning: failed to load reward net: {e}")

    irl_trainer = MaxEntIRL(reward_net, expert_trajectories, lr=1e-4, risk_score=risk_score_value)

    env_train = create_env_init(args, data_loader=train_loader)
    for i in range(args.max_epochs):
        print(f"\n=== Epoch {i+1}/{args.max_epochs} ===")
        irl_epochs = getattr(args, 'irl_epochs', 50)
        irl_trainer.train(env_train, model, num_epochs=irl_epochs,
                          batch_size=args.batch_size, device=args.device)
        print("reward net train over.")

        for batch_idx, data in enumerate(train_loader):
            corr, ts_features, features, ind, pos, neg, labels, pyg_data, mask = process_data(data, device=args.device)
            env_train = StockPortfolioEnv(
                args=args, corr=corr, ts_features=ts_features, features=features,
                ind=ind, pos=pos, neg=neg,
                returns=labels, pyg_data=pyg_data, reward_net=reward_net, device=args.device,
                ind_yn=args.ind_yn, pos_yn=args.pos_yn, neg_yn=args.neg_yn,
                risk_profile=getattr(args, 'risk_profile', None)
            )
            env_train.seed(seed=args.seed)
            env_train, _ = env_train.get_sb_env()
            model.set_env(env_train)
            rl_timesteps = getattr(args, 'rl_timesteps', 10000)
            print(f"Training RL agent for {rl_timesteps} timesteps...")
            model = model.learn(total_timesteps=rl_timesteps)
            mean_reward, std_reward = evaluate_policy(model, env_train, n_eval_episodes=10)
            print(f"Evaluation after RL training: Mean Reward = {mean_reward:.4f}, Std Reward = {std_reward:.4f}")

        print("\n=== Intermediate Test Evaluation (after RL learn) ===")
        model_predict(args, model, test_loader, split=f"epoch{i+1}_test")

    try:
        save_dir = getattr(args, 'save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        ckpt_path = os.path.join(save_dir, f"reward_net_{args.market}_{ts}.pt")
        torch.save(reward_net.state_dict(), ckpt_path)
        print(f"Saved reward network to {ckpt_path}")
    except Exception as e:
        print(f"Warning: could not save reward net: {e}")

    print("\n=== Final Test Evaluation ===")
    final_eval = model_predict(args, model, test_loader, split="final_test")

    candidate_path = None
    try:
        save_dir = getattr(args, 'save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        ts = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        policy_name = getattr(args, 'policy', 'policy').lower()
        candidate_path = os.path.join(save_dir, f"ppo_{policy_name}_{args.market}_{ts}.zip")
        model.save(candidate_path)
        print(f"Saved PPO model checkpoint to {candidate_path}")
    except Exception as e:
        print(f"Warning: could not save PPO model: {e}")

    apply_promotion_gate(args, candidate_path, final_eval.get("summary"), final_eval.get("log"))

    return model

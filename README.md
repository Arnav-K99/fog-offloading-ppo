# PPO-Based Task Offloading in Edge-Fog-Cloud Networks

Reinforcement learning system that intelligently routes compute tasks through a 3-tier fog computing network to minimize end-to-end latency, energy consumption, and cost.

## Architecture

```
User/Edge Device → Routers (Fog Layer) → Hubs (Fog Layer) → Cloud
```

The agent learns to select the optimal router+hub path for each task, avoiding congested nodes and reducing processing delay.

## Results

| Metric | Random Routing | PPO Routing | Improvement |
|--------|---------------|-------------|-------------|
| Avg Latency | ~7.88 sec | ~7.04 sec | **10.66%** |
| Avg Energy | 123.17 J| 117.86 J | **4.31%** |
| Avg Cost | 52.42 | 49.96 | **4.69%** |

PPO consistently outperforms random routing by learning to avoid high-queue, high-load nodes dynamically.

## Key Components

- **D1–D8 Latency Model** — 8-component delay model covering transmission, queue, and processing delays across all 3 tiers
- **PPO Agent** — trained using Stable Baselines3 with a 33-dimensional state space including task properties, node queue depths, CPU loads, and bandwidth
- **Dynamic Environment** — queues and loads update every step, forcing the agent to learn reactive routing
- **Priority Users** — premium tasks (priority > 0.5) receive 1.5× reward scaling, encouraging the agent to prioritize low-latency paths for high-priority users
- **Energy & Cost** — computation and transmission energy modeled using standard MEC formulas; cost computed as weighted sum of energy, latency, and bandwidth

## State Space (33 dimensions)

| Feature | Description |
|---------|-------------|
| size_mb / TASK_SIZE_MAX | Normalized task size |
| cycles_per_byte / CPU_CYCLE_MAX | Normalized compute intensity |
| priority | Task priority (0–1) |
| router_queue × 5 | Queue lengths (sorted ascending) |
| router_load × 5 | CPU loads (sorted by queue) |
| router_bandwidth × 5 | Bandwidths (sorted by queue) |
| hub_queue × 5 | Hub queue lengths |
| hub_load × 5 | Hub CPU loads |
| hub_bandwidth × 5 | Hub bandwidths |

Nodes are **sorted by queue length** before being fed to the network — this eliminates index bias and forces the agent to learn position-invariant routing.

## Reward Function

```python
reward = -priority_scale * (
    latency
    + energy_term
    + 5.0 * (router_queue[chosen] - min_router_queue)
    + 8.0 * (hub_queue[chosen]    - min_hub_queue)
    + 0.5 * router_load[chosen]
    + 1.5 * hub_load[chosen]
)
```

Relative queue penalties (chosen − min) teach the agent to prefer the least loaded node, not just avoid all congestion.

## Network Configuration

| Parameter | Value |
|-----------|-------|
| Routers | 5 |
| Hubs | 5 |
| Action space | Discrete(25) — 5×5 router/hub combinations |
| Router bandwidth | 20–100 Mbps |
| Hub bandwidth | 50–200 Mbps |
| Cloud uplink | 300 Mbps |
| Hub CPU | 5 GHz |
| Cloud CPU | 10 GHz |

## Training

```python
model = PPO(
    "MlpPolicy", env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    ent_coef=0.05,
    seed=42
)
model.learn(total_timesteps=200000)
```

## Setup

```bash
pip install stable-baselines3 gymnasium numpy
```

Open `ARCH1.ipynb` in Jupyter and run all cells. The trained model is saved as `ppo_offloading_model.zip`.

## Live Demo

Interactive visualizer showing PPO vs random routing paths, node stats, and latency comparison:
👉 [Live Demo](https://YOUR_USERNAME.github.io/YOUR_REPO_NAME)

## Planned Improvements

- CNN feature extractor for spatial network state representation
- Multi-agent routing across parallel edge devices
- Real trace-driven workload simulation

## References

- Mach & Becvar (2017) — Mobile Edge Computing: A Survey on Architecture and Computation Offloading
- Standard MEC energy model: E = κ · f² · C
- PPO: Schulman et al. (2017) — Proximal Policy Optimization Algorithms

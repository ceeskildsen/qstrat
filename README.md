qstrat — Signals → Portfolio → Diagnostics



Cross-sectional equity research pipeline: momentum and short-horizon mean-reversion signals flow into a mean–variance portfolio with realistic constraints, and the project produces PM-style diagnostics and plots.



Results at a glance



<p align="center"> <img src="figures/equity\_vs\_riskfree.png" width="48%" /> <img src="figures/rolling\_sharpe.png" width="48%" /> </p> <p align="center"> <img src="figures/constraint\_binding\_summary.png" width="48%" /> <img src="figures/mom\_decile\_equity\_gap30\_lb126.png" width="48%" /> </p> <p align="center"> <img src="figures/sl\_mr\_ic\_ir\_heatmap.png" width="48%" /> </p>



Quickstart



Option A — Windows without activating the virtual environment



Navigate to the repo root: C:\\Users\\carle\\Projects\\qstrat



Run: .\\.qstrat\\Scripts\\python.exe -m pip install -U pip



Run: .\\.qstrat\\Scripts\\python.exe -m pip install -e .



Run the end-to-end demo: .\\.qstrat\\Scripts\\python.exe -m projects.alpha\_to\_portfolio



(Optional) Run the A/B study: .\\.qstrat\\Scripts\\python.exe -m projects.ab\_test



Option B — Activate the virtual environment for this PowerShell session



Navigate to the repo root: C:\\Users\\carle\\Projects\\qstrat



Allow activation in this session: Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass



Activate: . .\\.qstrat\\Scripts\\Activate.ps1 (dot, space, then path)



Install: .\\.qstrat\\Scripts\\python.exe -m pip install -U pip



Install the package: .\\.qstrat\\Scripts\\python.exe -m pip install -e .



Run the demo: .\\.qstrat\\Scripts\\python.exe -m projects.alpha\_to\_portfolio



(Optional) Run the A/B study: .\\.qstrat\\Scripts\\python.exe -m projects.ab\_test



Other entry points

• .\\.qstrat\\Scripts\\python.exe -m projects.portfolio\_diagnostics

• .\\.qstrat\\Scripts\\python.exe -m projects.signal\_lab\_mom

• .\\.qstrat\\Scripts\\python.exe -m projects.signal\_lab\_mr

• .\\.qstrat\\Scripts\\python.exe -m projects.ab\_test (writes equity\_ab.png, drawdown\_ab.png, summary.csv)



Repository layout



src/ — library code (signals, optimizer, risk, backtest, plotting)

projects/ — runnable scripts and workflows

outputs/ — generated PNGs and CSVs (ignored by git)

figures/ — selected images copied from outputs/ so the README can display them



Pipeline



• Data: loaders and sector mapping (src/data/prices.py, src/data/sectors.py).

• Signals and features: momentum (12-1), short-horizon mean-reversion, blending and standardization (src/signals.py, src/features.py, src/alpha\_models\_rules.py).

• Risk and constraints: EWMA/sample covariance, market-beta and sector neutrality, position caps (src/risk\_models.py, src/backtest/risk\_cov.py, src/constraints.py).

• Optimization: mean–variance quadratic program using OSQP/SCS/Clarabel (src/optimizer.py).

• Diagnostics and plots: equity and drawdowns, rolling Sharpe, exposures, constraint binding, IC/IR and heatmaps (src/metrics.py, src/diagnostics.py, src/plots.py).



Configs and reproducibility



Main knobs live in projects/alpha\_run/config.py:

• rebalance\_freq (e.g., “monthly”), train\_window\_days

• long\_only vs dollar-neutral, position caps and gross control

• market-beta and sector neutrality switches

• transaction cost model



Note: Data is fetched from yfinance; exact numbers can drift slightly over time. Pin your end dates for strict replication.



Outputs



• outputs/alpha\_to\_portfolio/: pnl.csv, weights.csv, equity\_vs\_riskfree.png, equity\_excess\_over\_bil.png

• outputs/portfolio\_diagnostics/: rolling\_sharpe.png/.csv, drawdown.png/.csv, constraint\_binding\_summary.png/.csv, beta\_exposure.png/.csv, sector\_exposure.png/.csv

• outputs/ab\_test/: equity\_ab.png, drawdown\_ab.png, summary.csv

• outputs/signal\_lab\_mom/: decile equity PNGs, mom\_monthly\_ic\_summary.csv, mom\_decile\_stats\_\*.csv, grid summaries

• outputs/signal\_lab\_mr/: sl\_mr\_ic\_ir\_heatmap.png, sl\_mr\_blend\_icir\_bar.png, sl\_mr\_grid\_summary.csv, best-stats CSVs



What’s implemented



• Signals: 12-1 momentum; short-horizon mean-reversion; IC/IR grids; decile long/short curves.

• Portfolio: mean–variance optimizer with position caps, gross control, market-beta and sector neutrality; OSQP/SCS/Clarabel solvers.

• Diagnostics: equity vs risk-free/BIL, drawdowns, rolling Sharpe, constraint binding, beta and sector exposures.



Roadmap



• 6-1 momentum variant and tighter neutralization path

• Uncertainty-aware scaling (vol targeting / GP overlay)

• One-pager report assembled from diagnostics



Environment



Python 3.10–3.13. Install with: ..qstrat\\Scripts\\python.exe -m pip install -e .

Windows note: ecos is not required; this repo uses OSQP/SCS/Clarabel.



Disclaimer and License



For research and education only; not investment advice.

MIT License.


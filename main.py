import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# 1. Caricamento dei dati storici
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
data = yf.download(tickers, start="2018-01-01", end="2023-01-01")['Adj Close']

# 2. Calcolo dei rendimenti giornalieri
returns = data.pct_change().dropna()

# 3. Media dei rendimenti e matrice di covarianza (annualizzati)
mean_returns = returns.mean() * 252  # annualizzazione rendimenti
cov_matrix = returns.cov() * 252  # annualizzazione rischio (deviazione standard)


# 4. Funzione per calcolare la performance del portafoglio
def portfolio_performance(weights, mean_returns, cov_matrix):
    """Calcola il rendimento atteso e la deviazione standard (rischio) di un portafoglio."""
    returns = np.dot(weights, mean_returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return returns, risk


# 5. Funzione obiettivo per minimizzare il rischio
def minimize_risk(weights, mean_returns, cov_matrix):
    """Funzione obiettivo per minimizzare il rischio dato un certo rendimento atteso."""
    _, portfolio_risk = portfolio_performance(weights, mean_returns, cov_matrix)
    return portfolio_risk


# 6. Ottimizzazione del portafoglio
def optimize_portfolio(mean_returns, cov_matrix, target_return):
    """Ottimizza il portafoglio minimizzando il rischio per un dato rendimento atteso."""
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)

    # Vincoli: la somma dei pesi deve essere uguale a 1, e i rendimenti devono essere >= target
    constraints = [
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'ineq', 'fun': lambda weights: np.dot(weights, mean_returns) - target_return}
    ]

    # I pesi devono essere compresi tra 0 e 1 (no posizioni corte)
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Pesi iniziali casuali (ad esempio, uguali per ogni asset)
    initial_weights = np.ones(num_assets) / num_assets

    # Minimizzazione del rischio
    result = minimize(minimize_risk, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)

    return result


# 7. Simulazioni Monte Carlo per la frontiera efficiente
def simulate_portfolios(num_portfolios, mean_returns, cov_matrix):
    """Simulazioni Monte Carlo per la frontiera efficiente."""
    results = np.zeros((3, num_portfolios))
    num_assets = len(mean_returns)

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)  # Normalizzazione dei pesi per sommare a 1

        portfolio_return, portfolio_risk = portfolio_performance(weights, mean_returns, cov_matrix)
        results[0, i] = portfolio_risk
        results[1, i] = portfolio_return
        results[2, i] = portfolio_return / portfolio_risk  # Sharpe ratio

    return results


# 8. Visualizzazione della frontiera efficiente e composizione del portafoglio ottimale
def plot_efficient_frontier(mean_returns, cov_matrix, tickers, risk_free_rate=0.01):
    """Visualizza la frontiera efficiente, la Capital Market Line e la composizione del portafoglio ottimale."""
    num_portfolios = 10000
    results = simulate_portfolios(num_portfolios, mean_returns, cov_matrix)

    plt.figure(figsize=(10, 7))
    plt.scatter(results[0, :], results[1, :], c=results[2, :], cmap='viridis')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlabel('Rischio (Deviazione Standard Annualizzata)')
    plt.ylabel('Rendimento Atteso Annualizzato')
    plt.title('Frontiera Efficiente')

    # Linea rischio-free (Capital Market Line)
    max_sharpe_idx = np.argmax(results[2, :])  # Portafoglio con lo Sharpe Ratio massimo
    max_sharpe_portfolio = results[:, max_sharpe_idx]

    # Aggiungi la linea del rischio privo di rischio
    plt.plot([0, max_sharpe_portfolio[0]], [risk_free_rate, max_sharpe_portfolio[1]], linestyle='--', color='orange',
             label='Capital Market Line')

    # Ottimizzare il portafoglio per il rischio minimo
    target_return = 0.2  # esempio di rendimento target
    optimized_result = optimize_portfolio(mean_returns, cov_matrix, target_return)

    opt_weights = optimized_result.x
    opt_return, opt_risk = portfolio_performance(opt_weights, mean_returns, cov_matrix)

    # Aggiunta del portafoglio ottimale alla visualizzazione
    plt.scatter(opt_risk, opt_return, c='red', marker='*', s=200, label='Portafoglio Ottimale')
    plt.legend()

    # 9. Aggiunta della composizione del portafoglio ottimale al grafico
    weights_text = "\n".join([f"{ticker}: {weight:.2%}" for ticker, weight in zip(tickers, opt_weights)])
    plt.gca().text(0.55, 0.015, f"Composizione del Portafoglio Ottimale:\n{weights_text}",
                   bbox=dict(facecolor='white', alpha=0.5), transform=plt.gca().transAxes, fontsize=10)

    # Visualizza il grafico
    plt.show()


# 10. Esecuzione del grafico della frontiera efficiente e stampa della composizione del portafoglio ottimale
plot_efficient_frontier(mean_returns, cov_matrix, tickers)

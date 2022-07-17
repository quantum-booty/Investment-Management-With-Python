import numpy as np
import pandas as pd

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=-99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets

def get_fff_returns():
    """
    Load the Fama-French Research Factor Monthly Dataset
    """
    rets = pd.read_csv("data/F-F_Research_Data_Factors_m.csv",
                       header=0, index_col=0, na_values=-99.99)/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets


def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """
    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header=0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_file(filetype, weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios files
    Variant is a tuple of (weighting, size) where:
        weighting is one of "ew", "vw"
        number of inds is 30 or 49
    """    
    if filetype is "returns":
        name = f"{weighting}_rets" 
        divisor = 100
    elif filetype is "nfirms":
        name = "nfirms"
        divisor = 1
    elif filetype is "size":
        name = "size"
        divisor = 1
    else:
        raise ValueError(f"filetype must be one of: returns, nfirms, size")
    
    ind = pd.read_csv(f"data/ind{n_inds}_m_{name}.csv", header=0, index_col=0, na_values=-99.99)/divisor
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_returns(weighting="vw", n_inds=30):
    """
    Load and format the Ken French Industry Portfolios Monthly Returns
    """
    return get_ind_file("returns", weighting=weighting, n_inds=n_inds)

def get_ind_nfirms(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average number of Firms
    """
    return get_ind_file("nfirms", n_inds=n_inds)

def get_ind_size(n_inds=30):
    """
    Load and format the Ken French 30 Industry Portfolios Average size (market cap)
    """
    return get_ind_file("size", n_inds=n_inds)

def get_ind_market_caps(n_inds=30, weights=False):
    """
    Load the industry portfolio data and derive the market caps
    """
    ind_nfirms = get_ind_nfirms(n_inds=n_inds)
    ind_size = get_ind_size(n_inds=n_inds)
    ind_mktcap = ind_nfirms * ind_size
    if weights:
        total_mktcap = ind_mktcap.sum(axis=1)
        ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
        return ind_capweight
    #else
    return ind_mktcap

def get_total_market_index_returns(n_inds=30):
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_mcap = get_ind_market_caps()
    ind_capweight = ind_mcap.div(ind_mcap.sum(axis=1), axis=0)
    ind_return = get_ind_returns(weighting="vw", n_inds=n_inds)
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return

def compound(rets):
    return (1+rets).prod()-1

def period_return(r, period):
    n_months = r.shape[0]
    return (r+1).prod()**(period/n_months)-1


def drawdown(return_series: pd.Series):
    """
    Takes a times series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    drawdowns
    """
    wealth_index = (return_series+1).cumprod()
    prev_peaks = wealth_index.cummax()
    drawdown = (wealth_index-prev_peaks)/prev_peaks
    
    return pd.DataFrame({"Wealth index": wealth_index,
                         "Previous peaks": prev_peaks,
                         "Drawdown": drawdown})

def ann_return(r):
    n_months = r.shape[0]
    return (1+r).prod(axis=0)**(12/n_months)-1

def ann_vol(r):
    return r.std()*(12)**0.5

def ann_sharpe_ratio(r, ann_riskfree_r):
    mon_riskfree_r = (ann_riskfree_r+1)**(1/12)-1
    ex_r = r - mon_riskfree_r
    # obeying the stupid return convention so ann_ex_r starts from 0 rather than 1
    ann_ex_r = ann_return(ex_r)
    ann_volativity = ann_vol(r)
    return ann_ex_r / ann_volativity

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    deviation_r = r-r.mean()
    sigma_r = r.std(ddof=0) #
    exp = (deviation_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    deviation_r = r-r.mean()
    sigma_r = r.std(ddof=0)
    exp = (deviation_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not.
    Test is applied at the 1% level by default.
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    import scipy.stats
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r<0
    return r[is_negative].std(ddof=0)
    
def var_historic(r, level=5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or Data")
    
from scipy.stats import norm 
def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a Series or DataFrame
    """
    z = norm.ppf(level/100)
    if modified == True:
        # Cornish Fisher expansion
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z1 = (z**2 - 1)*s/6
        z2 = (z**3 - 3*z)*(k-3)/24
        z3 = -(2*z**3 - 5*z)*(s**2)/36
        z = z + z1 + z2 + z3
    
    return -(z*r.std(ddof=0)+r.mean())

def cvar_historic(r, level=5):
    """
    Computes the conditional VaR of Series or Data Frame
    """
    # 1) find var_historic
    # 2) filter returns in r below var_historic
    # 3) find its mean
    
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    elif isinstance(r, pd.Series): 
        r_below = r[r<=-var_historic(r, level)]
        return -r_below.mean()
    else:
        raise TypeError("Expected r to be a series or DataFrame")
        
def cvar_gaussian(r, level=5, modified=True):
    """
    Computes the conditional VaR of Series or Data Frame
    """
    # 1) find var_historic
    # 2) filter returns in r below var_historic
    # 3) find its mean
    
    if isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_gaussian, level=level, modified=modified)
    elif isinstance(r, pd.Series): 
        r_below = r[r<=-var_gaussian(r, level, modified)]
        return -r_below.mean()
    else:
        raise TypeError("Expected r to be a series or DataFrame")
        
def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, cov):
    """
    Weifghts -> Vol
    """
    return (weights.T @ cov @weights) ** 0.5


from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1),)*n
    return_is_target = {
        "type": "eq",
        "args": (er,),
        "fun": lambda weights, er: target_return - portfolio_return(weights, er)
    }
    weights_sum_to_1 = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                      args=(cov,), method="SLSQP",
                       options={"disp": False},
                       constraints=(return_is_target, weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x

def optimal_weights(n_points, er, cov):
    """
    -> list of weights to run the optimizer on to minimize the vol
    """
    target_returns = np.linspace(er.min(),er.max(),n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_returns]
    return weights

def maximize_sharpe_ratio(riskfree_rate, er, cov):
    """
    target_ret -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1),)*n

    weights_sum_to_1 = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
    }
    
    def neg_sharpe_ratio(weights, er, cov):
        return -(portfolio_return(weights, er)-riskfree_rate)/portfolio_vol(weights, cov)
    
    results = minimize(neg_sharpe_ratio, init_guess,
                      args=(er, cov), method="SLSQP",
                       options={"disp": False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                      )
    return results.x

def GMV(cov):
    """
    """
    n = cov.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1),)*n

    weights_sum_to_1 = {
            "type": "eq",
            "fun": lambda weights: np.sum(weights) - 1
    }
    w_GMV = minimize(portfolio_vol, init_guess,
                      args=(cov,), method="SLSQP",
                       options={"disp": False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds
                      )
    return w_GMV.x

def plot_ef(n_points, er, cov, show_cml=False, style=".-", riskfree_rate=0, show_ew=False, show_GMV=False):
    """
    Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vol = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volativity": vol
    })
    
    ax = ef.plot.line(x="Volativity", y="Returns", style=style)
    
    if show_cml:
        ax.set_xlim(left=0)
        weight_msr = maximize_sharpe_ratio(riskfree_rate, er, cov)
        r_msr = portfolio_return(weight_msr, er)
        vol_msr = portfolio_vol(weight_msr, cov)
        # Add Capital Market Line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color="green", marker="o", linestyle="dashed", markersize=12, linewidth=2)
        
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=12)

        
    if show_GMV:
        
        w_GMV = GMV(cov)
        r_GMV = portfolio_return(w_GMV, er)
        vol_GMB = portfolio_vol(w_GMV, cov)
        ax.plot([vol_GMB],[r_GMV], color="red", marker="o", markersize=12)
             
    return ax

def run_cppi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, ann_riskfree_rate=0.03, drawdown=None):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor
    peak = start

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])
        
    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = (ann_riskfree_rate+1)**(1/12)-1
        
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        if drawdown is not None:
            peak = np.maximum(peak, account_value)
            floor_value = peak*(1-drawdown)
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        ## update the account value for this time step
        account_value = risky_alloc*(risky_r.iloc[step]+1) + safe_alloc*(safe_r.iloc[step]+1)
        # save the values so I can look at the history and plot it etc
        account_history.iloc[step] = account_value
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w

    risky_wealth = start*(risky_r+1).cumprod()
    peak_history = account_history.cummax()
    
    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth,
        "Risky Allocation": risky_w_history,
        "Cushion History": cushion_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r": risky_r,
        "safe_r": safe_r
    }
    return backtest_result

def summary_stats(r, riskfree_rate=0.03):
    """
    Takes DataFrame of returns
    Return a DataFrame that contains aggregated summary stats for the returns in the columns of r
    """
    ann_v = r.aggregate(ann_vol)
    ann_r = r.aggregate(ann_return)

    ann_sr = r.aggregate(ann_sharpe_ratio, ann_riskfree_r=riskfree_rate)
    dd = r.aggregate(lambda r: drawdown(r)["Drawdown"].min())
    skew = r.aggregate(skewness)
    kurt = r.aggregate(kurtosis)
    cf_var5 = r.aggregate(var_gaussian, modified=True)
    hist_cvar5 = r.aggregate(cvar_historic)
    return pd.DataFrame({
        "Annualized Return": ann_r,
        "Annualized Vol": ann_v,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Cornish-Fisher VaR (5%)": cf_var5,
        "Historic CVaR": hist_cvar5,
        "Sharpe Ratio": ann_sr,
        "Max Drawdown": dd
         })
    
    
def gbm(n_years=10, n_scenarios=1000, mu=0.07, sigma=0.15, steps_per_year=12, s_0=100.0, prices=True):
    """
    Evolution of a Stock Price using a Geometric Brownian Motion Model
    """
    dt = 1/steps_per_year
    n_steps = int(n_years * steps_per_year)+1
    rets_plus_1 = np.random.normal(loc=1+mu*dt, scale=sigma*np.sqrt(dt), size=(n_steps, n_scenarios))
    rets_plus_1[0] = 1 # making sure it satisfies initial condition s0
    rets_val = s_0 * pd.DataFrame(rets_plus_1).cumprod() if prices else pd.DataFrame(rets_plus_1 - 1)
    return rets_val



def inst_to_ann(r):
    """
    Converts short rate to ann rate
    """
    return np.expm1(r)

def ann_to_inst(r):
    """
    Convert ann to a short rate
    """
    return np.log1p(r)

def cir(n_years=10, n_scenarios=2, a=0.04, b=0.03, vol=0.05, steps_per_year=12, r_0=None):
    """
    Implements the CIR model for interest rates
    r_0 is the initial annual rate, which we converted to instantanous/short rate
    b is the baseline interst rate
    returns annual rate and bond_prices
    """
    # first generate the interst rates using CIR model
    if r_0 is None: r_0 = b
    r_0 = ann_to_inst(r_0)    
    num_steps = int(n_years*steps_per_year) + 1
    dt = 1/steps_per_year
    shock = np.random.normal(loc=0, scale=vol*np.sqrt(dt), size=(num_steps, n_scenarios))
    rates = np.empty_like(shock)
    rates[0] = r_0
    
    for step in range(1, num_steps):
        r_t = rates[step-1]
        dr_t = a*(b-r_t)*dt + np.sqrt(r_t)*shock[step]
        rates[step] = abs(r_t + dr_t)
    
    # Second Calculate Bond prices using the interest rates
    def bond_price():
        h = np.sqrt(a**2 + 2*vol**2)
        tau = n_years - np.linspace(0, n_years, num_steps)
        tau = np.array([tau])
        A = (2*h*np.exp((a+h)*tau/2)/(2*h + (a+h)*np.expm1(tau*h)))**(2*a*b/vol**2)
        B = 2*np.expm1(tau*h)/(2*h + (a+h)*np.expm1(tau*h))
        return A.T*np.exp(-B.T*rates)
    
    P = bond_price()
    
    #Returns interest rate per period
    rates = (1+inst_to_ann(rates))**(1/steps_per_year) - 1
    rates = pd.DataFrame(data=rates, index=range(num_steps))
    bond_prices = pd.DataFrame(data=P, index=range(num_steps))

    return rates, bond_prices

def discount(t, r):
    """
    Compute the price of a pure discount bond that pays a dollar at time period t
    and r is the per-period interest rate
    returns a |t| x |r| Series or DataFrame
    r can be a float, Series or DataFrame, r is the annual in terest rate
    t is time in years
    returns a DataFrame indexed by t
    """
    discounts = pd.Series([(r+1)**-i for i in t])
    discounts.index = t
    return discounts

def pv(flows, r):
    """
    Compute the present value of a sequence of cash flows given by the time (as an index) and amounts
    r can be a scalar, or a Series or DataFrame with the number of rows matching the num of rows in flows
    input r is the rate with duration that matches with the index
    """
    dates = flows.index
    discounts = discount(dates, r)
    return discounts.multiply(flows, axis='rows').sum()

def funding_ratio(assets, liabilities, r):
    """
    Computes the funding ratio of a series of liabilities, based on an interest rate and current value of assets
    """
    return pv(assets, r)/pv(liabilities, r)

def bond_cash_flows(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12):
    """
    Return a series of cash flows generated by a bond, indexed by a coupon number
    The coupons_per_year is the number of times coupon is paid per year. i.e. if coupons_per_year = 12, you get paid 12 times a year
    """
    n_coupons = round(maturity*coupons_per_year)
    coupon_amt = principal*coupon_rate/coupons_per_year
    coupon_times = np.arange(1, n_coupons+1) # returns np.array([1,2,3,...,n_coupons])
    cash_flows = pd.Series(data=coupon_amt, index=coupon_times)
    cash_flows.iloc[-1] += principal # the principal is paid back at the end
    return cash_flows


def bond_price(maturity, principal=100, coupon_rate=0.03, coupons_per_year=12, discount_rate=0.03):
    """
    Computes the price of a bond that pays regular coupons until maturity
    at which time the principal and the final coupon is returned
    This is not designed to be efficient, rather,
    it is to illustrate the underlying principle behind bond pricing!
    If discount_rate is a DataFrame, then this is assumed to be the rate on each coupon date
    and the bond value is computed over time.
    i.e. The index of the discount_rate DataFrame is assumed to be the coupon number
    """
    if isinstance(discount_rate, pd.DataFrame):
        pricing_dates = discount_rate.index
        
        prices = pd.DataFrame(index=pricing_dates, columns=discount_rate.columns)
        for t in pricing_dates:
            prices.loc[t] = bond_price(maturity-t/coupons_per_year, principal, coupon_rate, coupons_per_year, discount_rate.loc[t])
        return prices
    else: # base case ... single time period
        if maturity <= 0: return principal+principal*coupon_rate/coupons_per_year
        cash_flows = bond_cash_flows(maturity, principal, coupon_rate, coupons_per_year)
        return pv(cash_flows, discount_rate)
    
def macaulay_duration(flows, discount_rate):
    """
    Computes the Macaulay Duration of a sequence of cash flows
    """
    discounted_flows = discount(flows.index, discount_rate) * flows
    weights = discounted_flows/discounted_flows.sum()
    return weights.T @ flows.index

def bond_total_return(monthly_prices, principal, coupon_rate, coupons_per_year):
    """
    Computes the total return of a Bond based on monthly bond prices and coupon payments
    Assumes that dividends (coupons) are paid out at the end of the period (e.g. end of 3 months for quarterly div)
    and that dividends are reinvestd in the bond
    """
    
    coupons = pd.DataFrame(data = 0, index=monthly_prices.index, columns=monthly_prices.columns)
    t_max = monthly_prices.index.max()
    pay_date = np.linspace(12/coupons_per_year, t_max, int(coupons_per_year*t_max/12), dtype=int)
    coupons.iloc[pay_date] = principal*coupon_rate/coupons_per_year
    total_returns = (monthly_prices + coupons)/monthly_prices.shift()-1
    return total_returns.dropna()

def bt_mix(r1, r2, allocator, **kwargs):
    """
    Runs a back test (simulation) of allocating between a two sets of returns
    r1 and r2 are T x N DataFrames of returns where T is the time step index and N is the nmber of scenarios
    allocator is a function that takes two sets of returns and allocate specific parameters, and produces
    an allocation to the first portfoio (the rest of the money is invested in the GHP) as a T x 1 DataFrame
    Returns a T x N DataFrame of the resulting N portfoio scenarios
    """
    if not r1.shape == r2.shape:
        raise ValueError("r1 and r2 need to be the same shape")    
    weights = allocator(r1, r2, **kwargs)
    if not weights.shape == r1.shape:
        raise ValueError("Allocator returned weights that dont match f1") 
    r_mix = weights*r1 + (1-weights)*r2
    return r_mix

def fixedmix_allocator(r1, r2, w1, **kwargs):
    """
    Produces a time series over T steps of allocations between the PSP and GHP across N scenarios
    PSP and GHP are T x N DataFrames that represent the returns of the PSP and GHP such that :
    - each column is a scenario
    - each row is the price for a timestep
    Retruns an T x N DataFrame of PSP weights
    """
    return pd.DataFrame(data=w1, index=r1.index, columns=r1.columns)

def terminal_values(rets):
    """
    Returns the final compounded return for each scenarios
    """
    return (1+rets).prod()

def terminal_stats(rets, floor = 0.8, goal=np.inf, name="Stats"):
    """
    Produce Summary Statistics on the terminal values per invested dollar across a range of N scenarios
    rets is a T x N DataFrame of returns, where T is the time-step (we assume reys is sorted by time)
    Returns a 1 column DataFrame of Summary Stats indexed by the stat name
    """
    terminal_wealth = (1+rets).prod()
    breach = terminal_wealth < floor
    reach = terminal_wealth >= goal
    p_breach = breach.mean() if breach.sum() > 0 else np.nan
    p_reach = reach.mean() if reach.sum() > 0 else np.nan
    e_short = (floor - terminal_wealth[breach]).mean() if breach.sum() > 0 else np.nan
    e_surplus = (terminal_wealth[reach]-goal).mean() if reach.sum() > 0 else np.nan
    sum_stats = pd.DataFrame.from_dict({
        "mean": terminal_wealth.mean(),
        "std": terminal_wealth.std(),
        "p_breach": p_breach,
        "e_short": e_short,
        "p_reach": p_reach,
        "e_surplus": e_surplus
    }, orient="index", columns=[name])
    return sum_stats

def glidepath_allocation(r1, r2, start_glide=1, end_glide=0):
    """
    Simulates a Target-Date-Fund style gradual move from r1 to r2.
    """
    n_points = r1.shape[0]
    n_col = r1.shape[1]
    path = pd.Series(data=np.linspace(start_glide, end_glide, num=n_points))
    paths = pd.concat([path]*n_col, axis=1)
    paths.index = r1.index
    paths.columns = r1.columns
    return paths

def floor_allocator(psp_r, ghp_r, floor, zc_prices, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cusion in the PSP
    Returns a DataFrame with the same shape as the PSP/GHP representing the weights in the PSP
    The zc_prices are present values of 1 dollar zero coupon bonds
    Therefore as we go closer to the end, the floor_value will converge to floor as zc_prices converge to 1
    As account_value goes closer to the present value of the floor, safe asset GHP will be more allocated.
    """
    if zc_prices.shape != psp_r.shape:
        raise ValueError("PSP and ZC prices must have the same shape")
    n_steps, n_scenarios = psp_r.shape
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = floor*zc_prices.iloc[step] ## Present value of floor assuming today's rate and flat YC
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # weights cannot be lower than or higher than 0 and 1
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        w_history.iloc[step] = psp_w
    return w_history

def drawdown_allocator(psp_r, ghp_r, maxdd, m=3):
    """
    Allocate between PSP and GHP with the goal to provide exposure to the upside of the PSP without going violating the floor.
    Uses a CPPI-style dynamic risk budgeting algorithm by investing a multiple of the cusion in the PSP
    Returns a DataFrame with the same shape as the PSP/GHP representing the weights in the PSP
    """
    if ghp_r.shape != psp_r.shape:
        raise ValueError("PSP and GHP returns must have the same shape")
    
    n_steps, n_scenarios = psp_r.shape
            
    account_value = np.repeat(1, n_scenarios)
    floor_value = np.repeat(1, n_scenarios)
    peak_value = np.repeat(1, n_scenarios)
    w_history = pd.DataFrame(index=psp_r.index, columns=psp_r.columns)
    for step in range(n_steps):
        floor_value = (1 - maxdd)*peak_value ### Floor is based on prev peak
        cushion = (account_value - floor_value)/account_value
        psp_w = (m*cushion).clip(0, 1) # weights cannot be lower than or higher than 0 and 1
        ghp_w = 1-psp_w
        psp_alloc = account_value*psp_w
        ghp_alloc = account_value*ghp_w
        # recompute the new account value and prev peak at the end of this step
        account_value = psp_alloc*(1+psp_r.iloc[step]) + ghp_alloc*(1+ghp_r.iloc[step])
        peak_value = np.maximum(peak_value, account_value)
        w_history.iloc[step] = psp_w
    return w_history

import statsmodels.api as sm
def regress(dependent_variable, explanatory_variables, alpha=True):
    """
    Runs a linear regression to decompose the dependent variable into the explanatory variables
    returns an object of type statsmodel's RegressionResults on which you can call
       .summary() to print a full summary
       .params for the coefficients
       .tvalues and .pvalues for the significance levels
       .rsquared_adj and .rsquared for quality of fit
    """
    if alpha:
        explanatory_variables = explanatory_variables.copy()
        explanatory_variables["Alpha"] = 1

    lm = sm.OLS(dependent_variable, explanatory_variables).fit()
    return lm

from scipy.optimize import minimize
def style_analysis(dependent_variable, explanatory_variables):
    """
    Returns the optimal weights that minimizes the Tracking error between
    a portfolio of the explanatory variables and the dependent variable
    """
    n = explanatory_variables.shape[1]
    init_guess = np.repeat(1/n, n)
    bounds = ((0, 1.0),)*n
    weights_sum_to_1 = {
        "type": "eq",
        "fun": lambda weights: np.sum(weights) - 1
    }
    
    def portfolio_tracking_error(weights, dependent_variable, explanatory_variables):
        return (dependent_variable.subtract(weights @ explanatory_variables.T, axis=0)**2).sum()
    
    results = minimize(portfolio_tracking_error,
                       init_guess,
                       args=(dependent_variable, explanatory_variables),
                       method='SLSQP',
                       options={"disp": False},
                       constraints=weights_sum_to_1,
                       bounds=bounds)
    
    weights = pd.Series(results.x, index=explanatory_variables.columns)
    return weights

def tracking_error(r_a, r_b):
    """
    Returns the Tracking Error between the two return series
    """
    return np.sqrt(((r_a - r_b)**2).sum())

def rolling_style(dependent_var, exp_var, window):
    weights = pd.DataFrame(index=exp_var.index, columns=exp_var.columns)
    for i in range(window, len(weights)):
        weights.iloc[i] = style_analysis(dependent_var.iloc[i-window:i], exp_var.iloc[i-window:i])
    return weights.dropna()

def rolling_factor(dependent_var, exp_var, window):
    beta = pd.DataFrame(index=exp_var.index, columns=exp_var.columns)
    for i in range(window, len(beta)):
        beta.iloc[i] = regress(dependent_var.iloc[i-window:i], exp_var.iloc[i-window:i]).params
    return beta.dropna()

def weight_ew(r, cap_weights=None, max_cw_mult=None, microcap_threshold=None, **kwargs):
    """
    Returns the weights of the EW portfolio based on the asset returns "r" as a DataFrame
    If supplied a set of capweights and a capweight tether, it is applied and reweighted 
    """
    n = len(r.columns)
    ew = pd.Series(1/n, index=r.columns)
    if cap_weights is not None:
        cw = cap_weights.loc[r.index[0]]
        if microcap_threshold is not None and microcap_threshold>0:
            microcap = cw < microcap_threshold
            ew[microcap] = 0
            ew = ew/ew.sum()
        if max_cw_mult is not None:
            ew = np.minimum(ew, cw*max_cw_mult)
            ew = ew/ew.sum()
    return ew

def weight_cw(r, cap_weights, **kwargs):
    """
    Returns the weights of the CW portfolio based on the time series of capweights
    """
    w = cap_weights.loc[r.index[0]]
    return w/w.sum()

def sample_cov(r, **kwargs):
    """
    Returns the sample covariance of the supplied returns
    """
    return r.cov()

def weight_gmv(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the GMV portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, **kwargs)
    return GMV(est_cov)

def cc_cov(r, **kwargs):
    """
    Estimates a covariance matrix by using the Elton/Gruber Constant Correlation model
    """
    n = r.shape[1]
    rhos = r.corr()
    rho_average = (rhos.values.sum()-n)/(n*(n-1))
    const_cor = np.full_like(rhos, rho_average)
    np.fill_diagonal(const_cor, 1)
    std = r.std()
    const_cov = const_cor * np.outer(std, std)
    return pd.DataFrame(const_cov, index=r.columns, columns=r.columns)

def shrinkage_cov(r, delta=0.5, **kwargs):
    """
    Covariance estimator that shrinks between the Sample Covariance and the Constant Correlation Estimators
    """
    const_cov = cc_cov(r, **kwargs)
    samp_cov = sample_cov(r, **kwargs)
    return delta*const_cov + (1-delta)*samp_cov

def backtest_ws(r, estimation_window=60, weighting=weight_ew, **kwargs):
    """
    Calculates returns with rolling weight of window=estimation_window (5 years default).
    Backtests a given weighting scheme, given some parameters:
    r : asset returns to use to build the portfolio
    estimation_window: the window to use to estimate parameters
    weighting: the weighting scheme to use, must be a function 
    that takes "r", and a variable number of keyword-value arguments
    """
    ret = lambda i: r.iloc[i]@weighting(r.iloc[i-estimation_window:i], **kwargs)
    returns = [ret(i) for i in range(estimation_window, len(r))]

    return pd.Series(returns, index=r.index[estimation_window:])



def as_colvec(x):
    if (x.ndim == 2):
        return x
    else:
        return np.expand_dims(x, axis=1)
    
def implied_returns(delta, sigma, w):
    """
Obtain the implied expected returns by reverse engineering the weights
Inputs:
delta: Risk Aversion Coefficient (scalar)
sigma: Variance-Covariance Matrix (N x N) as DataFrame
    w: Portfolio weights (N x 1) as Series
Returns an N x 1 vector of Returns as Series
    """
    ir = delta * sigma.dot(w).squeeze() # to get a series from a 1-column dataframe
    ir.name = 'Implied Returns'
    return ir

# Assumes that Omega is proportional to the variance of the prior
def proportional_prior(sigma, tau, p):
    """
    Returns the He-Litterman simplified Omega
    Inputs:
    sigma: N x N Covariance Matrix as DataFrame
    tau: a scalar
    p: a K x N DataFrame linking Q and Assets
    returns a P x P DataFrame, a Matrix representing Prior Uncertainties
    """
    helit_omega = p.dot(tau * sigma).dot(p.T)
    # Make a diag matrix from the diag elements of Omega
    return pd.DataFrame(np.diag(np.diag(helit_omega.values)),index=p.index, columns=p.index)


from numpy.linalg import inv

def bl(w_prior, sigma_prior, p, q,
                omega=None,
                delta=2.5, tau=.02):
    """
# Computes the posterior expected returns based on 
# the original black litterman reference model
#
# W.prior must be an N x 1 vector of weights, a Series
# Sigma.prior is an N x N covariance matrix, a DataFrame
# P must be a K x N matrix linking Q and the Assets, a DataFrame
# Q must be an K x 1 vector of views, a Series
# Omega must be a K x K matrix a DataFrame, or None
# if Omega is None, we assume it is
#    proportional to variance of the prior
# delta and tau are scalars
    """
    if omega is None:
        omega = proportional_prior(sigma_prior, tau, p)
    # Force w.prior and Q to be column vectors
    # How many assets do we have?
    N = w_prior.shape[0]
    # And how many views?
    K = q.shape[0]
    # First, reverse-engineer the weights to get pi
    pi = implied_returns(delta, sigma_prior,  w_prior)
    # Adjust (scale) Sigma by the uncertainty scaling factor
    sigma_prior_scaled = tau * sigma_prior  
    # posterior estimate of the mean, use the "Master Formula"
    # we use the versions that do not require
    # Omega to be inverted (see previous section)
    # this is easier to read if we use '@' for matrixmult instead of .dot()
    #     mu_bl = pi + sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ (q - p @ pi)
    mu_bl = pi + sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega).dot(q - p.dot(pi).values))
    # posterior estimate of uncertainty of mu.bl
#     sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled @ p.T @ inv(p @ sigma_prior_scaled @ p.T + omega) @ p @ sigma_prior_scaled
    sigma_bl = sigma_prior + sigma_prior_scaled - sigma_prior_scaled.dot(p.T).dot(inv(p.dot(sigma_prior_scaled).dot(p.T) + omega)).dot(p).dot(sigma_prior_scaled)
    return (mu_bl, sigma_bl)


# for convenience and readability, define the inverse of a dataframe
def inverse(d):
    """
    Invert the dataframe by inverting the underlying matrix
    """
    return pd.DataFrame(inv(d.values), index=d.columns, columns=d.index)

def w_msr(sigma, mu, scale=True):
    """
    Optimal (Tangent/Max Sharpe Ratio) Portfolio weights
    by using the Markowitz Optimization Procedure
    Mu is the vector of Excess expected Returns
    Sigma must be an N x N matrix as a DataFrame and Mu a column vector as a Series
    This implements page 188 Equation 5.2.28 of
    "The econometrics of financial markets" Campbell, Lo and Mackinlay.
    """
    w = inverse(sigma).dot(mu)
    if scale:
        w = w/sum(w) # fix: this assumes all w is +ve
    return w


def risk_contribution(w, cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    # Warning, pandas does have built-in outer product so w.T@w does not work
    var = (w*cov).multiply(w,axis=0)
    risk_contribution = var.sum(axis=1) / var.sum().sum()
    return risk_contribution

def target_risk_contributions(target_risk, cov):
    """
    Returns the weights of the portfolio that gives you the weights such
    that the contributions to portfolio risk are as close as possible to
    the target_risk, given the covariance matrix
    """
    def cost(w, target_risk, cov):
        w_contribs = risk_contribution(w, cov)
        return ((target_risk - w_contribs)**2).sum()
    
    init_w = np.repeat(1/len(cov), len(cov))
    
    test_c = cost(init_w, target_risk, cov)
    
    bounds = ((0.0, 1.0),) * len(cov)
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1}
    
    result = minimize(cost,
             init_w,
             args=(target_risk, cov),
             method='SLSQP',
             bounds=bounds,
             constraints=weights_sum_to_1
            )
    return result.x

def equal_risk_contributions(cov):
    """
    Returns the weights of the portfolio that equalizes the contributions
    of the constituents based on the given covariance matrix
    """
    equal_risk = np.repeat(1/len(cov), len(cov))
    weight = target_risk_contributions(equal_risk, cov)
    return weight

def weight_erc(r, cov_estimator=sample_cov, **kwargs):
    """
    Produces the weights of the ERC portfolio given a covariance matrix of the returns 
    """
    est_cov = cov_estimator(r, **kwargs)
    return equal_risk_contributions(est_cov)
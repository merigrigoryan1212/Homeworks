"""
Bass-model analysis: Jetson Air (TIME Best Inventions 2025) using US air-source
heat-pump (ASHP) shipments as the look-alike series.
Run:  python analysis.py      -> writes img/*.png, data/*.csv, results.json
"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import least_squares, brentq

DATA = "data/us_air_source_heat_pump_shipments_2001_2025.csv"
d = pd.read_csv(DATA)

# ---------- external constants (sources in report) ----------
HOMES_2020        = 123.53   # M housing units, EIA RECS 2020
HOMES_HEATING     = 117.74   # M homes that use space-heating equipment, EIA RECS 2020 (HC6.8)
HP_CENTRAL        = 16.13    # M homes, central heat pump main heating, RECS 2020
HP_DUCTLESS       = 1.06     # M homes, ductless heat pump main heating, RECS 2020
ADOPTED_2020      = HP_CENTRAL + HP_DUCTLESS      # 17.19
M_LOW, M_BASE, M_HIGH = 0.5*HOMES_2020, HOMES_HEATING, HOMES_2020
JETSON_INSTALLS_JAN2026 = 1000   # "more than 1,000 systems" (Business Wire, 20 Jan 2026)
JETSON_MONTHS     = 4.5          # launch Sept 2025 -> Jan 2026 (approx.)

# ---------- Bass functions (lecture notation) ----------
def F(t, p, q):
    e = np.exp(-(p+q)*t); return (1-e)/(1+(q/p)*e)
def f(t, p, q):
    e = np.exp(-(p+q)*t); return ((p+q)**2/p)*e/(1+(q/p)*e)**2
def t_peak(p, q): return np.log(q/p)/(p+q)

def fit(y, M=None):
    """Least squares on period sales: S_i = M[F(i)-F(i-1)], i=1..n (t=0 at start of first year).
       M free (3 params) or fixed (2 params). Multi-start, positive p,q."""
    y = np.asarray(y, float); n = len(y); t1 = np.arange(1, n+1.); t0 = t1-1
    best, sols = None, []
    for p0 in [0.0005, 0.003, 0.01, 0.03, 0.1]:
        for q0 in [0.05, 0.15, 0.3, 0.5]:
            for m0 in ([50, 100, 200, 500] if M is None else [None]):
                if M is None:
                    fun = lambda x: x[2]*(F(t1,x[0],x[1])-F(t0,x[0],x[1]))-y
                    x0, lb, ub = [p0,q0,m0], [1e-6,1e-3,y.sum()], [1,3,5000]
                else:
                    fun = lambda x: M*(F(t1,x[0],x[1])-F(t0,x[0],x[1]))-y
                    x0, lb, ub = [p0,q0], [1e-6,1e-3], [1,3]
                try: r = least_squares(fun, x0, bounds=(lb,ub), max_nfev=2000)
                except Exception: continue
                sols.append(r.cost)
                if best is None or r.cost < best.cost: best = r
    sse = 2*best.cost; r2 = 1-sse/((y-y.mean())**2).sum()
    x = best.x
    out = dict(p=float(x[0]), q=float(x[1]), M=float(x[2]) if M is None else float(M),
               r2=float(r2), rmse=float(np.sqrt(sse/n)), n=n,
               M_at_bound=bool(M is None and x[2] >= 4999),
               multistart_spread=float(max(sols)-min(sols)) if len(sols)>1 else 0)
    return out

def window(a, b): return d[(d.year>=a)&(d.year<=b)].shipments_millions.values

# ---------- 1. fits ----------
R = {}
specs = [("2001-2025 | M free", (2001,2025), None),
         ("2009-2025 | M free", (2009,2025), None),
         ("2001-2025 | M = %.2f" % M_BASE, (2001,2025), M_BASE),
         ("2009-2025 | M = %.2f" % M_BASE, (2009,2025), M_BASE),
         ("2001-2025 | M = %.2f" % M_HIGH, (2001,2025), M_HIGH),
         ("2009-2025 | M = %.2f" % M_HIGH, (2009,2025), M_HIGH),
         ("2009-2025 | M = %.2f" % M_LOW,  (2009,2025), M_LOW)]
fits = []
for name, (a,b), M in specs:
    r = fit(window(a,b), M); r["spec"] = name; r["window"] = f"{a}-{b}"; r["M_mode"] = "free" if M is None else "fixed"
    r["t_peak_years"] = float(t_peak(r["p"], r["q"])); r["peak_calendar_year"] = float(a-1+r["t_peak_years"])
    fits.append(r)
fit_df = pd.DataFrame(fits)[["spec","p","q","M","r2","rmse","t_peak_years","peak_calendar_year","M_at_bound"]]
fit_df.to_csv("data/bass_fit_results.csv", index=False)
R["fits"] = fits
BASE = [x for x in fits if x["spec"].startswith("2009-2025 | M = 117")][0]
FREE09 = [x for x in fits if x["spec"]=="2009-2025 | M free"][0]

# ---------- 2. leave-one-year-out (base spec) ----------
y09 = window(2009,2025); loo = []
for k in range(len(y09)):
    # drop one observation but keep time positions
    idx = [i for i in range(len(y09)) if i != k]
    t1 = np.array(idx)+1.; t0 = t1-1; yy = y09[idx]
    best=None
    for p0 in [0.003,0.01,0.03]:
        for q0 in [0.05,0.12,0.3]:
            r = least_squares(lambda x: M_BASE*(F(t1,x[0],x[1])-F(t0,x[0],x[1]))-yy,[p0,q0],bounds=([1e-6,1e-3],[1,3]))
            if best is None or r.cost<best.cost: best=r
    loo.append(best.x)
loo = np.array(loo)
R["loo"] = dict(p_min=float(loo[:,0].min()), p_max=float(loo[:,0].max()),
                q_min=float(loo[:,1].min()), q_max=float(loo[:,1].max()))

# ---------- 3. hold-out back-test: fit 2009-2020, predict 2021-2025 ----------
ytr = window(2009,2020); bt = fit(ytr, M_BASE)
t1 = np.arange(13,18.); t0 = t1-1
pred_ho = M_BASE*(F(t1,bt["p"],bt["q"])-F(t0,bt["p"],bt["q"]))
act_ho = window(2021,2025)
mape = float(np.mean(np.abs(pred_ho-act_ho)/act_ho)*100)
R["backtest"] = dict(p=bt["p"], q=bt["q"], r2_train=bt["r2"], years=list(range(2021,2026)),
                     pred=[float(v) for v in pred_ho], actual=[float(v) for v in act_ho],
                     mape=mape, rmse=float(np.sqrt(np.mean((pred_ho-act_ho)**2))))
# naive benchmark: 2020 value carried forward / linear trend 2009-2020
lin = np.polyfit(np.arange(2009,2021), ytr, 1); lin_pred = np.polyval(lin, np.arange(2021,2026))
R["backtest"]["linear_pred"] = [float(v) for v in lin_pred]
R["backtest"]["linear_mape"] = float(np.mean(np.abs(lin_pred-act_ho)/act_ho)*100)

# ---------- 4. From shipments to net new heat-pump households ----------
# RECS stock of homes with a (central) heat pump as MAIN heating equipment, million homes
STOCK = {2009: 9.8, 2015: 11.8, 2020: 16.13}     # 2009: CLASP appendix of EIA RECS; 2015, 2020: EIA RECS tables
def ship(y): return float(d.loc[d.year==y,"shipments_millions"].iloc[0])
def ratio(y0,y1):
    net = STOCK[y1]-STOCK[y0]; sh = sum(ship(y) for y in range(y0+1,y1+1))
    return dict(period=f"{y0}-{y1}", net_new_homes=net, shipments=sh, k=net/sh)
K_TABLE = [ratio(2009,2015), ratio(2015,2020), ratio(2009,2020)]
R["k_table"] = K_TABLE
K = {"Low":K_TABLE[0]["k"], "Base":K_TABLE[2]["k"], "High":K_TABLE[1]["k"]}

YEARS = np.arange(2021, 2051)
# shipment path: observed 2021-2025, Bass(2009-2025, M free) afterwards
pS, qS, MS = FREE09["p"], FREE09["q"], FREE09["M"]
t1 = YEARS-2008.0                                  # 2009 -> t=1
bass_ship = MS*(F(t1,pS,qS)-F(t1-1,pS,qS))
ship_path = np.where(YEARS<=2025, [ship(y) if y<=2025 else 0 for y in YEARS], bass_ship)
R["shipment_forecast"] = dict(p=pS,q=qS,M_s=MS,peak_year_model=float(2008+t_peak(pS,qS)),
        remaining_after_2025=float(MS-window(2009,2025).sum()),
        s2026=float(ship_path[YEARS==2026][0]), s2030=float(ship_path[YEARS==2030][0]),
        s2035=float(ship_path[YEARS==2035][0]), s2040=float(ship_path[YEARS==2040][0]))
scen = {}
for label,k in K.items():
    new = k*ship_path
    stock = STOCK[2020] + np.cumsum(new)
    scen[label] = dict(k=k,new=new,stock=stock,
        M_adopt=k*MS, stock_cap=STOCK[2009]+k*MS)
R["category"] = {l: dict(k=v["k"], new_2025=float(v["new"][YEARS==2025][0]), new_2026=float(v["new"][YEARS==2026][0]),
        new_2030=float(v["new"][YEARS==2030][0]), new_2035=float(v["new"][YEARS==2035][0]),
        stock_2025=float(v["stock"][YEARS==2025][0]), stock_2035=float(v["stock"][YEARS==2035][0]),
        stock_2050=float(v["stock"][YEARS==2050][0]), share_2025=float(v["stock"][YEARS==2025][0]/HOMES_2020),
        share_2035=float(v["stock"][YEARS==2035][0]/HOMES_2020), share_2050=float(v["stock"][YEARS==2050][0]/HOMES_2020),
        M_adopt=float(v["M_adopt"]), stock_cap=float(v["stock_cap"]), cap_share_of_homes=float(v["stock_cap"]/HOMES_2020),
        cap_share_of_heating_homes=float(v["stock_cap"]/HOMES_HEATING)) for l,v in scen.items()}
cat_df = pd.DataFrame({"year":YEARS,"shipments_M_observed_or_bass":ship_path,
        **{f"{l}_new_HP_households_M":v["new"] for l,v in scen.items()},
        **{f"{l}_HP_household_stock_M":v["stock"] for l,v in scen.items()}})
cat_df.to_csv("data/category_forecast_by_year.csv", index=False)

# ---------- 5. Jetson share scenarios (explicit assumptions) ----------
annualised = JETSON_INSTALLS_JAN2026/(JETSON_MONTHS/12)       # installs per year (US+Canada)
implied_share = annualised/1e6/scen["Base"]["new"][YEARS==2026][0]
R["jetson_anchor"] = dict(annualised_installs=float(annualised), implied_share_2026=float(implied_share))
S0 = 0.003                                   # 0.30% of net new heat-pump households in 2026 (~ implied share)
S_END = {"Low":0.005, "Base":0.02, "High":0.05}   # share reached in 2035, then held (assumption)
def share_path(s_end):
    return np.array([0 if y<2026 else S0+(s_end-S0)*min(1,(y-2026)/9) for y in YEARS])
jet = {}
for sl, se in S_END.items():
    sp = share_path(se); new = scen["Base"]["new"]*sp*1e6
    jet[sl] = dict(share=sp, new=new, cum=np.cumsum(new))
R["jetson"] = {sl: dict(s_end=S_END[sl], new_2026=float(v["new"][YEARS==2026][0]), new_2030=float(v["new"][YEARS==2030][0]),
        new_2035=float(v["new"][YEARS==2035][0]), new_2040=float(v["new"][YEARS==2040][0]),
        cum_2030=float(v["cum"][YEARS==2030][0]), cum_2035=float(v["cum"][YEARS==2035][0]),
        cum_2040=float(v["cum"][YEARS==2040][0]), cum_2050=float(v["cum"][YEARS==2050][0])) for sl,v in jet.items()}
grid = {}
for cl, cv in scen.items():
    grid[cl] = {sl: float((cv["new"]*share_path(se)*1e6)[YEARS<=2035].sum()/1e3) for sl,se in S_END.items()}
R["grid_cum2035_thousand"] = grid
jet_df = pd.DataFrame({"year":YEARS, "category_new_HP_households_M_base":scen["Base"]["new"],
        **{f"jetson_share_{k}":v["share"] for k,v in jet.items()},
        **{f"jetson_new_installs_{k}":v["new"] for k,v in jet.items()},
        **{f"jetson_cumulative_{k}":v["cum"] for k,v in jet.items()}})
jet_df.to_csv("data/jetson_forecast_by_year.csv", index=False)

# ---------- 6. descriptive facts ----------
cum = d.shipments_millions.cumsum()
R["facts"] = dict(total_ship_2001_2025=float(d.shipments_millions.sum()),
                  total_ship_2009_2025=float(window(2009,2025).sum()),
                  total_ship_2001_2020=float(window(2001,2020).sum()),
                  adopted_2020=ADOPTED_2020, share_2020=ADOPTED_2020/HOMES_2020,
                  peak_year=int(d.loc[d.shipments_millions.idxmax(),"year"]), peak_val=float(d.shipments_millions.max()),
                  s2001=1.44, s2022=4.33, s2025=3.64, hp_central_only_share=HP_CENTRAL/HOMES_2020)

# ---------- 7. figures ----------
plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":150})
C1, C2, C3 = "#1f5f8b", "#c0522f", "#6b8e23"
# fig1
fig,ax = plt.subplots(1,2,figsize=(10,3.6))
ax[0].bar(d.year,d.shipments_millions,color=C1); ax[0].set_title("US air-source heat pump shipments"); ax[0].set_ylabel("million units")
ax[1].plot(d.year,cum,color=C1,marker="o",ms=3); ax[1].set_title("Cumulative shipments (2001-2025)"); ax[1].set_ylabel("million units")
plt.tight_layout(); plt.savefig("img/fig1_shipments.png"); plt.close()
# fig2 fits
fig,ax = plt.subplots(figsize=(8,4))
ax.plot(d.year,d.shipments_millions,"o-",color="black",label="Observed shipments")
for name,(a,b),M,col in [("2001-2025, M free",(2001,2025),None,C2),("2009-2025, M free",(2009,2025),None,C3),("2009-2025, M = %.1f M homes"%M_BASE,(2009,2025),M_BASE,C1)]:
    r = fit(window(a,b),M); n=b-a+1; t1=np.arange(1,n+1.)
    ax.plot(np.arange(a,b+1), r["M"]*(F(t1,r["p"],r["q"])-F(t1-1,r["p"],r["q"])),"--",color=col,label=f"{name}  (R²={r['r2']:.2f})")
ax.set_ylabel("million units per year"); ax.set_title("Bass fits to US heat-pump shipments"); ax.legend(frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig("img/fig2_bass_fits.png"); plt.close()
# fig3 backtest
fig,ax = plt.subplots(figsize=(8,4))
ax.plot(d.year,d.shipments_millions,"o-",color="black",label="Observed")
yrs_all = np.arange(2009,2026); t1=np.arange(1,18.)
ax.plot(yrs_all, M_BASE*(F(t1,bt["p"],bt["q"])-F(t1-1,bt["p"],bt["q"])),"--",color=C1,label="Bass fitted on 2009-2020, extrapolated")
ax.plot(np.arange(2021,2026),lin_pred,":",color=C2,label="Linear trend 2009-2020, extrapolated")
ax.axvline(2020.5,color="grey",lw=.8); ax.text(2020.6,1.2,"hold-out",fontsize=8,color="grey")
ax.set_xlim(2008,2026); ax.set_ylabel("million units per year"); ax.set_title("Hold-out back-test"); ax.legend(frameon=False,fontsize=8)
plt.tight_layout(); plt.savefig("img/fig3_backtest.png"); plt.close()
# fig4 category forecast
fig,ax = plt.subplots(1,3,figsize=(12,3.6))
ax[0].bar(d.year,d.shipments_millions,color="#bbbbbb",label="Observed")
yy=np.arange(2009,2051); tt=yy-2008.0
ax[0].plot(yy,MS*(F(tt,pS,qS)-F(tt-1,pS,qS)),color=C1,label="Bass (2009-25 fit)")
ax[0].set_title("US heat-pump shipments"); ax[0].set_ylabel("million units"); ax[0].legend(frameon=False,fontsize=8)
for (k,v),col in zip(scen.items(),[C3,C1,C2]):
    ax[1].plot(YEARS,v["new"],color=col,label=f"{k} (k={v['k']:.2f})")
    ax[2].plot(YEARS,v["stock"]/HOMES_2020*100,color=col,label=k)
ax[1].set_title("Net new heat-pump households / yr"); ax[1].set_ylabel("million households"); ax[1].legend(frameon=False,fontsize=8)
ax[2].set_title("Heat-pump share of US homes (main heat)"); ax[2].set_ylabel("% of 123.53M homes")
plt.tight_layout(); plt.savefig("img/fig4_category_forecast.png"); plt.close()
# fig5 jetson
fig,ax = plt.subplots(1,2,figsize=(10,3.8)); m = YEARS<=2040
for (k,v),col in zip(jet.items(),[C3,C1,C2]):
    ax[0].plot(YEARS[m],v["new"][m]/1e3,color=col,label=f"{k} (share to {S_END[k]*100:.1f}% by 2035)")
    ax[1].plot(YEARS[m],v["cum"][m]/1e3,color=col,label=k)
ax[0].set_title("Jetson Air: new US installs per year"); ax[0].set_ylabel("thousand units"); ax[0].legend(frameon=False,fontsize=8)
ax[1].set_title("Jetson Air: cumulative US installs"); ax[1].set_ylabel("thousand units")
plt.tight_layout(); plt.savefig("img/fig5_jetson_forecast.png"); plt.close()

json.dump(R, open("results.json","w"), indent=2, default=float)
print(json.dumps({k:R[k] for k in ["facts","k_table","shipment_forecast","category","jetson_anchor","jetson","grid_cum2035_thousand"]}, indent=1, default=float))
print(fit_df.round(4).to_string())

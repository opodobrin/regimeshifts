import matplotlib.pyplot as plt
import regime_shifts as rs
import ews

ts = rs.sample_rs(std=0.1)
ts = ts * (-1)

fig, ax = plt.subplots()
ts.plot(ax=ax)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('System state', fontsize=12)
plt.show()

ts = rs.Regime_shift(ts)

detection_index = ts.as_detect()

fig, ax = plt.subplots()
detection_index.plot(ax=ax)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Detection Index', fontsize=12)
plt.show()

bef_rs = ts.before_rs()
fig, ax = plt.subplots()
bef_rs.plot(ax=ax)
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('System state', fontsize=12)
plt.show()

series = ews.Ews(bef_rs)
# The Ews class returns an extended Dataframe object, if we provided a series, it sets 0 for the column name.
series = series.rename(columns={0: 'Sample series'})

trend = series.gaussian_det(bW=60).trend
residuals = series.gaussian_det(bW=60).res

fig, axs = plt.subplots(2, 1, sharex=True)
bef_rs.plot(ax=axs[0], label='')
trend['Sample series'].plot(ax=axs[0], label='Trend bW=60', linewidth=2)
residuals['Sample series'].plot(ax=axs[1])
axs[1].set_xlabel('Time', fontsize=12)
axs[0].set_ylabel('System state', fontsize=12)
axs[1].set_ylabel('Residuals', fontsize=12)
axs[0].legend(frameon=False)
plt.show()

wL = 200  # Window length specified in number of points in the series
bW = 60
# Computing lag-1 autocorrelation using the ar1() method
ar1 = series.ar1(detrend=True, bW=bW, wL=wL)
var = series.var(detrend=True, bW=bW, wL=wL)  # Computing variance

print(f'AR(1) tau = {ar1.kendall:0.3f}')
print(f'Var tau = {var.kendall:0.3f}')

fig, axs = plt.subplots(3, 1, sharex=True, figsize=(7, 7))
ts.plot(ax=axs[0], legend=False)
ar1['Sample series'].plot(
    ax=axs[1], label=rf"Kendall's $\tau =$ {ar1.kendall:.2f}")
var['Sample series'].plot(
    ax=axs[2], label=rf"Kendall's $\tau =$ {var.kendall:.2f}")
axs[0].set_ylabel('System state', fontsize=13)
axs[1].set_ylabel('AR(1)', fontsize=13)
axs[2].set_ylabel('Variance', fontsize=13)
axs[1].legend(frameon=False)
axs[2].legend(frameon=False)
axs[2].set_xlabel('Time', fontsize=13)
plt.show()

# Computing lag-1 autocorrelation using the pearsonc() method
pearson = series.pearsonc(detrend=True, bW=bW, wL=wL)
fig, axs = plt.subplots()
ar1['Sample series'].plot(
    ax=axs, linewidth=3, label=rf"AR(1) $\tau =$ {ar1.kendall:.2f}")
pearson['Sample series'].plot(
    ax=axs, label=rf"Pearson corr. $\tau =$ {pearson.kendall:.2f}")
axs.legend(frameon=False)
axs.set_ylabel('Lag-1 autocorrelation', fontsize=13)
axs.set_xlabel('Time', fontsize=13)
plt.show()

sig_pearson = series.significance(
    indicator='pearsonc', n=1000, detrend=True, wL=wL, bW=bW, test='positive')
sig_variance = series.significance(
    indicator='var', n=1000, detrend=True, wL=wL, bW=bW, test='positive')
print(sig_variance.pvalue)

print(f'Lag-1 autocorrelation p-value: {sig_pearson.pvalue["Sample series"]}')
print(f'Variance p-value: {sig_variance.pvalue["Sample series"]}')
sig_pearson.plot()
plt.show()
sig_variance.plot()
plt.show()


rob = series.robustness(indicators=['pearsonc', 'var'])
print(rob['Sample series']['pearsonc'])

rob.plot(vmin=0.1, cmap='viridis')
plt.show()

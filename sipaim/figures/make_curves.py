import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Per-epoch trajectories from result/pem_seed_*_2020/metrics.csv (representative seed 2020)
panc = dict(ep=[0,1,2],           dice=[82.89,83.97,84.01],           hd=[7.81,6.03,5.69])
la5  = dict(ep=[0,1,2,3,4,5],     dice=[87.32,87.63,88.31,88.44,88.57,88.75], hd=[13.91,8.38,7.75,7.83,7.67,7.84])
la10 = dict(ep=[0,1,2,3,4,5],     dice=[89.40,89.59,89.71,90.04,90.02,90.03], hd=[9.88,7.33,7.16,6.97,6.85,6.81])

# Okabe-Ito colorblind-safe palette
C = {"panc":"#0072B2","la5":"#D55E00","la10":"#009E73"}
M = {"panc":"o","la5":"s","la10":"^"}
L = {"panc":"Pancreas-CT 20%","la5":"LA 5%","la10":"LA 10%"}

plt.rcParams.update({
    "font.size":7,"axes.titlesize":7.5,"axes.labelsize":7,
    "xtick.labelsize":6.5,"ytick.labelsize":6.5,"legend.fontsize":6,
    "axes.linewidth":0.6,"lines.linewidth":1.2,"lines.markersize":3.2,
    "font.family":"serif",
})

fig, (a1, a2) = plt.subplots(1, 2, figsize=(3.45, 1.32))

for k, d in (("panc",panc),("la5",la5),("la10",la10)):
    a1.plot(d["ep"], d["dice"], marker=M[k], color=C[k], label=L[k], clip_on=False)
    a2.plot(d["ep"], d["hd"],   marker=M[k], color=C[k], clip_on=False)

a1.set_title(r"Dice (%) $\uparrow$"); a2.set_title(r"HD95 (vox) $\downarrow$")
for ax in (a1, a2):
    ax.set_xlabel("PEM fine-tuning epoch")
    ax.set_xticks([0,1,2,3,4,5])
    ax.grid(True, alpha=0.25, linewidth=0.4)
    ax.spines[["top","right"]].set_visible(False)
    ax.margins(x=0.03)

a1.legend(frameon=False, handlelength=1.3, borderaxespad=0.2, loc="lower right")
fig.tight_layout(pad=0.3, w_pad=1.0)
fig.savefig("/home/tals/Documents/PostHocEM/sipaim/figures/pem_curves.pdf", bbox_inches="tight")
print("saved pem_curves.pdf")

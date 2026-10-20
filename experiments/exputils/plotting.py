"""Plotting helpers for the paper figures and LaTeX / Markdown tables."""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import numpy as np
from scipy.stats import ttest_1samp


def set_paper_style():
    """Fonts used in the paper (falls back to DejaVu Sans when Arial is missing)."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "mathtext.fontset": "stix",
        "font.size": 9,
    })


def _finish(fig_path, show):
    if fig_path is not None:
        plt.savefig(fig_path)
        print(f"Figure saved to {fig_path}")
    if show:
        plt.show()
    plt.close()


def plot_final_compact_gradient(da, method_dict, nominal_coverage=0.95, fig_path=None, show=False):
    # 1. Prep Data
    avg = da.mean(dim="n_repts")
    flat_avg = avg.stack(setting=("alpha", "n"))
    
    alphas = da.alpha.values
    sample_sizes = da.n.values
    methods = da.methods.values
    x_indices = np.arange(len(flat_avg.setting))

    # 2. Figure Setup
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 4), sharex=True)
    colors = ['#de285f', '#457B9D', '#2A9D8F', '#8338EC', '#F4A261']
    
    # 3. Background Gradients & Alpha Labels
    for i, a_val in enumerate(alphas):
        start_idx = i * len(sample_sizes)
        end_idx = (i + 1) * len(sample_sizes)
        
        # Subtle blue gradient
        for j in range(len(sample_sizes)):
            intensity = 0.04 + (j / len(sample_sizes)) * 0.2
            ax1.axvspan(start_idx + j - 0.5, start_idx + j + 0.5, color='tab:blue', alpha=intensity, lw=0)
            ax2.axvspan(start_idx + j - 0.5, start_idx + j + 0.5, color='tab:blue', alpha=intensity, lw=0)
        
        ax1.axvline(end_idx - 0.5, color='black', lw=0.8, alpha=0.3)
        ax2.axvline(end_idx - 0.5, color='black', lw=0.8, alpha=0.3)
        
        mid = start_idx + (len(sample_sizes)-1)/2
        ax1.text(mid, 1.02, a_val, 
            transform=ax1.get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=11, fontweight='regular')

    # 4. Lines: one segment per (method, a) block so that blocks are not connected
    for k, m_name in enumerate(methods):
        # We handle labels manually so they don't repeat in the legend
        first_alpha = True 
        
        for i, a_val in enumerate(alphas):
            start_idx = i * len(sample_sizes)
            end_idx = (i + 1) * len(sample_sizes)
            
            # Slice data for THIS alpha block only
            # avg shape is (methods, metric, alpha, n)
            y_cov_block = avg.sel(methods=m_name, metric="coverage", alpha=a_val).values
            y_siz_block = avg.sel(methods=m_name, metric="set_size", alpha=a_val).values
            x_block = x_indices[start_idx:end_idx]

            # Label only on the first alpha block for the legend
            lbl = rf"{method_dict[m_name]}" if first_alpha else None
            
            ax1.plot(x_block, y_cov_block, marker='D', markersize=3.5, color=colors[k], 
                     label=lbl, linewidth=1.2, markerfacecolor='none', markeredgewidth=1)

            ax2.plot(x_block, y_siz_block, marker='D', markersize=3.5, color=colors[k], 
                     linewidth=1.2, markerfacecolor='none', markeredgewidth=1)
            
            first_alpha = False

    # 5. Styling and Scientific Ticks
    ax1.axhline(nominal_coverage, color='#1a1a1a', linestyle='--', lw=1.5, label='Nominal Level')
    ax1.set_ylabel("Empirical Coverage", fontsize=11)
    ax2.set_ylabel("Avg. Set Size", fontsize=11)
    
    scientific_n = [f"{n/1000:g}" for n in list(sample_sizes) * len(alphas)]
    ax2.set_xticks(x_indices)
    ax2.set_xticklabels(scientific_n, rotation=0, fontsize=9)
    ax2.set_xlabel(r"Sample Size ($\times 10^3$)", fontsize=11, labelpad=10)
    ax1.set_xlim(-0.5, len(x_indices) - 0.5)
    
    # Legend
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45), 
               ncol=len(methods)+1, frameon=False, fontsize=11)

    plt.subplots_adjust(top=0.87, bottom=0.15, left = 0.1, right = 0.98, hspace=0.25)
    _finish(fig_path, show)


def plot_combined_volcano_compass_final(da_length_ratio, da_volume_ratio, break_low=None, break_high=None,
                                        draw_threshold=True, fig_path=None, show=False):
    fig = plt.figure(figsize=(4, 2))
    
    # 1. Setup Data & Colors
    alphas = np.sort(da_length_ratio.alpha.values)
    sample_sizes = np.sort(da_length_ratio.n.values)
    num_ns = len(sample_sizes)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, num_ns))
    cmap_n = mcolors.ListedColormap(colors)
    norm_n = mcolors.BoundaryNorm(np.append(sample_sizes - 500, sample_sizes[-1] + 500), cmap_n.N)
    
    gs = gridspec.GridSpec(1, 2, width_ratios=[1.2, 1], wspace=0.25)
    
    # --- 2. VOLCANO PLOT PROCESSING ---
    plot_x, plot_y, plot_alpha, plot_n = [], [], [], []
    for a_val in alphas:
        for n_val in sample_sizes:
            raw_data = da_length_ratio.sel(alpha=a_val, n=n_val).values
            log2_ratios = np.log2(np.mean(raw_data, axis=0))
            res = ttest_1samp(log2_ratios, 0, alternative='greater')
            plot_x.append(np.mean(log2_ratios))
            plot_y.append(-np.log10(res.pvalue))
            plot_alpha.append(a_val); plot_n.append(n_val)
    
    plot_x, plot_y = np.array(plot_x), np.array(plot_y)

    # Gap Logic
    if break_low is not None and break_high is not None:
        main_max, out_min, has_gap = break_low, break_high, (break_low != break_high)
    else:
        sorted_x = np.sort(plot_x)
        diffs = np.diff(sorted_x)
        gap_indices = np.where(diffs > 2.0)[0]
        if len(gap_indices) > 0:
            idx = gap_indices[0]
            main_max, out_min, has_gap = sorted_x[idx] + 0.3, sorted_x[idx+1] - 0.3, True
        else:
            has_gap = False

    if has_gap:
        inner_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[0], wspace=0.08, width_ratios=[3, 1])
        ax_v_main = fig.add_subplot(inner_gs[0])
        ax_v_out = fig.add_subplot(inner_gs[1], sharey=ax_v_main)
        volc_axes = [ax_v_main, ax_v_out]
    else:
        ax_v_main = fig.add_subplot(gs[0])
        volc_axes = [ax_v_main]

    # marker size decreases with a (stronger shift = larger marker)
    if len(alphas) > 1:
        p_sizes = ((max(alphas) - np.array(plot_alpha)) / (max(alphas) - min(alphas))) * 35 + 5 
    else:
        # a single value of a
        p_sizes = np.full_like(plot_alpha, 80, dtype=float)
    

    for ax in volc_axes:
        ax.scatter(plot_x, plot_y, c=plot_n, cmap=cmap_n, norm=norm_n, s=p_sizes, 
                   alpha=0.9, edgecolors='black', linewidth=0.1, zorder=3)
        if draw_threshold:
            ax.axhline(-np.log10(0.05), color='black', linestyle='--', alpha=1, lw=1)
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.tick_params(labelsize=7)

    if has_gap:
        ax_v_main.set_xlim(min(plot_x) - 0.02, main_max)
        ax_v_out.set_xlim(out_min, max(plot_x) + 0.3)
        ax_v_out.tick_params(axis='y', labelleft=False, left=False); ax_v_out.spines['left'].set_visible(False)
        d = .02; kw = dict(color='black', clip_on=False, lw=1)
        ax_v_main.plot((1-d, 1+d), (-d, +d), transform=ax_v_main.transAxes, **kw)
        ax_v_out.plot((-d, +d), (-d, +d), transform=ax_v_out.transAxes, **kw)

    ax_v_main.set_ylabel(r"$-\log_{10}$(p-val)", fontsize=9)
    ax_v_main.set_xlabel(r"Mean $\log_2$Ratio (GE/LP)", fontsize=9)

    # --- 3. CIRCULAR BAR CHART OF vol(Omega) / vol(Omega_LP) ---
    ax_c = fig.add_subplot(gs[1], projection='polar')
    pos = ax_c.get_position()
    

    da_sorted = da_volume_ratio.sortby("alpha").sortby("n")
    inter_group_gap, gateway_gap = 2, 10 
    total_slots = len(alphas) * num_ns + (len(alphas)-1) * inter_group_gap + gateway_gap
    all_angles = np.linspace(0, 2 * np.pi, total_slots, endpoint=False)
    
    
    current_slot = 0
    for i, a_val in enumerate(alphas):
        # 1. Sort subset by n descending (Largest -> Smallest)
        subset = da_sorted.sel(alpha=a_val).sortby("n", ascending=False)
        
        means = subset.mean(dim="n_repts").values * 100
        stds = subset.std(dim="n_repts").values * 100
        sector_angles = all_angles[current_slot : current_slot + num_ns]
        mid_angle = np.mean(sector_angles)
        
        # 2. Use reversed colors [::-1] so the first (largest) bar gets the darkest blue
        ax_c.bar(sector_angles, means, width=all_angles[1]-all_angles[0], 
                 color=colors[::-1], edgecolor='black', linewidth=0.1, alpha=0.9, zorder=3)

        ax_c.errorbar(sector_angles, means, yerr=stds, fmt='none', ecolor='black', lw=0.2, capsize=1.2, capthick=.2, zorder=4)
        ax_c.text(mid_angle, -15, f"{a_val:g}", fontsize=8, fontweight='regular', ha='center', va='center')
        current_slot += num_ns + inter_group_gap

    label_angle = all_angles[total_slots - (gateway_gap // 2)]
    ax_c.set_theta_offset(np.pi / 2); ax_c.set_theta_direction(-1)
    ax_c.spines['polar'].set_visible(False)
    
    for r in [0, 25, 50, 75, 100]:
        color, style = ('black', '-') if r == 100 else (('black', ':') if r == 0 else ('dimgray', '--'))
        ax_c.plot(np.linspace(0, 2*np.pi, 500), np.ones(500)*r, color=color, linestyle=style, lw=0.5, alpha=0.8, zorder=2)
        ax_c.text(label_angle, r, f"{r}%", color="black", fontsize=6, ha='center', va='center', 
                  backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='none', pad=0.1, alpha=1))

    ax_c.text(np.pi*1.65, -58, r"$a$", color="black", fontsize=11, ha='center', va='center', 
                  backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='none', pad=0.1, alpha=1))

    ax_c.set_xticks([]); ax_c.set_yticks([]); ax_c.grid(False)
    ax_c.set_ylim(-60, 100)

    # --- 4. TOP HORIZONTAL LEGENDS ---
    
    # --- ALPHA LEGEND (TOP LEFT - CIRCLES) ---
    if len(alphas) > 1:
        max_a, min_a = max(alphas), min(alphas)

        a_h = [plt.scatter([], [], s=((max_a-a)/(max_a-min_a))*35 + 5 , color='none', 
                           edgecolors='black', linewidth=0.5, label=f"{a:g}") for a in alphas]
        
        leg_a = fig.legend(handles=a_h, loc='lower left', ncol=len(alphas), 
                           bbox_to_anchor=(0.00, 0.82), 
                           fontsize=5.5, frameon=False, title=r"$a$", 
                           title_fontsize=11, handletextpad=-0.1, columnspacing=0.5)

    # --- SAMPLE SIZE LEGEND (TOP RIGHT - SQUARES) ---
    # We use 's' for square markers to distinguish from the alpha circles
    n_h = [plt.scatter([], [], marker='s', s=20, color=colors[i], 
                       edgecolors='black', linewidth = 0.2, label=f"{sample_sizes[i]/1000:g}") for i in range(num_ns)]
    
    leg_n = fig.legend(handles=n_h, loc='lower right', ncol=len(sample_sizes), 
                       bbox_to_anchor=(1, 0.82), 
                       fontsize=5.5, frameon=False, title= r"Sample Size ($\times 10^3$)", 
                       title_fontsize=8, handletextpad=0.1, columnspacing=0.5)

    plt.subplots_adjust(top=0.80, bottom=0.2, left=0.15, right=0.99)
    ax_c.set_position([0.57, -0.32, 0.42, 1.5])
    _finish(fig_path, show)


def fig1_plot_classes(ax, data, title, clors, pos_title=(0.5, 1.05)):
    """Scatter plot of a two-class sample (Figure 1 panels).

    ``data`` has columns (x1, x2, y); ``clors`` gives the colors of class 0
    and class 1.  Points are drawn in a fixed random order so that neither
    class is always on top.
    """
    data = np.random.RandomState(1652).permutation(data)
    x_min, x_max = data[:, 0].min(), data[:, 0].max()
    y_min, y_max = data[:, 1].min(), data[:, 1].max()
    x_buf, y_buf = (x_max - x_min) * 0.1, (y_max - y_min) * 0.1

    point_colors = [clors[i] for i in data[:, 2].astype(int)]
    ax.scatter(data.T[0], data.T[1], marker="s", facecolors="none", edgecolors=point_colors,
               s=4.5, linewidth=0.5, alpha=0.6)
    ax.text(pos_title[0], pos_title[1], title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=9, fontweight="regular")

    ax.set_xlim(x_min - x_buf, x_max + x_buf)
    ax.set_ylim(y_min - y_buf, y_max + y_buf)
    ax.set_xticks([])
    ax.set_yticks([])
    off_ax = ["top", "right"] if "100" in title else ["top", "left"]
    if "Calib" in title or "Test" in title:
        off_ax = ["top"]
    for spine in off_ax:
        ax.spines[spine].set_visible(False)


def fig1_plot_weight_boxes(ax, w_estimates, title, colors):
    """Boxplots of the estimated weights of the two classes (array of shape (n_repts, 2))."""
    class_0_color, class_1_color = colors
    # ``tick_labels`` requires matplotlib >= 3.9
    bp = ax.boxplot(w_estimates, tick_labels=[r"$\widehat{\omega}_1$", r"$\widehat{\omega}_2$"],
                    widths=0.6, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], [class_0_color, class_1_color]):
        patch.set_facecolor(color)
        patch.set_alpha(1)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.0)
    for i, color in enumerate([class_0_color, class_1_color]):
        plt.setp(bp["whiskers"][i * 2:i * 2 + 2], color=color, lw=1.5)
        plt.setp(bp["caps"][i * 2:i * 2 + 2], color=color, lw=1.5)
        plt.setp(bp["medians"][i], color="white", lw=1)

    ax.set_title(title, fontsize=9, pad=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(-0.5, 5.6)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def plot_result_box(ax, data, pos, color):
    """One boxplot of ``data`` at x-position ``pos`` (prediction-set sizes in Figure 1)."""
    flier_props = dict(marker="d", markersize=4, markerfacecolor=color, markeredgecolor="none", alpha=1.0)
    bp = ax.boxplot(data, positions=[pos], widths=0.5, patch_artist=True, showfliers=True,
                    flierprops=flier_props)
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for median in bp["medians"]:
        median.set(color="white", linewidth=1)
    plt.setp(bp["whiskers"], color=color, linewidth=1.5, linestyle="--")
    plt.setp(bp["caps"], color=color, linewidth=1.5)

    ax.set_yticks([1.25, 1.50, 1.75])
    ax.set_yticklabels(["1.25", "1.50", "1.75"])
    return bp


def plot_coverage_histograms(ax, data_list, linstyls, labels, colors, lgn_pos="upper left",
                             hist_range=(0.7, 1.0), n_bins=40):
    """Step histograms of the empirical coverage, one per array in ``data_list``,
    with a vertical line at each mean."""
    for data, label, color, lsty in zip(data_list, labels, colors, linstyls):
        ax.hist(data, bins=n_bins, range=hist_range, density=False, histtype="step",
                color=color, label=label, linewidth=1, zorder=2)
        ax.axvline(data.mean(), lw=1, color=color, linestyle=lsty)

    ax.set_xlabel("Empirical Coverage", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_yticks([0, 40, 80, 120])
    ax.set_yticklabels([0, 40, 80, 120])
    ax.legend(fontsize=8, frameon=False, loc=lgn_pos)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=0))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)


def _cell(means, stds, i, j, digits=2):
    if stds is not None and len(stds) > 0:
        return f"{means[i, j]:.{digits}f} ({stds[i, j]:.{digits}f})"
    return f"{means[i, j]:.{digits}f}"


def generate_stats_latex_table(means, stds, headers, row_names, digits=2):
    """LaTeX tabular with "mean (std)" cells.

    ``means`` and ``stds`` have shape (len(row_names), n); ``headers`` has
    n + 1 entries (the first one labels the row names).
    """
    lines = ["\\begin{tabular}{|l|" + "c" * means.shape[1] + "|}", "\\hline",
             " & ".join(headers) + " \\\\ \\hline"]
    for i, name in enumerate(row_names):
        lines.append(" & ".join([name] + [_cell(means, stds, i, j, digits) for j in range(means.shape[1])])
                     + " \\\\")
    lines += ["\\hline", "\\end{tabular}"]
    return "\n".join(lines)


def generate_stats_markdown_table(means, stds, headers, row_names, digits=2):
    """Markdown table with "mean (std)" cells (``stds`` may be None)."""
    lines = ["| " + " | ".join(headers) + " |", "|" + " --- |" * (means.shape[1] + 1)]
    for i, name in enumerate(row_names):
        lines.append("| " + " | ".join([name] + [_cell(means, stds, i, j, digits)
                                               for j in range(means.shape[1])]) + " |")
    return "\n".join(lines)

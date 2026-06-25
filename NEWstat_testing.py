import pandas as pd
from scipy import stats
import os

# Point this at the new segmented CSV produced by sam3.py
FILE_NAME = "sam3_s10_seg4_06-25_13h33m.csv"  # <-- edit to your actual filename

# 1. Load the data
current_thesis_dir = os.path.dirname(os.path.abspath(__file__))
file_location = os.path.join(current_thesis_dir, "results", FILE_NAME)
df = pd.read_csv(file_location)

# 2. Clean the text columns
df['type'] = df['type'].astype(str).str.strip().str.lower()
df['filename'] = df['filename'].astype(str).str.lower()

# 3. Build the unique ID per (video, segment)
df['video_id'] = df['filename'].str.extract(r'^(\d+)')

# segment_id now comes from a real column written by sam3.py.
# Fallback to the old filename-parsing scheme if you load a legacy CSV.
if 'segment_id' in df.columns:
    df['segment_id'] = df['segment_id'].astype(str).str.strip()
else:
    df['segment_id'] = df['filename'].str.extract(r'_(\d+)\.avi$').fillna('0')

df['uid'] = df['video_id'].astype(str) + "_" + df['segment_id'].astype(str)

target_metric = 'hit_rate_perc'   # switch to 'avg_drift_pixl' to test spatial drift
effects = ['gaussian', 'motion', 'overexposure', 'underexposure', 'grain', 'occlusion']

print(f"=== STATISTICAL ANALYSIS FOR: {target_metric.upper()} ===")

# 4. Isolate the baseline data using the exact UID (video + segment)
baseline_df = df[df['type'] == 'original'].set_index('uid')[target_metric]

for effect in effects:
    print(f"\n" + "=" * 45)
    print(f" TESTING ISOLATED EFFECT: {effect.upper()}")
    print(f"=============================================")

    effect_data = df[df['filename'].str.contains(effect)].copy()

    mild_df = effect_data[effect_data['filename'].str.contains('mild')].set_index('uid')[target_metric]
    medium_df = effect_data[effect_data['filename'].str.contains('medium')].set_index('uid')[target_metric]
    severe_df = effect_data[effect_data['filename'].str.contains('severe')].set_index('uid')[target_metric]

    # Align the modified segments together using their unique (video+segment) IDs
    combined = pd.DataFrame({
        'mild': mild_df,
        'medium': medium_df,
        'severe': severe_df
    }).dropna()

    if combined.empty:
        print(f"-> Skipping {effect.upper()} because data arrays are empty.")
        continue

    # Map the matching baseline segment onto each row, then drop unmatched
    combined['baseline'] = combined.index.map(baseline_df)
    combined = combined.dropna(subset=['baseline'])

    if combined.empty:
        print(f"-> Skipping {effect.upper()} because no baseline segments matched.")
        continue

    baseline_arr = combined['baseline'].values
    mild_arr = combined['mild'].values
    medium_arr = combined['medium'].values
    severe_arr = combined['severe'].values

    n_blocks = len(baseline_arr)
    print(f"Paired blocks (video x segment): {n_blocks}")

    # 5. The Friedman Test (repeated-measures, non-parametric)
    stat, p_friedman = stats.friedmanchisquare(baseline_arr, mild_arr, medium_arr, severe_arr)
    print(f"\nFriedman p-value: {p_friedman:.4f}")

    # 6. Post-Hoc Test (Wilcoxon Signed-Rank), Holm-corrected across the 3 comparisons
    if p_friedman < 0.05:
        print("-> SIGNIFICANT OVERALL CONDITION EFFECT. Comparing each severity to baseline...")

        _, p_mild = stats.wilcoxon(baseline_arr, mild_arr, zero_method='zsplit')
        _, p_med = stats.wilcoxon(baseline_arr, medium_arr, zero_method='zsplit')
        _, p_sev = stats.wilcoxon(baseline_arr, severe_arr, zero_method='zsplit')

        # Holm correction across the 3 post-hoc comparisons
        labels = ['Mild', 'Medium', 'Severe']
        raw = [p_mild, p_med, p_sev]
        order = sorted(range(3), key=lambda i: raw[i])
        holm = [None, None, None]
        for rank, i in enumerate(order):
            holm[i] = min(1.0, raw[i] * (3 - rank))
        # enforce monotonicity
        running = 0.0
        for i in order:
            running = max(running, holm[i])
            holm[i] = running

        for i in range(3):
            print(f"   Baseline vs {labels[i]:<6} raw p: {raw[i]:.4f}   Holm-adj: {holm[i]:.4f}")

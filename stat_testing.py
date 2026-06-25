import pandas as pd
from scipy import stats
import os

FILE_NAME = "sam3_s10_06-22_19h05m.csv"

# 1. Load the data
current_thesis_dir = os.path.dirname(os.path.abspath(__file__))
file_location = os.path.join(current_thesis_dir, "results", FILE_NAME)
df = pd.read_csv(file_location)

# 2. Clean the text columns
df['type'] = df['type'].astype(str).str.strip().str.lower()
df['filename'] = df['filename'].astype(str).str.lower()

# 3. Extract IDs
df['video_id'] = df['filename'].str.extract(r'^(\d+)')
# Extract segment ID (e.g., '_6.avi' -> '6'). Fill with '0' for the unsegmented originals.
df['segment_id'] = df['filename'].str.extract(r'_(\d+)\.avi$').fillna('0')

# --- NEW: Create a global UID for EVERY row upfront ---
df['uid'] = df['video_id'] + "_" + df['segment_id']

target_metric = 'hit_rate_perc'
effects = ['gaussian', 'motion', 'overexposure', 'underexposure', 'grain', 'occlusion']

print(f"=== STATISTICAL ANALYSIS FOR: {target_metric.upper()} ===")

# 4. Isolate the baseline data using the exact UID, not just video_id
baseline_df = df[df['type'] == 'original'].set_index('uid')[target_metric]

for effect in effects:
    print(f"\n" + "="*45)
    print(f" TESTING ISOLATED EFFECT: {effect.upper()}")
    print(f"=============================================")
    
    effect_data = df[df['filename'].str.contains(effect)].copy()
    
    # UID is already created globally above, so we just set the index
    mild_df = effect_data[effect_data['filename'].str.contains('mild')].set_index('uid')[target_metric]
    medium_df = effect_data[effect_data['filename'].str.contains('medium')].set_index('uid')[target_metric]
    severe_df = effect_data[effect_data['filename'].str.contains('severe')].set_index('uid')[target_metric]
    
    # Align the modified segments together using their unique segment IDs
    combined = pd.DataFrame({
        'mild': mild_df,
        'medium': medium_df,
        'severe': severe_df
    }).dropna() 
    
    if combined.empty:
        print(f"-> Skipping {effect.upper()} because data arrays are empty.")
        continue

    # --- NEW: Map the baseline using the combined.index (which is the uid) ---
    # We no longer need to extract the base video_id here!
    combined['baseline'] = combined.index.map(baseline_df)
    
    # Drop any rows where the baseline mapping failed
    combined = combined.dropna(subset=['baseline'])
    
    # Convert to arrays for the statistical tests
    baseline_arr = combined['baseline'].values
    mild_arr = combined['mild'].values
    medium_arr = combined['medium'].values
    severe_arr = combined['severe'].values
    
    print(f"Segments found -> Baseline: {len(baseline_arr)}, Mild: {len(mild_arr)}, Medium: {len(medium_arr)}, Severe: {len(severe_arr)}")
    
    # 5. The Friedman Test
    stat, p_friedman = stats.friedmanchisquare(baseline_arr, mild_arr, medium_arr, severe_arr)
    print(f"\nFriedman p-value: {p_friedman:.4f}")
    
    # 6. Post-Hoc Test (Wilcoxon Signed-Rank Test)
    if p_friedman < 0.05: 
        print("-> SIGNIFICANT OVERALL CONDITION EFFECT DETECTED! Comparing each severity to the baseline...")
        # zero_method='zsplit' handles cases where the baseline and modified arrays have identical values
        _, p_base_mild = stats.wilcoxon(baseline_arr, mild_arr, zero_method='zsplit')
        _, p_base_med = stats.wilcoxon(baseline_arr, medium_arr, zero_method='zsplit')
        _, p_base_sev = stats.wilcoxon(baseline_arr, severe_arr, zero_method='zsplit')
        
        print(f"   Baseline vs Mild p-value:   {p_base_mild:.4f}")
        print(f"   Baseline vs Medium p-value: {p_base_med:.4f}")
        print(f"   Baseline vs Severe p-value: {p_base_sev:.4f}")
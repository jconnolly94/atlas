import pandas as pd
import os

# --- Configuration ---
run_id = "20250331_001013_AllAgents_LaneLevel_EarlyTerm_adf81491"
base_dir = 'data/runs'
# --- End Configuration ---

run_path = os.path.join(base_dir, run_id)
data_path = os.path.join(run_path, 'data')

step_file = os.path.join(data_path, 'step_data.parquet')
episode_file = os.path.join(data_path, 'episode_data.parquet')

print(f"--- Reading Episode Data: {episode_file} ---")
try:
    if os.path.exists(episode_file):
        episode_df = pd.read_parquet(episode_file)
        print("Episode Data Info:")
        episode_df.info() # Shows column types and non-null counts
        print("\nEpisode Data Shape:", episode_df.shape) # Shows (rows, columns)
        print("\nEpisode Data Head:\n", episode_df.head())
        print("\nEpisode Data Tail:\n", episode_df.tail())
        print("\nUnique Agents in Episode Data:", episode_df['agent_type'].unique())
        print("\nUnique Episodes in Episode Data:", episode_df['episode'].unique())
    else:
        print(f"Episode file not found: {episode_file}")
except Exception as e:
    print(f"Error reading episode file: {e}")

print(f"\n--- Reading Step Data: {step_file} ---")
try:
    if os.path.exists(step_file):
        step_df = pd.read_parquet(step_file)
        print("Step Data Info:")
        step_df.info()
        print("\nStep Data Shape:", step_df.shape)
        print("\nStep Data Head:\n", step_df.head())
        print("\nStep Data Tail:\n", step_df.tail())
        # Check distinct agents/episodes if the file is large
        if len(step_df) > 1000: # Avoid iterating huge DFs unless needed
            print("\nUnique Agents in Step Data:", step_df['agent_type'].unique())
            print("\nUnique Episodes in Step Data:", step_df['episode'].unique())
        print("\nMax Step Number:", step_df['step'].max())
    else:
        print(f"Step file not found: {step_file}")
except Exception as e:
    print(f"Error reading step file: {e}")
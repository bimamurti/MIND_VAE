import os
import numpy as np
from collections import defaultdict
from pathlib import Path

def count_pedestrians_per_frame(dataset_names, data_folder="data/Long_seq"):
    """
    Count average pedestrians per frame for given datasets.
    
    Args:
        dataset_names: list of dataset names
        data_folder: path to the data folder
    
    Returns:
        dict with statistics for each dataset
    """
    results = {}
    
    for dataset_name in dataset_names:
        dataset_path = os.path.join(data_folder, dataset_name, "train")
        
        if not os.path.exists(dataset_path):
            print(f"Warning: Path does not exist: {dataset_path}")
            results[dataset_name] = None
            continue
        
        # Find all CSV files
        csv_files = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(".csv"):
                    csv_files.append(os.path.join(root, file))
        
        if not csv_files:
            print(f"Warning: No CSV files found in {dataset_path}")
            results[dataset_name] = None
            continue
        
        print(f"\nProcessing {dataset_name} ({len(csv_files)} files)...")
        
        # Count pedestrians per frame across all CSV files
        frame_agent_counts = defaultdict(set)  # frame_time -> set of agent IDs
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if not parts:
                            continue
                        
                        try:
                            t = int(float(parts[0]))  # frame timestamp
                            idx = int(float(parts[1]))  # agent ID
                            frame_agent_counts[t].add(idx)
                        except (ValueError, IndexError):
                            continue
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        if not frame_agent_counts:
            print(f"Warning: No data found in {dataset_name}")
            results[dataset_name] = None
            continue
        
        # Calculate statistics
        pedestrian_counts = list(frame_agent_counts.values())
        pedestrian_per_frame = np.array([len(agents) for agents in pedestrian_counts])
        
        stats = {
            'total_frames': len(pedestrian_per_frame),
            'total_unique_agents': len(set().union(*frame_agent_counts.values())),
            'avg_pedestrians_per_frame': np.mean(pedestrian_per_frame),
            'std_pedestrians_per_frame': np.std(pedestrian_per_frame),
            'min_pedestrians_per_frame': np.min(pedestrian_per_frame),
            'max_pedestrians_per_frame': np.max(pedestrian_per_frame),
            'median_pedestrians_per_frame': np.median(pedestrian_per_frame),
        }
        
        results[dataset_name] = stats
        
        # Print results for this dataset
        print(f"  Total frames: {stats['total_frames']}")
        print(f"  Total unique agents: {stats['total_unique_agents']}")
        print(f"  Average pedestrians per frame: {stats['avg_pedestrians_per_frame']:.2f}")
        print(f"  Std dev: {stats['std_pedestrians_per_frame']:.2f}")
        print(f"  Min: {stats['min_pedestrians_per_frame']:.0f}, Max: {stats['max_pedestrians_per_frame']:.0f}")
        print(f"  Median: {stats['median_pedestrians_per_frame']:.2f}")
    
    return results

def print_summary(results):
    """Print summary table of all datasets"""
    print("\n" + "="*100)
    print("SUMMARY TABLE")
    print("="*100)
    print(f"{'Dataset':<20} {'Frames':<10} {'Unique Agents':<15} {'Avg Ped/Frame':<15} {'Std Dev':<10} {'Min':<5} {'Max':<5}")
    print("-"*100)
    
    for dataset_name, stats in results.items():
        if stats is None:
            print(f"{dataset_name:<20} {'N/A':<10}")
        else:
            print(f"{dataset_name:<20} {stats['total_frames']:<10} {stats['total_unique_agents']:<15} "
                  f"{stats['avg_pedestrians_per_frame']:<15.2f} {stats['std_pedestrians_per_frame']:<10.2f} "
                  f"{stats['min_pedestrians_per_frame']:<5.0f} {stats['max_pedestrians_per_frame']:<5.0f}")

if __name__ == "__main__":
    # Define datasets
    # datasets = [
    #     "data90g3",
    #     "data90d7",
    #     "dataMOT212",
    #     "data120b2",
    # ]
    # datasets=[
    #     "crossing_90_d_7",
    #     "crossing_90_g_3",
    #     "crossing_120_b_02",
    #     "MOT212"
    # ]
    datasets=[
        "duri_morning_world",
        "duri_evening_world",
        "vredeburg_world",
        "duri_platform1_world",
        "paisley_world"
    ]
    # Change to project directory if needed
    homefolder = '/home/USER/Documents/projects/MIND_VAE/'
    os.chdir(homefolder)
    
    # Count pedestrians
    #results = count_pedestrians_per_frame(datasets,data_folder="data/julichmotfull")
    results = count_pedestrians_per_frame(datasets,data_folder="data/crowdTraj World/dataset GT world CL")
    # Print summary
    print_summary(results)
    
    # Save results to file
    output_file = "pedestrian_counts.txt"
    with open(output_file, 'w') as f:
        f.write("PEDESTRIAN COUNT STATISTICS\n")
        f.write("="*100 + "\n\n")
        
        for dataset_name, stats in results.items():
            f.write(f"Dataset: {dataset_name}\n")
            if stats is None:
                f.write("  No data found\n")
            else:
                f.write(f"  Total frames: {stats['total_frames']}\n")
                f.write(f"  Total unique agents: {stats['total_unique_agents']}\n")
                f.write(f"  Average pedestrians per frame: {stats['avg_pedestrians_per_frame']:.2f}\n")
                f.write(f"  Std dev: {stats['std_pedestrians_per_frame']:.2f}\n")
                f.write(f"  Min: {stats['min_pedestrians_per_frame']:.0f}\n")
                f.write(f"  Max: {stats['max_pedestrians_per_frame']:.0f}\n")
                f.write(f"  Median: {stats['median_pedestrians_per_frame']:.2f}\n")
            f.write("\n")
    
    print(f"\nResults saved to {output_file}")

import psutil
import time
import csv
import sys
import os

def monitor_resources(interval=1.0, duration=60, output_file="resource_usage.csv"):
    """
    Monitors CPU and RAM usage and logs them to a CSV file.
    Runs for `duration` seconds, sampling every `interval` seconds.
    """
    print(f"Starting resource monitor. Logging to {output_file} for {duration} seconds...")
    
    # Create or overwrite the CSV
    with open(output_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "CPU_Usage_Percent", "RAM_Usage_MB"])
        
        start_time = time.time()
        while time.time() - start_time < duration:
            current_time = time.strftime('%H:%M:%S')
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # RAM usage in MB
            memory = psutil.virtual_memory()
            ram_mb = memory.used / (1024 * 1024)
            
            print(f"[{current_time}] CPU Usage: {cpu_percent}% | RAM Usage: {ram_mb:.2f} MB")
            
            writer.writerow([current_time, cpu_percent, f"{ram_mb:.2f}"])
            time.sleep(interval)
            
    print(f"Resource monitoring complete. Data saved to {output_file}")

if __name__ == "__main__":
    try:
        duration = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    except ValueError:
        duration = 60
        
    monitor_resources(duration=duration)

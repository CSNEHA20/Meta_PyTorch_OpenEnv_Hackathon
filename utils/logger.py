import csv
import os

class TrainingLogger:
    """Logs training metrics to a CSV file."""
    
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        # Write header if file is new
        if not os.path.exists(self.filename):
            with open(self.filename, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "reward", "epsilon"])

    def log(self, episode, reward, epsilon):
        """Appends a new row of metrics to the CSV file."""
        with open(self.filename, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([episode, reward, epsilon])

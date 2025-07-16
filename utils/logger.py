import os
import csv
from datetime import datetime
class Logger:
    def __init__(self, log_dir="logs", filename=None, headers=None):
        os.makedirs(log_dir, exist_ok=True)
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"log_{timestamp}.csv"
        self.filepath = os.path.join(log_dir, filename)
        self.headers = headers if headers else ["epoch", "loss", "ssim", "lpips", "edge"]
        self._initialize_log_file()
    def _initialize_log_file(self):
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
    def log(self, values: dict):
        """
        Log a dictionary of values for current epoch/step.
        Example:
            logger.log({
                "epoch": 1,
                "loss": 0.1234,
                "ssim": 0.9123,
                "lpips": 0.3456,
                "edge": 0.0567
            })
        """
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            row = [values.get(header, "") for header in self.headers]
            writer.writerow(row)
    def get_log_path(self):
        return self.filepath

import time


class Timer:
    def __init__(self, name="Timer"):
        self.name = name
        self.start_time = time.time()  # Seconds

        # For averages
        self.segment_total = 0
        self.segment_count = 0

    def start(self):
        self.start_time = time.time()

    def get_time_elapsed(self):
        return time.time() - self.start_time

    def print_time_elapsed(self, format_function=None):
        if format_function is None:
            format_function = self.format_minutes

        time_elapsed = self.get_time_elapsed()
        time_format = format_function(time_elapsed)
        print(f"{self.name} - Time elapsed: {time_format}")

    def predict_time_remaining(self, completed, total):
        completion = completed / total
        time_left_estimate = (self.get_time_elapsed() / completion * (1 - completion))
        return time_left_estimate

    def print_time_remaining(self, completed, total, display_steps=None, format_function=None):
        if format_function is None:
            format_function = self.format_minutes

        if display_steps is None or completed % (total // (display_steps+1)) == 0:
            time_remaining = self.predict_time_remaining(completed, total)
            time_format = format_function(time_remaining)
            print(f"{self.name} - {completed / total:.2%} complete. Estimated time remaining: {time_format}")

    def record_segment(self):
        time_elapsed = self.get_time_elapsed()
        self.segment_total += time_elapsed
        self.segment_count += 1

    def get_average_segment_time(self):
        return self.segment_total / self.segment_count

    def print_average_segment_time(self, format_function=None):
        if format_function is None:
            format_function = self.format_seconds

        average = self.get_average_segment_time()
        time_format = format_function(average)
        print(f"{self.name} - Average time: {time_format} ({self.segment_count} samples)")

    @staticmethod
    def format_minutes(time_seconds):
        minutes = time_seconds // 60
        seconds = time_seconds % 60
        return f"{minutes}m{seconds:.2f}s"

    @staticmethod
    def format_seconds(time_seconds):
        return f"{time_seconds}s"

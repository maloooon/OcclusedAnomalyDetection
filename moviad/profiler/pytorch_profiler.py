import contextlib
from datetime import datetime
import os
from logging import Logger

import numpy as np
import torch
import cProfile
import pstats
import io
from memory_profiler import memory_usage


class Profiler:
    def __init__(self):
        self.gpu_allocated = []
        self.gpu_cached = []
        self.start_time = None
        self.end_time = None
        self._profiling_active = False
        self._profiler = None
        self.memory_usage = []
        self.title = None

    def start_profiling(self, use_cuda=False, title: str = None):
        """Start profiling session before inference begins"""
        self.start_time = datetime.now()
        self._profiling_active = True
        self._profiler = cProfile.Profile()
        self._profiler.enable()
        self.title = title
        return self

    @contextlib.contextmanager
    def profile_step(self):
        """Context manager to profile a single inference step"""
        if not self._profiling_active:
            yield
            return

        # Track GPU memory if available
        if torch.cuda.is_available():
            before_allocated = torch.cuda.memory_allocated(0) / 1024 ** 2  # MB
            before_cached = torch.cuda.memory_reserved(0) / 1024 ** 2  # MB

        # Track memory usage for this step
        mem_usage_start = memory_usage(-1, interval=0.1, timeout=0.1)

        try:
            yield
        finally:
            # Collect memory metrics after the step
            mem_usage = memory_usage(-1, interval=0.1, timeout=0.1)
            self.memory_usage.append(max(mem_usage))

            # Collect GPU metrics after the step
            if torch.cuda.is_available():
                self.gpu_allocated.append(torch.cuda.memory_allocated() / 1024 ** 2)  # MB
                self.gpu_cached.append(torch.cuda.memory_reserved() / 1024 ** 2)  # MB

    def end_profiling(self, log_file="profiler_logs.log"):
        """End profiling and save aggregated results"""
        if not self._profiling_active:
            return

        # Stop profiler
        self._profiler.disable()
        self.end_time = datetime.now()
        self._profiling_active = False

        # Save results
        self.save(log_file)

    def save(self, log_file="profiler_logs.log") -> None:
        # Generate profiling stats
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(self._profiler, stream=s).sort_stats(sortby)
        ps.print_stats(20)  # Top 20 functions
        stats_str = s.getvalue()

        # Generate report timestamp
        timestamp = self.end_time.strftime('%Y%m%d_%H%M%S')

        log_content = [
            f"\n{'=' * 50}",
            f"{self.title if self.title is not None else ""} Results - {self.end_time.strftime('%Y-%m-%d %H:%M:%S')}",
        ]

        if self.memory_usage:
            log_content.extend([
                f"\nMemory Usage Statistics:",
                f"Mean Memory: {np.mean(self.memory_usage):.2f} MiB (std: {np.std(self.memory_usage):.2f} MiB)",
                f"Max Memory: {np.max(self.memory_usage):.2f} MiB",
            ])

        if self.gpu_allocated:
            log_content.extend([
                f"\nGPU Memory Statistics:",
                f"Mean Allocated: {np.mean(self.gpu_allocated):.2f} MB (std: {np.std(self.gpu_allocated):.2f} MB)",
                f"Max Allocated: {np.max(self.gpu_allocated):.2f} MB",
                f"Mean Cached: {np.mean(self.gpu_cached):.2f} MB (std: {np.std(self.gpu_cached):.2f} MB)",
                f"Max Cached: {np.max(self.gpu_cached):.2f} MB",
            ])

        log_content.extend([
            f"\nProfiling duration: {(self.end_time - self.start_time).total_seconds():.2f} seconds",
            f"\nTop 20 function calls:\n",
            stats_str,
            f"{'=' * 50}\n"
        ])


        # Save detailed profile output
        with open(log_file, 'a') as f:
            f.write('\n'.join(log_content))
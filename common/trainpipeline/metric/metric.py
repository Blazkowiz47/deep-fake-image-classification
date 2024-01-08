"""
Contains the metric factory.
"""


from importlib import import_module


def get_metric(metric: str):
    """
    Fetches metric.
    """
    module = import_module(f"common.trainpipeline.metric.{metric}")
    return getattr(module, "Metric", None)

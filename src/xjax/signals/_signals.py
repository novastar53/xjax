from blinker import signal

__all__ = [
    "train_epoch_completed",
    "train_epoch_started",
]

train_epoch_started = signal("train_epoch_started")
"""Signals start of epoch.

Keyword Args:
    epoch (int): the epoch
    elapsed (float): elapsed time since training started
"""

train_epoch_completed = signal("train_epoch_completed")
"""Signals completion of epoch.

Keyword Args:
    epoch (int): the epoch
    loss (float): the training loss of last batch in epoch
    validation_loss (float): the validation loss of teh last batch in the epoch
    elapsed (float): elapsed time since training started
"""

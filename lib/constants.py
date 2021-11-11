from pathlib import Path

__all__ = ['DATA', 'MONTHS']

DATA = Path('./data')

MONTHS_NAME = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTHS = [(str(i)) if i >= 10 else f"0{i}" for i in range(1, 13)]
MONTHS = {month: name for month, name in zip(MONTHS, MONTHS_NAME)}
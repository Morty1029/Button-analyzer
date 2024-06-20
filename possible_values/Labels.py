from dataclasses import dataclass
from decorators.patterns import singleton


@dataclass
@singleton
class Labels:
    def __init__(self):
        self.labels = [
            'color',
            'font-family',
            'font-style',
            'font-weight',
            'text-align',
            'text-transform',
            'background-color',
            'border-color',
            'border-style'
        ]

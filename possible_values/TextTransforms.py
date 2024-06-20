from enum import Enum


class TextTransforms(Enum):
    NONE = 'none'
    CAPITALIZE = 'capitalize'
    UPPERCASE = 'uppercase'
    LOWERCASE = 'lowercase'


class LabelsTextTransforms(Enum):
    NONE = 0
    CAPITALIZE = 1
    UPPERCASE = 2
    LOWERCASE = 3

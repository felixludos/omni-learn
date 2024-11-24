from .imports import *

__all__ = ['human_size', 'fixed_width_format_positive', 'fixed_width_format_value', 'display_json']



def human_size(size: int):
    return naturalsize(size, gnu=True, format='%.0f').replace('G', 'M').replace('T', 'B')



def fixed_width_format_positive(val: float, width: int) -> str:
    return fixed_width_format_value(val, width, force_positive=True)



def fixed_width_format_value(val: float, width: int, force_positive: bool = False) -> str:
    # https://chatgpt.com/c/6721c2ed-98ec-8005-b967-9b81f6b8d5f1

    if force_positive:
        s = fixed_width_format_value(val, width+1, force_positive=False)
        if s[0] == ' ':
            return s[1:]
        return s

    import math

    if width < 1:
        raise ValueError("Width must be at least 1.")

    # Determine the sign and adjust width accordingly
    sign = '-' if val < 0 else ' '
    number_width = width - 1  # Account for the sign character

    abs_val = abs(val)

    # Try fixed-point notation
    integer_part = int(abs_val)
    integer_part_str = str(integer_part)
    integer_part_length = len(integer_part_str)

    if integer_part_length <= number_width:
        # Calculate available space for decimal places
        decimal_places = number_width - integer_part_length - (1 if integer_part_length < number_width else 0)
        format_str = f"{{:0{number_width}.{decimal_places}f}}"
        formatted_number = format_str.format(abs_val)
        if len(formatted_number) <= number_width:
            formatted = sign + formatted_number
            if len(formatted) == width:
                return formatted
    # Scientific notation
    if abs_val == 0:
        exponent = 0
    else:
        exponent = int(math.floor(math.log10(abs_val)))
    mantissa = abs_val / (10 ** exponent)

    # Available space for mantissa and exponent
    exponent_str = f"e{exponent}" if exponent >= 0 else f"e-{abs(exponent)}"
    exponent_length = len(exponent_str)
    mantissa_width = number_width - exponent_length
    if mantissa_width < 1:
        raise ValueError(f"Cannot format value {val} in width {width}")

    for decimal_places in range(mantissa_width, -1, -1):
        mantissa_rounded = round(mantissa, decimal_places)
        # Adjust mantissa and exponent if mantissa_rounded >= 10
        if mantissa_rounded >= 10:
            mantissa_rounded /= 10
            exponent += 1
            exponent_str = f"e{exponent}" if exponent >= 0 else f"e-{abs(exponent)}"
            exponent_length = len(exponent_str)
            mantissa_width = number_width - exponent_length
            if mantissa_width < 1:
                continue  # Not enough width, try next decimal_places

        if decimal_places > 0:
            mantissa_format = f"{{:.{decimal_places}f}}"
            mantissa_str = mantissa_format.format(mantissa_rounded)
        else:
            mantissa_rounded_int = int(round(mantissa_rounded))
            mantissa_str = f"{mantissa_rounded_int}"
            if mantissa_width >= len(mantissa_str) + 1:
                mantissa_str += '.'
        mantissa_str_length = len(mantissa_str)
        if mantissa_str_length <= mantissa_width:
            formatted = sign + mantissa_str + exponent_str
            if len(formatted) == width:
                return formatted
    raise ValueError(f"Cannot format value {val} in width {width}")

import uuid
from IPython.display import display_javascript, display_html, display

class _RenderJSON(object):
    def __init__(self, json_data):
        if isinstance(json_data, dict):
            self.json_str = json.dumps(json_data)
        else:
            self.json_str = json_data
        self.uuid = str(uuid.uuid4())

    def _ipython_display_(self):
        display_html('<div id="{}" style="height: 600px; width:100%;"></div>'.format(self.uuid), raw=True)
        display_javascript("""
        require(["https://rawgit.com/caldwell/renderjson/master/renderjson.js"], function() {
        document.getElementById('%s').appendChild(renderjson(%s))
        });
        """ % (self.uuid, self.json_str), raw=True)
def display_json(obj):
    return _RenderJSON(obj)



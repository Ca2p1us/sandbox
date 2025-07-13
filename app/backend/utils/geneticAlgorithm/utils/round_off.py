from decimal import Decimal, ROUND_HALF_UP


def round_off_to_digit(number: float) -> float:
    """整数に四捨五入する

    Args:
        number (float): [description]

    Returns:
        float: [description]
    """

    ratio = Decimal(str(number)).quantize(Decimal('0'), rounding=ROUND_HALF_UP)
    return float(ratio)

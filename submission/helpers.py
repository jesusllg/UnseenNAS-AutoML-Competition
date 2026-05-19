import math


def div_remainder(n, interval):
    factor = math.floor(n / interval)
    remainder = int(n - (factor * interval))
    return factor, remainder


def show_time(seconds):
    if seconds < 60:
        return "{:.2f}s".format(seconds)
    elif seconds < (60 * 60):
        minutes, seconds = div_remainder(seconds, 60)
        return "{}m,{}s".format(minutes, seconds)
    else:
        hours, seconds = div_remainder(seconds, 60 * 60)
        minutes, seconds = div_remainder(seconds, 60)
        return "{}h,{}m,{}s".format(hours, minutes, seconds)

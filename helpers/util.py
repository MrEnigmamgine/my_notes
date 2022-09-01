"""Contains helper function for general use."""

#####################################################
#               FORMATTING TRICKS                   #
#####################################################

def make_valid_py_id(varStr: str) -> str: 
    """Converts a string into a valid python identifier"""
    import re
    return re.sub('\W|^(?=\d)','_', varStr).lower()

def year_to_decade_name(year: int) -> str:
    """Convert a year int into a categorical str"""
    import math
    decade = math.floor(int(year) / 10) * 10
    return f"{decade}s"

def unstring_usd(string: str) -> float:
    """Returns a float from a string representing a USD amount."""
    return float(string.replace('$','').replace(',',''))

def as_currency(amount: float) -> str:
    """Return a USD formatted string from a float."""
    if amount >= 0:
        return '${:,.2f}'.format(amount)
    else:
        return '-${:,.2f}'.format(-amount)

def format_float(x):
    """Limits a float to 2 digits."""
    return float("{:.2f}".format(x))

def get_n_colors(n: int, palette='bright') -> list:
    """Gets a list consisting of n colors."""
    import seaborn as sns
    colors = []
    for i in range(n):
        colors.append(sns.color_palette(palette=palette)[i])
    return colors

def cat_to_colors(series):
    """Given a series, return a series of equal length consisting of colors."""
    cats = series.value_counts().index.to_list()
    colors = get_n_colors(len(cats))
    x = series.apply(lambda x: colors[cats.index(x)])
    return x
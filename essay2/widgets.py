

import os
import subprocess
import json
import re
import glob
import dateutil.parser



# '__price_range_parsing_functions':

def fix_bad_str(string):
    """ Formats nuisance characters. """
    if string:
        # These are various long dash characters used in the document
        for r in [r'\x97', r'\x96', r'[-]+', r'●']:
            string = re.sub(r, '-', string)
        # Other nuisance chars
        string = re.sub(r'\x95', '->', string)
        string = re.sub(r'\x93', '"', string)
        string = re.sub(r'\x94', '"', string)
        string = re.sub(r'/s/', '', string)
        string = re.sub(r'\x92', "'", string)
        string = re.sub(r'\xa0', ' ', string)
        string = re.sub(r'\u200b', '', string)
        string = re.sub(r'\s+', ' ', string)
    return string.strip()

def fix_dollars(string_list):
    """Split and strip a string, appending dollar signs where necessary"""

    new_strlist = []
    prepend_next = False
    for s in string_list:
        s = re.sub(r'^(U[\.]?S[\.]?)?\$\s*', '$', s)
        if s.strip() == '$':
            prepend_next = True
            continue
        new_str = ' '.join(e.strip() for e in s.split('\n'))
        if prepend_next == True:
            if not new_str.startswith('$'):
                new_str = '$' + new_str
            prepend_next = False

        new_strlist += [new_str]
    return new_strlist


def as_cash(string):
    if '$' not in string:
        return None
    string = string.replace('$','').replace(',','')
    return float(string) if string.strip() else None




# 'excel_cell_movement_functions':

def next_row(char, n=1):
    "Shifts cell reference by n rows."

    is_xls_cell = re.compile(r'^[A-Z].*[0-9]$')
    if not is_xls_cell.search(char):
        raise(Exception("'{}' is not a valid cell".format(char)))

    if n == 0:
        return char

    idx = [i for i,x in enumerate(char) if x.isdigit()][0]
    if int(char[idx:]) + n < 0:
        return char[:idx] + '1'
    else:
        return char[:idx] + str(int(char[idx:]) + n)

def next_col(char, n=1):
    "Shifts cell reference by n columns."

    is_xls_cell = re.compile(r'^[A-Z].*[0-9]$')
    if not is_xls_cell.search(char):
        raise(Exception("'{}' is not a valid cell".format(char)))

    if n == 0:
        return char

    def next_char(char):
        "Next column in excel"
        if all(c=='Z' for c in char):
            return 'A' * (len(char) + 1)
        elif len(char) == 1:
            return chr(ord(char) + 1)
        elif char.endswith('Z'):
            return next_char(char[:-1]) + 'A'
        else:
            return 'A' + next_char(char[1:])

    def prev_char(char):
        "Previous column in excel"
        if len(char) == 1:
            return chr(ord(char) - 1) if char != 'A' else ''
        elif not char.endswith('A'):
            return char[:-1] + prev_char(char[-1])
        elif char.endswith('A'):
            return prev_char(char[:-1]) + 'Z'

    idx = [i for i,x in enumerate(char) if x.isdigit()][0]
    row = char[idx:]
    col = char[:idx]
    for i in range(abs(n)):
        col = next_char(col) if n > 0 else prev_char(col)
    return col + row



#!/usr/bin/env python3
import re
import sys

initial_case_regexp     = re.compile(r'^\s+case\s+\S+?:(\s*(//.*?|/\*.*?\*/\s+))?$')
fallthrough_case_regexp = re.compile(r'^\s+(case|default)\b')

for path in sys.argv[1:]:
    with open(path, 'r') as stream:
        prev_line = stream.readline()
        line_number = 1

        while True:
            next_line           = stream.readline()
            if len(next_line) == 0:
                break

            initial_match       = initial_case_regexp.match(prev_line)
            fallthrough_match   = fallthrough_case_regexp.search(next_line)

            if initial_match and fallthrough_match:
                print("%s:%d:%s" % (path, line_number, initial_match.group(0)))
            
            line_number += 1
            prev_line = next_line

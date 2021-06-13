#!/bin/python3

import sys

with open(sys.argv[1]) as f:
    for line in f:
        line = line[:-1]
        if len(line.strip()) == 0:
            print(line)
            continue
        fields = line.strip().split()
        print('\t'.join(fields[4:]))

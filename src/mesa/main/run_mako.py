#! /usr/bin/env python

from mako.template import Template
from sys import argv

template = Template(filename=argv[1])
print template.render(argv = argv[1:])

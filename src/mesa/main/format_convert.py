#!/usr/bin/env python
#
# Copyright 2014 Intel Corporation
# All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sub license, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice (including the
# next paragraph) shall be included in all copies or substantial portions
# of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
# OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT.
# IN NO EVENT SHALL VMWARE AND/OR ITS SUPPLIERS BE LIABLE FOR
# ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import format_parser as parser

def __get_datatype(_type, size):
   if _type == parser.FLOAT:
      if size == 32:
         return 'float'
      elif size == 16:
         return 'uint16_t'
      else:
         assert False
   elif _type == parser.UNSIGNED:
      if size <= 8:
         return 'uint8_t'
      elif size <= 16:
         return 'uint16_t'
      elif size <= 32:
         return 'uint32_t'
      else:
         assert False
   elif _type == parser.SIGNED:
      if size <= 8:
         return 'int8_t'
      elif size <= 16:
         return 'int16_t'
      elif size <= 32:
         return 'int32_t'
      else:
         assert False
   else:
      assert False

def channel_datatype(c):
   return __get_datatype(c.type, c.size)

def format_datatype(f):
   if f.layout == parser.PACKED:
      if f.block_size() == 8:
         return 'uint8_t'
      if f.block_size() == 16:
         return 'uint16_t'
      if f.block_size() == 32:
         return 'uint32_t'
      else:
         assert False
   else:
      return __get_datatype(f.channel_type(), f.channel_size())


# ===========================================================================
#
# SYNOPSIS
#
#   AX_CHECK_PYTHON_MAKO_MODULE(MIN_VERSION_NUMBER)
#
# DESCRIPTION
#
#   Check whether Python mako module is installed and its version higher than
#   minimum requested.
#
#   Example of its use:
#
#   For example, the minimum mako version would be 0.7.3. Then configure.ac
#   would contain:
#
#   AC_CHECK_PYTHON_MAKO_MODULE(0.7.3)
#
# LICENSE
#
#   Copyright (c) 2014 Intel Corporation.
#
#   This program is free software; you can redistribute it and/or modify it
#   under the terms of the GNU General Public License as published by the
#   Free Software Foundation; either version 2 of the License, or (at your
#   option) any later version.
#
#   This program is distributed in the hope that it will be useful, but
#   WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
#   Public License for more details.
#
#   You should have received a copy of the GNU General Public License along
#   with this program. If not, see <http://www.gnu.org/licenses/>.
#
#   As a special exception, the respective Autoconf Macro's copyright owner
#   gives unlimited permission to copy, distribute and modify the configure
#   scripts that are the output of Autoconf when processing the Macro. You
#   need not follow the terms of the GNU General Public License when using
#   or distributing such scripts, even though portions of the text of the
#   Macro appear in them. The GNU General Public License (GPL) does govern
#   all other use of the material that constitutes the Autoconf Macro.
#
#   This special exception to the GPL applies to versions of the Autoconf
#   Macro released by the Autoconf Archive. When you make and distribute a
#   modified version of the Autoconf Macro, you may extend this special
#   exception to the GPL to apply to your modified version as well.

dnl macro that checks for mako module in python
AC_DEFUN([AC_CHECK_PYTHON_MAKO_MODULE],
[AC_MSG_CHECKING(for module mako in python)
    python $srcdir/src/mesa/main/python_mako.py $1
    if test $? -ne 0 ; then
    AC_MSG_ERROR(mako $1 or later is required.)
    fi
])

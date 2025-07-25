# check_enable_instrument_functions.m4
#
# SYNOPSIS
#
#   CHECK_ENABLE_INSTRUMENT_FUNCTIONS()
#
# DESCRIPTION
#
#   Enable function instrumentation using GCC's -finstrument-functions.
#   This feature is only available in debug builds.
#

AC_DEFUN([CHECK_ENABLE_INSTRUMENT_FUNCTIONS],[
    AC_ARG_ENABLE([instrument-functions],
        [AS_HELP_STRING([--enable-instrument-functions],
            [Enable function call instrumentation (debug mode only)])])

    instrument_functions_enabled=no
    AS_IF([test "${enable_instrument_functions}" = "yes"], [
        AS_IF([test "${ax_enable_debug}" != "yes"], [
            AC_MSG_ERROR([Function instrumentation can only be enabled in debug mode. Use --enable-debug.])
        ])
        
        # Check if compiler supports -finstrument-functions
        AC_LANG_PUSH([C++])
        saved_CXXFLAGS="${CXXFLAGS}"
        CXXFLAGS="${CXXFLAGS} -finstrument-functions"
        AC_MSG_CHECKING([whether compiler supports -finstrument-functions])
        AC_COMPILE_IFELSE([AC_LANG_PROGRAM([],[])],
            [AC_MSG_RESULT([yes])
             instrument_functions_enabled=yes],
            [AC_MSG_RESULT([no])
             AC_MSG_ERROR([Compiler does not support -finstrument-functions])])
        CXXFLAGS="${saved_CXXFLAGS}"
        AC_LANG_POP([C++])
    ])

    AS_IF([test "${instrument_functions_enabled}" = "yes"], [
        AC_DEFINE([ENABLE_INSTRUMENT_FUNCTIONS], [1], [Define to 1 if function instrumentation is enabled])
        CXXFLAGS="${CXXFLAGS} -finstrument-functions"
        CXXFLAGS="${CXXFLAGS} -finstrument-functions-exclude-file-list=/usr/include/,/usr/lib/"
        AC_MSG_NOTICE([Function instrumentation enabled (excluding standard library)])
    ])

    AM_CONDITIONAL([ENABLE_INSTRUMENT_FUNCTIONS], [test "${instrument_functions_enabled}" = "yes"])
])

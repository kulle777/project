TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        canny.c \
        opencl_util.c \
        util.c \
        vectoradd.c

DISTFILES += \
    Makefile \
    nonmax.cl \
    phase.cl \
    sobel.cl \
    vecadd.cl

HEADERS += \
    #canny.c# \
    opencl_util.h \
    util.h

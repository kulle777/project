TEMPLATE = app
CONFIG += console
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += \
        canny.c \
        opencl_util.c \
        util.c

DISTFILES += \
    nonmax.cl \
    phase.cl \
    sobel.cl

HEADERS += \
    #canny.c# \
    opencl_util.h \
    util.h

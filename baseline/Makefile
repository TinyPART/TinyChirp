# RIOTBASE= ../../../TinyPART/RIOT
RIOTBASE= $(HOME)/RIOT

BOARD ?= native
APPLICATION = TinyBirdSound_baseline

USEMODULE += printf_float

# Modules to include:
# USEMODULE += shell
# USEMODULE += shell_cmds_default
# USEMODULE += ps

# Use the default file system
USEMODULE += vfs_default

CFLAGS += -DFATFS_FFCONF_OPT_USE_LFN=1
CFLAGS += -DFATFS_FFCONF_OPT_USE_STRFUNC=1

USEMODULE += xtimer random


# For LittleFS on real devices, the main stack size has to be
# increased:
CFLAGS += -DTHREAD_STACKSIZE_MAIN=4196

INCLUDES += -I$(CURDIR)/include

include $(RIOTBASE)/Makefile.include

# name of your application
APPLICATION = raw_audio_transformer

# If no BOARD is found in the environment, use this default:
BOARD ?= native

# This has to be the absolute path to the RIOT base directory:
RIOTBASE ?= $(CURDIR)/../..
# Comment this out to disable code in RIOT that does safety checking
# which is not needed in a production environment but helps in the
# development process:
DEVELHELP ?= 1
#CFLAGS_OPT = -O0
CFLAGS += -DTHREAD_STACKSIZE_DEFAULT=2048
USEMODULE += printf_float xtimer
# Change this to 0 show compiler invocation lines by default:
QUIET ?= 1

include $(RIOTBASE)/Makefile.include

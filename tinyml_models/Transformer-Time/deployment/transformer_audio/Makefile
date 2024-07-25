RIOTBASE= $(HOME)/RIOT

BOARD ?= native
APPLICATION = transformer_audio

WERROR ?= 0
# DEVELHELP ?= 1


USEMODULE += xtimer random 


USEMODULE += mlmci
USEMODULE += transformer_exec
EXTERNAL_MODULE_DIRS += $(CURDIR)

include $(RIOTBASE)/Makefile.include

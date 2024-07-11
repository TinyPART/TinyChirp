RIOTBASE= $(HOME)/RIOT

BOARD ?= nrf52840dk
APPLICATION = TinyChirp_mel_spectrogram

#USEMODULE += printf_float


USEMODULE +=  random cmsis cmsis-dsp xtimer
#
USEPKG += cmsis 


CFLAGS += -DTHREAD_STACKSIZE_MAIN=16000

#DEVELHELP ?= 1
#CFLAGS_OPT = -O0

INCLUDES += -I$(CURDIR)/include

include $(RIOTBASE)/Makefile.include

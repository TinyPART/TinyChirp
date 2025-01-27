APPLICATION = TinyChirp_DEMO
WERROR = 0
BOARD ?= microbit-v2

RIOTBASE= $(HOME)/RIOT

DEVELHELP ?= 0

FEATURES_REQUIRED = periph_adc periph_gpio
USEMODULE += ztimer
USEMODULE += ztimer_msec ztimer_usec
USEMODULE += printf_float

CFLAGS += -DADC_GAIN=SAADC_CH_CONFIG_GAIN_Gain4

USEMODULE += printf_float xtimer
QUIET ?= 1

include $(RIOTBASE)/Makefile.include

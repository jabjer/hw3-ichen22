ENV_NAME ?= ligo
ENV_FILE ?= environment.yml

.PHONY: env html clean

env:
	@if conda env list | grep -qE '^[[:space:]]*$(ENV_NAME)[[:space:]]'; then \
		conda env update -n $(ENV_NAME) -f $(ENV_FILE) --prune; \
	else \
		conda env create -n $(ENV_NAME) -f $(ENV_FILE); \
	fi
	@conda run -n $(ENV_NAME) python -m pip install -U mystmd

html:
	@conda run -n $(ENV_NAME) myst build --html

clean:
	rm -rf _build figures audio
	mkdir -p figures audio
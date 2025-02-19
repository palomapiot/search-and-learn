.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src

style:
	ruff check --select I --fix $(check_dirs)
	ruff format $(check_dirs)

quality:
	ruff check --select I $(check_dirs)
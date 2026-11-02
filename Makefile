.PHONY: test run report verify
test:
	python -m pytest -q
run:
	python -m causal_targeting run --rebuild
report:
	python -m causal_targeting report
verify:
	python -m causal_targeting verify

run_tests:
	python -m unittest discover .
	coverage run --source=src -m unittest discover -s .
	coverage report -m

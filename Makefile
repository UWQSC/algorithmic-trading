run_tests:
	python3 -m unittest discover .
	coverage run --source=src -m unittest discover -s .
	coverage report -m

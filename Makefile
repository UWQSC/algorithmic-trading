run_tests:
	python3 -m unittest discover tests -p '*_test.py'
	coverage run --source=src -m unittest discover tests -p '*_test.py'
	coverage report -m

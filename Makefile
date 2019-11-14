# Make file to run scripts
# Note: use ctrl-v tab to put actual tabs

run_all :
	python3 heart_anomaly.py SPECT
	python3 heart_anomaly.py spect-resplit-itg
	python3 heart_anomaly.py spect-resplit
	python3 heart_anomaly.py spect-itg
	python3 heart_anomaly.py spect-orig

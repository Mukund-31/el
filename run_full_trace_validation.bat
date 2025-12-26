@echo off
echo Running Full Real-World Trace Validation (300 Episodes)...
echo This may take 5-10 minutes.
cd 7thsem
python validation_framework.py --episodes 300 --trace_file ../queue_data.csv --output results_trace_300
echoines Validation Complete! Check results_trace_300 folder.
pause

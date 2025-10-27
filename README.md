# IML_CW1

Requirements
- Python 3.8+
- numpy
- matplotlib

Install dependencies:
```
pip install numpy matplotlib
```

Run
- Open a terminal in the repository folder (Coursework1).
- Ensure `clean_dataset.txt` and `noisy_dataset.txt` are in the same folder as the script.
- Run the script (adjust filename if yours has a .py extension):
```
python main.py
```
or, if the file is named without extension and executable on your system:
```
python .\main
```

What it does
- Loads `clean_dataset.txt` and `noisy_dataset.txt`.
- Trains and displays a decision tree built on the entire clean dataset (plot).
- Performs 10-fold cross-validation separately on the clean and noisy datasets and prints accuracy, confusion matrix, per-class precision/recall/F1 and macro-averaged metrics.

Notes
- Use the same random seed defined in the script for reproducible splits.
- If you rename the script or move the data files, update the paths accordingly.
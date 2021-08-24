# gptsne
Rough usage (from the src/ directory):   
`python3 -m gpmallt.gpmal_lt --help`

e.g. `python3 -m gpmallt.gpmal_lt --erc -nn 30 -lf deviation_weighted -threads 4 --trees 2 -d dermatology --dir "datasets/"`

* Datasets used in the paper are in datasets/
* You can add your own datasets in csv format, with a header line
* Most GP parameters are configured in gpmallt/rundata_lo.py

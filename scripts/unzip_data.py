#!/usr/bin/env python3

import zipfile
import pandas as pd
import numpy as np

filename1 = "ENGLISH/WTL1050.pdf"
zip_path1 = "data/WTODOCS_2019APR08-155838.ZIP"

with zipfile.ZipFile(zip_path1, "r") as zip_ref:
	#zip_ref.printdir()
	zip_ref.extract(filename1, "data/")

filename2 = "Tableau_stats.xlsx"
zip_path2 = "data/[Fwd__RE__Collaboration_with_Leeds].zip"

with zipfile.ZipFile(zip_path2, "r") as zip_ref:
	#zip_ref.printdir()
	zip_ref.extract(filename2, "data/")


#!/usr/bin/env python3

import pandas as pd 
from datetime import datetime

# Load translation time data
tr_times = pd.read_excel("data/Tableau_stats.xlsx") 

# Split columns for french and spanish translations + change column names
french = tr_times.iloc[2:,3:8].drop("TRANSLATION REVISION", axis=1)
french.columns=["TRANSLATION START", "TRANSLATION END", "POOL START", "POOL END"]
spanish = tr_times.iloc[2:,8:-1].drop("TRANSLATION REVISION.1", axis=1)
spanish.columns=["TRANSLATION START", "TRANSLATION END", "POOL START", "POOL END"]

# Change string to datetime + correct typos
french["TRANSLATION END"].iloc[5] = datetime.strptime(french["TRANSLATION END"].iloc[5], '%d.%m.%Y')
french["TRANSLATION START"].iloc[5] = datetime.strptime(french["TRANSLATION START"].iloc[5], '%d.%m.%Y')
french["POOL END"].iloc[5] = datetime.strptime(french["POOL END"].iloc[5], '%d.%m.%Y')
french["POOL START"].iloc[5] = datetime.strptime(french["POOL START"].iloc[5], '%d.%m.%Y')

spanish["TRANSLATION END"].iloc[5] = datetime.strptime(spanish["TRANSLATION END"].iloc[5], '%d.%m.%Y')
spanish["TRANSLATION START"].iloc[5] = datetime.strptime(spanish["TRANSLATION START"].iloc[5], '%d.%m.%Y')
spanish["POOL END"].iloc[5] = datetime.strptime(spanish["POOL END"].iloc[5], '%d.%m.%Y')
spanish["POOL START"].iloc[5] = datetime.strptime(spanish["POOL START"].iloc[5], '%d.%m.%Y')

spanish["TRANSLATION START"].iloc[95] = datetime(2018, 5, 23)

# Time taken for translation (excluding weekends)
french["TIME TAKEN"] = french["TRANSLATION END"] - french["TRANSLATION START"]
spanish["TIME TAKEN"] = spanish["TRANSLATION END"] - spanish["TRANSLATION START"]

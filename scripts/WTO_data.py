#!/usr/bin/env python3

import numpy as np
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt 

# Load translation time data
orig_df = pd.read_excel("data/Tableau_stats.xlsx") 

# Split columns for french and spanish translations + change column names
french = orig_df.iloc[2:,3:8].drop("TRANSLATION REVISION", axis=1)
french.columns=["TRANSLATION START", "TRANSLATION END", "POOL START", "POOL END"]
spanish = orig_df.iloc[2:,8:-1].drop("TRANSLATION REVISION.1", axis=1)
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
def working_days(start, end):
    """Output the number of working days between two given dates.
    Note: this assumes dates provided work on a Monday to Sunday basis.
    It does not take into account any holidays. """

    daydiff = end.weekday() - start.weekday() # difference in weekday, eg. Wednesday - Tuesday = -1
    days = ((end-start).days - daydiff) / 7 * 5 + min(daydiff, 5) - (max(end.weekday() - 4, 0) % 5 ) # convert week to workdays

    return days

day_list = []
for i in french.index:
    days = working_days(french["TRANSLATION START"].loc[i], french["TRANSLATION END"].loc[i]) + 1 # +1 so that finishing on the same day it started counts as one day
    day_list.append(days)

french["TRANSLATION DAYS"] = day_list

day_list = []
for i in spanish.index:
    days = working_days(spanish["TRANSLATION START"].loc[i], spanish["TRANSLATION END"].loc[i]) + 1 # +1 so that finishing on the same day it started counts as one day
    day_list.append(days)

spanish["TRANSLATION DAYS"] = day_list

# join time talen for each language into one df
time_df = pd.concat([french["TRANSLATION DAYS"], spanish["TRANSLATION DAYS"]], axis=1, sort=False)
time_df.columns = ["DAYS FRENCH", "DAYS SPANISH"]

# join with number of words in original document
time_df = pd.concat([orig_df["WORDS"].iloc[2:], time_df], axis=1, sort=False)

## plot time taken against number of words
# plt.plot(time_df["WORDS"], time_df["DAYS FRENCH"], '.')
# plt.plot(time_df["WORDS"], time_df["DAYS SPANISH"], '.')
# plt.xlabel("Number of words")
# plt.ylabel("Days taken to translate")
# plt.legend()
# plt.show()

#plt.plot(time_df["FRENCH"], time_df["SPANISH"], '.')

# Number of words translated per day (average)
w_perday = pd.DataFrame()
w_perday["PERDAY FRENCH"] = time_df["WORDS"].div(time_df["DAYS FRENCH"], axis=0)
w_perday["PERDAY SPANISH"] = time_df["WORDS"].div(time_df["DAYS SPANISH"], axis=0)


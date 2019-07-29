#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd 
from datetime import datetime
import matplotlib.pyplot as plt 

def working_days(start, end):
    """Output the number of working days between two given dates.
    Note: this assumes dates provided work on a Monday to Sunday basis.
    It does not take into account any holidays. """

    daydiff = end.weekday() - start.weekday() # difference in weekday, eg. Wednesday - Tuesday = -1
    days = ((end-start).days - daydiff) / 7 * 5 + min(daydiff, 5) - (max(end.weekday() - 4, 0) % 5 ) # convert week to workdays

    return days

def remove_outliers(time_df, q):
    """ 
    New: Remove values outside q to (100-q) quantiles.

    Old: Only keep cases where the time taken is less than n times as long for one language compared to the other.
    Also drop cases where the difference in days is larger than 8*n. (20 days for n=2.5)
    """
    # old version
    # # remove outliers based on ratio and minimum difference
    # time_df = time_df.loc[(time_df["DAYS FRENCH"] / time_df["DAYS SPANISH"]) < n]
    # time_df = time_df.loc[(time_df["DAYS SPANISH"] / time_df["DAYS FRENCH"]) < n] 
    # time_df = time_df.loc[abs(time_df["DAYS FRENCH"] - time_df["DAYS SPANISH"]) < 8*n]

    # remove outliers based on quantiles
    perday_ratio = time_df['DAYS FRENCH'] / time_df['DAYS SPANISH']
    uq = perday_ratio.quantile(q=(100-q)/100)
    lq = perday_ratio.quantile(q=q/100)
    perday_ratio = perday_ratio.loc[perday_ratio < uq]
    perday_ratio = perday_ratio.loc[perday_ratio > lq]
    time_df = time_df.loc[perday_ratio.index]

    return time_df

def main():

    # Load translation time data
    orig_df = pd.read_excel("data/wto/Tableau_stats.xlsx") 

    # Drop duplicate entries
    orig_df.drop_duplicates(subset=['JOB N°', 'SYMBOL', 'WORDS'],inplace=True)

    # Split columns for french and spanish translations + change column names
    french = orig_df.iloc[1:,3:8].drop("TRANSLATION REVISION", axis=1)
    french.columns=["TRANSLATION START", "TRANSLATION END", "POOL START", "POOL END"]
    spanish = orig_df.iloc[1:,8:-1].drop("TRANSLATION REVISION.1", axis=1)
    spanish.columns=["TRANSLATION START", "TRANSLATION END", "POOL START", "POOL END"]

    # Change string to datetime - UPDATE: no need after duplicates dropped
    # french["TRANSLATION END"].iloc[5] = datetime.strptime(french["TRANSLATION END"].iloc[5], '%d.%m.%Y')
    # french["TRANSLATION START"].iloc[5] = datetime.strptime(french["TRANSLATION START"].iloc[5], '%d.%m.%Y')
    # french["POOL END"].iloc[5] = datetime.strptime(french["POOL END"].iloc[5], '%d.%m.%Y')
    # french["POOL START"].iloc[5] = datetime.strptime(french["POOL START"].iloc[5], '%d.%m.%Y')

    # spanish["TRANSLATION END"].iloc[5] = datetime.strptime(spanish["TRANSLATION END"].iloc[5], '%d.%m.%Y')
    # spanish["TRANSLATION START"].iloc[5] = datetime.strptime(spanish["TRANSLATION START"].iloc[5], '%d.%m.%Y')
    # spanish["POOL END"].iloc[5] = datetime.strptime(spanish["POOL END"].iloc[5], '%d.%m.%Y')
    # spanish["POOL START"].iloc[5] = datetime.strptime(spanish["POOL START"].iloc[5], '%d.%m.%Y')

    # Correct typos
    spanish["TRANSLATION START"].loc[97] = datetime(2018, 5, 23)
    french["TRANSLATION END"].loc[14] = datetime(2017, 2, 8)

    # Time taken for translation (excluding weekends)
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

    # join time taken for each language into one df
    time_df = pd.concat([french["TRANSLATION DAYS"], spanish["TRANSLATION DAYS"]], axis=1, sort=False)
    time_df.columns = ["DAYS FRENCH", "DAYS SPANISH"]

    # join with number of words in original document
    time_df = pd.concat([orig_df["WORDS"].iloc[2:], time_df], axis=1, sort=False)

    # get rid of outliers (difference between french and spanish unusually large)
    time_df = remove_outliers(time_df, args.q)

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

    time_df = pd.concat([time_df, w_perday], axis=1, sort=False)

    job_df = orig_df[["JOB N°", "SYMBOL"]].iloc[2:]
    output_df = pd.concat([job_df, time_df], axis=1, sort=False).dropna(how='any')

    print(output_df.head(10).round(2))

    if args.s:
        output_path = args.op
        if output_path.endswith((".csv", ".txt", ".data")): 
            output_df.to_csv(output_path, index=False)
            print("\nOutput saved to %s" % output_path)
        else:
            print("\nOutput path must include filename. \nNote: output will be saved as comma separated values.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", action="store_true",
        help="Save output to csv.")
    parser.add_argument("-q", type=float, default=10,
        help="Quantiles to remove (<q and >100-q).")
    parser.add_argument("-op", type=str, default="data/wto_timed.csv",
        help="Path for output file, including filename.")
    args = parser.parse_args()    
    main()

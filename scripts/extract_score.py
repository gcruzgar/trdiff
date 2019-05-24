#!/usr/bin/env python3
import pandas as pd  
import argparse

def main():

    filename = args.filename
    #filename = "data/en-fr-100.pra"
        
    # read ter output
    with open(filename, "r") as f:
        ht=f.readlines() 

    df = pd.DataFrame(data=ht, columns=['line']) # save as dataframe for ease
    df = df.loc[df['line'].str.contains("Score")] # only need lines containing scores
    # only need actual score:
    df['line'] = df['line'].str.replace("Score: ", "")
    df['line'] = df['line'].str.replace("\((.*?)\)\\n", "")

    df.rename(index=str,columns={'line': 'score'}, inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['score'] = df['score'].astype('float')

    print("Number of scores processed: %d" % df.shape[0])

    # Save outputs to new file
    out_path = args.o
    with open(out_path, 'w') as f:
        for item in df['score']:
            f.write("%s\n" % item)
    print("output saved to %s" % out_path)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('filename', type=str, nargs='?',
        help="file to extract (including path from current directory)")
    parser.add_argument('-o', type=str, nargs='?', default="mt_score.txt",
        help="specify output filename")
    args = parser.parse_args()

    main()

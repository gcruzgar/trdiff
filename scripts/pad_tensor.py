import torch
import argparse

def main():

    #t_path="xlm-embeddings-810000-820000"
    t1 = args.t
    t2 = t1+10000
    t_path="xlm-embeddings-"+str(t1)+str(t2)
    
    #ind = 4500
    ind = args.i

    t = torch.load(t_path+".pt")

    first_half = t[0:ind]
    second_half = t[ind:]
    pad = torch.zeros(500, 1024)

    t_new = torch.cat((first_half, pad, second_half))

    torch.save(t_new, t_path+"-padded.pt")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-t", type=int, nargs='?', help='start index of tensor to read')
    parser.add_argument("-i", type=int, nargs='?', help='start index of missing values')
    args = parser.parse_args()

    main()    
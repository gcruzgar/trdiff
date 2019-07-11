import os
import re
import torch
import argparse

def main():

    xlm_path=args.p
    if xlm_path.endswith("/") == False:
            xlm_path = xlm_path+"/"
    print("Loading tensors from %s..." % xlm_path)

    file_list = os.listdir(xlm_path)
    print("%d tensors found\n" % len(file_list))

    ids=[]  
    t_all=torch.Tensor()
    for f in file_list:
        ids.extend(re.findall("\d+", f))
        t = torch.load(xlm_path+f)
        t_all=torch.cat((t_all, t), dim=0)

        torch.save(t_all, "xlm-embeddings-"+min(ids)+"-"+max(ids)+".pt")

if __name__ == "__main__":

    parser= argparse.ArgumentParser()
    parser.add_argument("-p", type=str, nargs='?', default="xlm-embeddings/",
            help="path to xlm embeddings")
    args = parser.parse_args()
        
    main()
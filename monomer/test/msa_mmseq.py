
#The code below is a reworked version
#of the original code by ColabFold developers:
'''
MIT License

Copyright (c) 2021 Sergey Ovchinnikov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import sys
import json
import time
import requests
import tarfile
import argparse
from glob import glob
from collections import OrderedDict
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
from Bio import SeqIO

#from alphafold.data import pipeline

def get_MSAs_mmseq(seq_dict, output_path="./"):
    '''
	Query the MMseqs server to obtain MSAs for all sequences in the sequence
	dictionary. Each msa is downloaded to a subfolder in the path directory,
    named as the corresponding directory key.
    Returns an dictionary with the same keys as seq_dict,
    containing the paths to MSAs in a3m format.
    '''
    dones = {key: os.path.isfile(os.path.join(output_path, f'{key}', 'mmseqs', 'aggregated.a3m'))
												for key in seq_dict}

    # call mmseqs2 api
    ids = OrderedDict()
    for key, seq in seq_dict.items():
        if dones[key]:
            ids[key] = None
        else:
            fasta_str = SeqRecord(Seq(seq)).format('fasta')
            data = {'q': fasta_str, 'mode': 'all'}  # different modes?

            submitted = False
            while not submitted:
                resp = requests.post('https://a3m.mmseqs.com/ticket/msa', data)
                try:
                    ids[key] = resp.json()['id']
                    submitted = True
                except:
                    time.sleep(20)

    while False in dones.values():
        for key, id_ in ids.items():
            if not dones[key]:
                resp = requests.get(f'https://a3m.mmseqs.com/ticket/{id_}')
                status = resp.json()['status']
                if status == 'ERROR':
                    # TODO: raise exception & logging
                    print(f'Error: MMseqs2 id {id_} seq {key}')
                    exit(1)
                elif status == 'COMPLETE':
                    # Download results
                    resp = requests.get(f'https://a3m.mmseqs.com/result/download/{id_}', stream=True)
                    dir_ = f'{key}'
                    os.makedirs(os.path.join(output_path, dir_, 'mmseqs'), exist_ok=True)
                    tar_gz_file = os.path.join(output_path, dir_, 'mmseqs', 'mmseqs2.tar.gz')
                    with open(tar_gz_file, 'wb') as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Unpack
                    with tarfile.open(tar_gz_file) as f:
                        f.extractall(os.path.join(output_path, f'{key}', 'mmseqs'))

                    agg_a3m_file = os.path.join(output_path, f'{key}', 'mmseqs', 'aggregated.a3m')
                    with open(agg_a3m_file, 'w') as f:
                        a3m_file = os.path.join(output_path, f'{key}', 'mmseqs', 'uniref.a3m')  # bfd.mgnify30.metaeuk30.smag30.a3m
                        with open(a3m_file) as g:
                            f.write(g.read().replace('\x00', ''))
                    dones[key] = True
                time.sleep(5)

    #agg_a3m_dict = {}
    #agg_msa_dict = {}
    #for key in seq_dict.keys():
    #    agg_a3m_file = os.path.join(output_path, f'{key}', 'mmseqs', 'aggregated.a3m')
    #    a3m_lines = "".join(open(agg_a3m_file,"r").readlines())
    #    parsed_MSA = pipeline.parsers.parse_a3m(a3m_lines)
        #print(key)
        #print(parsed_MSA)
        #print()
        #exit()
    #    agg_msa_dict[key] = {'unpairable': [parsed_MSA]}

    #return agg_msa_dict

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Query MMSeqs server to obtain MSAs in a3m format.')
    parser.add_argument('-fas', '--fasta_dir',
                        help='Directory containing the fasta files to be queried. \
                              Only the first record from each fasta file will be processed',
                        default='./')
    parser.add_argument('-out', '--output_dir',
                        help='Directory where the results will be saved',
                        default='./')
    args = parser.parse_args()

    seq_dict = {}
    fasta_files = glob(args.fasta_dir.strip('/') + '/*.fa')
    fasta_files += glob(args.fasta_dir.strip('/') + '/*.fas')
    fasta_files += glob(args.fasta_dir.strip('/') + '/*.fasta')

    for filename in fasta_files:
        print(filename)
        label = os.path.splitext(os.path.basename(filename))[0]
        with open(filename, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                seq_dict[label] = record.seq
                break
    a3m_dict = get_MSAs_mmseq(seq_dict, output_path=args.output_dir)
    #print(a3m_dict)

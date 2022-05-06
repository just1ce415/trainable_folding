#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from glob import glob
from path import Path
import json
import seaborn as sns
import prody
from multiprocessing import Pool
import tqdm
import itertools
import json
from rdkit import Chem
from rdkit.Chem import AllChem
from io import StringIO
from collections import OrderedDict, Counter, defaultdict
import traceback
import urllib
from multiprocessing import Pool
import seaborn as sns
from copy import deepcopy
import pybel
import contextlib
import subprocess

from gemmi import cif

from pdbtools import utils


# In[ ]:





# In[2]:


dataset_dir = Path('data')
msa_dir = dataset_dir / '15k/folding/MMSEQ_submission_second_try'
cif_dir = dataset_dir / '15k/folding/cifs'


# In[ ]:





# In[3]:


def format_request(entities):
    request = '''
    {
      polymer_entities(entity_ids: [?]){
        rcsb_cluster_membership {
          cluster_id,
          identity
        }
        rcsb_polymer_entity_container_identifiers {
            rcsb_id
        }
        entry {
          exptl{
            method
          }
          rcsb_accession_info {
            initial_release_date,
            deposit_date
          }
          rcsb_entry_info {
            resolution_combined
          }
        }
      }
    }
    '''
    
    string = '"' + '", "'.join(entities) + '"'
    request = string.join(request.split('?'))
    url = 'https://data.rcsb.org/graphql?query=' + urllib.parse.quote_plus(request)
    return url


def get_rscb_json(url, num_tries=5, timeout=5):
    for i in range(num_tries):
        try:
            with urllib.request.urlopen(url) as f:
                result = json.loads(f.read().decode())
        except urllib.error.HTTPError:
            print('HTTPError. Trying again' if i != num_tries - 1 else '. Reached max number of trials')
            result = {}
            sleep(timeout)
        if result:
            break
    return result


def load_entities_list_in_chunks(entities_list, chunk_size=1000, timeout=5):
    out = []
    for idx in tqdm.tqdm(range(0, len(entities_list), chunk_size)):
        chunk = entities_list[idx:idx + chunk_size]
        url = format_request(chunk)
        chunk_result = get_rscb_json(url)
        out += chunk_result['data']['polymer_entities']
    return out


#url = format_request(["4HHB_1", "1H8E_4"])
#get_rscb_json(url)
'3EHD_1'
load_entities_list_in_chunks(["4HHB_1", "1H8E_4"])


# In[4]:


#load_entities_list_in_chunks(["3EHD_1", '4HHB_1'])


# In[5]:


case_msa_dirs = sorted(msa_dir.glob('*'))


# In[6]:


entities_list = [x.basename().split('.')[0].upper() for x in case_msa_dirs]
entities_rcsb_info = load_entities_list_in_chunks(entities_list)
entities_rcsb_info = {x['rcsb_polymer_entity_container_identifiers']['rcsb_id']: x for x in entities_rcsb_info}


# In[7]:


list(entities_rcsb_info.values())[0]


# In[9]:


cases = []

letters = list('ACDEFGHIKLMNPQRSTVWYX')
assert len(letters) == 21, len(letters)


def loop_to_list(block, category):
    cat = block.find_mmcif_category(category)
    assert len(cat) > 0, (category, 'does not exist')
    out = []
    for row in cat:
        row_dict = OrderedDict()
        for key in cat.tags:
            row_dict[key] = row[key[cat.prefix_length:]]
        out.append(row_dict)
    return out


for case_msa_dir in tqdm.tqdm(case_msa_dirs):
    #case_msa_dir = case_msa_dirs[0]
    pdb_id = case_msa_dir.basename()[:4]
    case_entity_id = case_msa_dir.basename().split('.')[0][5:]
    
    cif_file = cif_dir / pdb_id + '.cif'
    try:
        ag = prody.parseMMCIF(cif_file)
        assert ag is not None
    except KeyboardInterrupt:
        raise
    except:
        print('Could not parse', cif_file)
        traceback.print_exc()
        continue
    
    cif_parsed = cif.read_file(cif_file).sole_block()
    entity_id_to_asym_id = defaultdict(list)
    table = cif_parsed.find_mmcif_category('_struct_asym')
    for asym_id, entity_id in zip(table.find_column('id'), table.find_column('entity_id')):
        entity_id_to_asym_id[entity_id].append(asym_id)
        
    # parse entity poly seq, to compare with the pdbx_seq_one_letter_code_can
    # this is necessary because for example in 1t6j_1 
    # len(_entity_poly.pdbx_seq_one_letter_code) == len(_entity_poly_seq) == 174 and
    # len(_entity_poly.pdbx_seq_one_letter_code_can) == 176 due to some mistake,
    # so we skip such cases
    entity_poly_seq = loop_to_list(cif_parsed, '_entity_poly_seq')
    entity_poly_seq = [x for x in entity_poly_seq if x['_entity_poly_seq.entity_id'] == case_entity_id]

    # alt locations have the same residue number,
    # keep only the first one to match the pdbx_seq_one_letter_code_can string
    seen_ids = []
    buf = []
    for item in entity_poly_seq:
        if item['_entity_poly_seq.num'] not in seen_ids:
            seen_ids.append(item['_entity_poly_seq.num'])
            buf.append(item)
    entity_poly_seq = buf

    entity_info = None
    table = cif_parsed.find_mmcif_category('_entity_poly')
    for entity_id, pdbx_strand_id, pdbx_seq_one_letter_code_can in zip(
        table.find_column('entity_id'), 
        table.find_column('pdbx_strand_id'), 
        table.find_column('pdbx_seq_one_letter_code_can')
    ):
        #print(pdbx_seq_one_letter_code_can)
        seq_conv = pdbx_seq_one_letter_code_can.replace('\n', '').replace(';', '').replace('U', 'X').replace('O', 'X')
        if case_entity_id == entity_id:
            entity_info = OrderedDict(
                entity_id=entity_id,
                pdbx_strand_ids=pdbx_strand_id.split(','),
                asym_ids=entity_id_to_asym_id[entity_id],
                pdbx_seq_one_letter_code_can=seq_conv
            )
            assert all([aa in letters for aa in seq_conv]), (case_msa_dir, seq_conv, set(seq_conv) - set(letters))
    assert entity_info is not None, (cif_file, case_entity_id)
    
    if len(entity_info['pdbx_seq_one_letter_code_can']) != len(entity_poly_seq):
        print('Mismatching sequences in', cif_file, case_entity_id)
        continue
        
    # TODO: res < 9A, add seq clusters 40%
    rcsb_key = case_msa_dir.basename().split('.')[0].upper()
    if rcsb_key not in entities_rcsb_info:
        print('No info for ', rcsb_key)
        continue
    
    case_dict = OrderedDict(
        pdb_id=pdb_id,
        entity_id=case_entity_id,
        entity_info=entity_info,
        cif_file=cif_file.relpath(dataset_dir),
        a3m_files=[
            case_msa_dir.relpath(dataset_dir) / 'uniref.a3m',
            case_msa_dir.relpath(dataset_dir) / 'bfd.mgnify30.metaeuk30.smag30.a3m'
        ],
        rcsb_info=entities_rcsb_info[rcsb_key]
    )
    cases.append(case_dict)


# In[10]:


utils.write_json(cases, 'data/15k/folding/debug_15k.json')


# In[46]:


len(cases)


# In[33]:


cases[100]


# In[ ]:





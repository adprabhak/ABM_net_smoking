
import errno
import os
from datetime import date, datetime
import sys
import pandas as pd
from ABM import *
import pickle

def filecreation(path,fname):
    mydir = os.path.join(path,fname)
    try:
        os.makedirs(mydir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise  # This was not a "directory exist" error..
    return mydir

index =int(sys.argv[1]) -1
config_name = sys.argv[2]
config= pd.read_pickle('config/' +  config_name)
              
i0=config['i0'][int(index)]
q0=config['q0'][int(index)]
b=config['b'][int(index)]
g=config['g'][int(index)]
d=config['d'][int(index)]
o=config['o'][int(index)]
s1=config['s1'][int(index)]
s2=config['s2'][int(index)]
s3=config['s3'][int(index)]
pop=config['pop'][int(index)]
file_name_list=config['filename'][int(index)]
style =int(config['style'][int(index)])
m_BA= config['m_BA'][int(index)]
p_ER = config['p_ER'][int(index)]
t1 =  60
      
if style == 0:
    G = nx.complete_graph(pop)
elif style == 1:
    G = nx.barabasi_albert_graph(pop,m_BA)    # The network 
elif style == 2:
    G = nx.erdos_renyi_graph(pop, p_ER, seed=None, directed=False)
elif style == 3:
    pickle_off = open ("degree_hill.txt", "rb")
    deg_distri = pickle.load(pickle_off)
    G = nx.configuration_model(deg_distri,create_using=nx.classes.graph.Graph)
elif style == 4:
    while True:
        try:
            G = nx.LFR_benchmark_graph(n=1000, tau1=2.5,tau2=1.5,  mu =0.6, average_degree=3)
            break
        except:
            pass
        
elif style == 5:
    G= nx.watts_strogatz_graph(n = 1000, k = 4, p = 0.3)
    
folder = 'data/'+  config_name
dir1= filecreation('/exports/eddie/scratch/s2006399/',folder)
if not os.path.exists(dir1+'/config_fil'): 
    config.to_pickle('config_fil')  # Writing a copy of the config file to the data directory

model = Imodel(i0=i0, q0=q0, b=b, g= g, d= d, o=o,s1=s1, s2=s2,s3=s3,net = G, filename =file_name_list, directory= dir1,index = index,style=style)
for i in range(t1+1):
    model.step()
run_data = model.datacollector.get_model_vars_dataframe()
#run_data.to_pickle(str(folder)+'/data_'+str(index))

run_data['b']=b
run_data['g']=g
run_data['d']=d
run_data['o']=o
run_data['s1']=s1
run_data['s2']=s2
run_data['s3']=s3

run_data['S']=run_data['agents_S'].str.len()
run_data['I']=run_data['agents_I'].str.len()
run_data['Q']=run_data['agents_Q'].str.len()
run_data['s_norm'] = run_data['S']/(run_data['S'] + run_data['I']+ run_data['Q'])
run_data['i_norm'] = run_data['I']/(run_data['S'] + run_data['I']+ run_data['Q'])
run_data['q_norm'] = run_data['Q']/(run_data['S'] + run_data['I']+ run_data['Q'])

run_data['quit_ratio']= run_data['Q'] / (run_data['I'] + run_data['Q'])


run_data.to_pickle(str(dir1)+"/run_data_"+str(index))
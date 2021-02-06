#from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random
import umap
from sklearn.model_selection import train_test_split

# All groups & experiments
# Group:{Exp:[Avg, Sub-avg, Rare]}

all_exp = {
    0:{
        0: [0,0,0]
    },
    1:{
        1:[10, 50, 40],
        2:[10, 45, 45],
        3:[10, 40, 50],
        4:[10, 35, 55],
        5:[10, 55, 35]
    },
    2:{
        6:[15, 50, 35],
        7:[15, 45, 40],
        8:[15, 40, 45],
        9:[15, 35, 50]
    },
    3:{
        10:[20, 45, 35],
        11:[20, 40, 40],
        12:[20, 35, 45]
    },
    4:{
        13:[25, 40, 35],
        14:[25, 35, 40]
    },
    5:{
        15:[30, 35, 35]
    },
    6:{
        16:[70, 20, 10, 'Average'],
        17:[10, 20, 70, 'Rare'],
        18:[10, 70, 20, 'Sub-Average'],
        19:[20, 70, 10, 'Sub-Average']
    },
    7:{
        20:[63, 27, 10, 'Average', 'Sub-Average', 'Rare'],
        21:[63, 10, 27, 'Average', 'Rare', 'Sub-avg'],
        22:[10, 63, 27, 'Sub-Average', 'Rare', 'Average'],
        23:[27, 63, 10, 'Sub-Average', 'Average', 'Rare'],
        24:[27, 10, 63, 'Rare', 'Average', 'Sub-Average'],
        25:[10, 27, 63, 'Rare', 'Sub-Average', 'Average']
    }
}

ds_dim = None

# List of functions:::
def printExps(groups):
    '''Listing all experiments'''
    print('\t\t\tAvg\tSub-Avg\tRare')
    print('--------------------------------------------')
    for group in groups:
        print(f'Group: {group}')
        this_group = groups[group]
        for exp in this_group:
            if group != 6 and group != 7:
                print(f'Experiment: {exp}>\t\t{this_group[exp][0]}\t{this_group[exp][1]}\t{this_group[exp][2]}')
            elif group == 6:
                print(f'Experiment: {exp}>\t\t{this_group[exp][0]}\t{this_group[exp][1]}\t{this_group[exp][2]}\t\t train on {this_group[exp][3]}')
            elif group == 7:
                print(f'Experiment: {exp}>\t\ttrain on {this_group[exp][3]} validate on {this_group[exp][4]} test on {this_group[exp][5]}')
            print('--------------------------------------------')

def expDetails(exps):
    '''Choosing group and experiment'''
    printExps(exps)
    train = None
    avg, sub, rare = None, None, None
    stratify = None
    g_idx = input('Enter group No.\t')
    exp_idx = input('Enter exp No.\t')
    ds = input('(1) MNIST\n(2) FashionMNIST\n(3) SmallNorb\n(4) Shapes3D\n(5) TrafficSign\n(6) Cifar10\n')
    model = input('(1) VGG16\n(2) InceptionV3\n(3) ResNet50\n')

    global ds_dim
    ds_dim = getDS(ds)[1]

    ds = getDS(ds)[0]
    if ds == 'Shapes3D':
        ds = getShapes3D()

    model = getModel(model)

    print('DATASET is:\t', ds)
    print('MODEL is:\t', model)

    if g_idx == '0' and exp_idx == '0':
        g_idx = 'GroundTruthExp'
        # strat, stratify = stratified()
        print('This is a Ground Truth Experiment')
        # print(f'The data split is {strat}')
        return ds, model, g_idx, exp_idx, avg, sub, rare, stratify, train

    [avg, sub, rare] = exps[int(g_idx)][int(exp_idx)][:3]
    if g_idx == '6':
        if exp_idx == '16':
            train = 'avg'
        elif exp_idx == '17':
            train = 'rare'
        elif exp_idx == '18' or exp_idx == '19':
            train = 'sub-avg'
        else:
            print('Error: Please enter valid expirement index [16, 17, 18 or 19]')
        print(f'Model will be trained on {train}')
        return ds, model, g_idx, exp_idx, avg, sub, rare, stratify, train

    if g_idx == '7':
        extreme_exp = getExtremeExp(int(exp_idx))
        if exp_idx == '20' or exp_idx == '21':
            train = 'avg'
        elif exp_idx == '22' or exp_idx == '23':
            train = 'sub-avg'
        elif exp_idx == '23' or exp_idx == '24':
            train = 'rare'
        else:
            print('Error: Please enter valid expirement index [20, 21, 22, 23, 24, 25]')
        print(f'Model will be trained on {extreme_exp[0]} validated on {extreme_exp[1]} and tested on {extreme_exp[2]}')
        return ds, model, g_idx, exp_idx, avg, sub, rare, stratify, train

    # strat, stratify = stratified()

    print(f'for Group {g_idx} Experiment {exp_idx} split as the following:')
    print('Avg\tSub-Avg\tRare')
    print(f'{avg}\t{sub}\t{rare}')
    # print(f'the data spliting is {strat}')
    return ds, model, g_idx, exp_idx, avg, sub, rare, stratify, train

def getDS(DS):
    '''Get DataSet number'''
    switcher = {
    '1': ['MNIST', (28, 28)],
    '2': ['FashionMNIST', (28, 28)],
    '3': ['SmallNorb', (96, 96)],
    '4': ['Shapes3D',(64, 64, 3)],
    '5': ['TrafficSign', (32, 32, 3)],
    '6': ['Cifar10', (32, 32, 3)]
    }
    return switcher.get(DS, 'Invalid Dataset choice')

def getShapes3D():
    '''Get Shapes3D training Label'''
    label = input('Shapes3D\n(1) label_floor_hue\n(2) label_object_hue\n(3) label_orientation\n(4) label_scale\n(5) label_shape')
    switcher = {
    '1': 'Shapes3D_floor_hue',
    '2': 'Shapes3D_object_hue',
    '3': 'Shapes3D_orientation',
    '4': 'Shapes3D_scale',
    '5': 'Shapes3D_shape'
    }
    return switcher.get(label, 'Invalid Dataset choice')

def getModel(model):
    '''Get DataSet number'''
    switcher = {
    '1': 'VGG',
    '2': 'Inception',
    '3': 'ResNet'
    }
    return switcher.get(model, 'Invalid Model choice')

def getExtremeExp(idx):
    '''Get Group 7 experiment details'''
    switcher = {
    20:['Average', 'Sub-Average', 'Rare'],
    21:['Average', 'Rare', 'Sub-avg'],
    22:['Sub-Average', 'Rare', 'Average'],
    23:['Sub-Average', 'Average', 'Rare'],
    24:['Rare', 'Average', 'Sub-Average'],
    25:['Rare', 'Sub-Average', 'Average']
    }
    error_msg = 'Please enter valid expirement index [20, 21, 22, 23, 24, 25]'
    return switcher.get(idx, error_msg)

# def stratified():
#     '''Check if data split stratified'''
#     stratify = input('Data split is stratified (Y/N)')
#     if stratify.lower() == 'y':
#         stratify = True
#         strat = 'stratified'
#     else:
#         stratify = False
#         strat = 'not stratified'
#     return strat, stratify

def TFDStoNP(tfds_DS):
    '''Convert from TensorFlow to Numpy for SmallNorb Dataset'''
    iterator = tfds_DS.__iter__()
    X = [[] for i in range(len(tfds_DS))]
    Y = [[] for i in range(len(tfds_DS))]
    for i in range(len(tfds_DS)):
        next_element = iterator.get_next()
        pt = next_element[0]
        en = next_element[1]
        X[i] = pt.numpy().reshape(96,96).copy()
        Y[i] = en.numpy().copy()
    X = np.array(X)
    Y = np.array(Y)
    return X, Y

#def create_dir(DataSet = str(), ModelName = str(), group_idx = str(), exp_idx = str(), stratify = None):
#  '''Create required directories for the experiment in Google Colab'''
#
#  if stratify:
#    st_path = '_withLables'
#  else:
#    st_path = ''
#
#  model_path = Path('/content/drive/My Drive/Thesis Notebooks/' + DataSet + '/' + ModelName)
#
#  group_folder = str('Group_' + group_idx)
#  exp_folder = str('Exp' + exp_idx + st_path)
#
#  group_path = Path(str(model_path) + '/' + group_folder)
#  exp_path = Path(str(group_path) + '/' + exp_folder)
#
#  if not model_path.exists():
#    print('Warning: Model Directory is not available')
#  elif not group_path.exists():
#    run_bash('cd {model_path}')
#    print('Model Directory is available')
#    run_bash('mkdir {group_folder}')
#    print('Group Directory is created')
#    run_bash('cd {group_path}')
#    print('Group Directory is available')
#    run_bash('mkdir {exp_folder}')
#    print('Experiment Directory is created')
#    run_bash('cd {exp_path}')
#    print('Experiment Directory is available')
#  elif not exp_path.exists():
#    run_bash('cd "group_path"')
#    print('Group Directory is available')
#    run_bash('mkdir "exp_folder"')
#    print('Experiment Directory is created')
#    run_bash('(cd "exp_path"')
#    print('Experiment Directory is available')
#  else:
#    run_bash('cd {exp_path}')
#    print('Experiment Directory is available')

#def run_bash(script):
# import subprocess
# import tempfile
#  '''Run bash in python'''
#  with tempfile.NamedTemporaryFile() as scriptfile:
#    scriptfile.write(script.encode(encoding='utf-8'))
#    scriptfile.flush()
#    subprocess.call(['/bin/bash', scriptfile.name])

def clustersDet(hdb_labels):
    '''Cluster details >>> No. of clusters, No. of Points in each cluster'''
    m = 0
    no_of_clusters = 0
    for i in range(10):
        m += np.count_nonzero(hdb_labels == i)
        if np.count_nonzero(hdb_labels == i) != 0:
            no_of_clusters += 1
    print(f'{i}\t', np.count_nonzero(hdb_labels == i))

    print('total Data point\t', m)
    print('total No. of Clusters\t', no_of_clusters)

    return no_of_clusters

def embedding1D(no_cluster, data, hdb_labels, y_labels, p_out = True):
    '''Clusters Embedding'''
    cluster_list = list() #Creat a list contains all clusters after Embedding
    label_list = list() #Creat a list contains all labels after Embedding
    original_image_list = list() #Original Images List

    for k in range(no_cluster):
        if p_out:
          print('\n\nEmbedding Cluster No. >>>\t', k)
        cluster_list.append(umap.UMAP(random_state=42, n_neighbors=30, min_dist=0, n_components=1).fit_transform(data[hdb_labels==k]))
        label_list.append(y_labels[hdb_labels==k])
        original_image_list.append(data[hdb_labels==k])

    return cluster_list, label_list, original_image_list

def expRegions(avg, sub, rare, p_out = True):
    '''define regions'''
    if avg<=0 or sub<=0 or rare<=0:
        print('Regions Error = At least One Region is Negative')
        return None
    elif avg + sub + rare != 100:
        print('Regions Error = Regions Sum is Not Equal to 100%')
        return None
    avg, sub, rare = avg/100, sub/100, rare/100
    R1_h = 0.5 + (avg/2)
    R1_l = 0.5 - (avg/2)
    R2_h = 1 - (rare/2)
    R2_l = 0 + (rare/2)
    av = R2_h - R2_l
    R = [R1_h, R1_l, R2_h, R2_l]
    if p_out:
        print(f'Average Region:\t{R[0]}-{R[1]}')
        print(f'Sub-Avg Region:\t{R[2]}-{R[0]} & {R[3]}-{R[0]}')
        print(f'Rare Region:\t{R[2]}-{1} & {0}-{R[3]}')
    return R

def splitData(n, avg, sub, rare, cluster_list, y_list, original_list, p_out = True):
    '''Data Splitting'''
    #creat seperate lists for splitted datasets
    R_1, R_2, R_3 = [[] for i in range(n)], [[] for i in range(n)], [[] for i in range(n)] #creat a list contains all Region 1 datasets
    R_1_labels, R_2_labels, R_3_labels = [[] for i in range(n)], [[] for i in range(n)], [[] for i in range(n)] #creat a list contains all Region 1 labels
    R_1_original, R_2_original, R_3_original = [[] for i in range(n)], [[] for i in range(n)], [[] for i in range(n)] #creat a list contains all Region 1 Original Images

    # Regions / Quantile Limits
    # Regions Limitis:
    regions_perc = expRegions(avg, sub, rare)
    R1_high = regions_perc[0]
    R1_low = regions_perc[1]
    R2_high = regions_perc[2]
    R2_low = regions_perc[3]

    # Number of Rows list is Required for splitting/looping on all rows of each cluster
    no_rows = list() #creat a list for Number of Rows values of clusters
    co1 = 0 # creat counter to be used inside the loop

    for cluster_i in cluster_list:
        no_rows.append(cluster_i.shape[0]) #Cluster number of rows
        co1 += 1 #end of the loop, move to the next cluster, increase the counter by 1

    # Cluster splitting > 3 divisions (Average (Region 1), Sub-Average (Region 2), Rare (Region 3))
    co2 = 0 # creat counter to be used inside the loop

    for cluster_q in cluster_list:
        for r in range(no_rows[co2]):
            if cluster_q[r][0] <= np.quantile(cluster_q, R1_high, axis=0) and cluster_q[r][0] >= np.quantile(cluster_q, R1_low, axis=0):
                R_1[co2] = np.append(R_1[co2], np.array(cluster_q[r], dtype= np.float32))
                R_1_labels[co2] = np.append(R_1_labels[co2], y_list[co2][r])
                R_1_original[co2].append(original_list[co2][r])
            elif cluster_q[r][0] <= np.quantile(cluster_q, R2_high, axis=0) and cluster_q[r][0] >= np.quantile(cluster_q, R2_low, axis=0):
                R_2[co2] = np.append(R_2[co2], np.array(cluster_q[r], dtype= np.float32))
                R_2_labels[co2] = np.append(R_2_labels[co2], y_list[co2][r])
                R_2_original[co2].append(original_list[co2][r])
            else:
                R_3[co2] = np.append(R_3[co2], np.array(cluster_q[r], dtype= np.float32))
                R_3_labels[co2] = np.append(R_3_labels[co2], y_list[co2][r])
                R_3_original[co2].append(original_list[co2][r])

        R_1_original[co2] = np.array(R_1_original[co2])
        R_2_original[co2] = np.array(R_2_original[co2])
        R_3_original[co2] = np.array(R_3_original[co2])

        if p_out:
            #Printing splitted datasets (cluster regions) details
            print("\n************************************************************************************************************\n\n")
            print(f"\n>>>Cluster {co2} Splitted datasets:::<<<\n")
            print("\t\tData Size \tLabels Size")
            print("Average \t", R_1[co2].size, "\t\t", R_1_labels[co2].size) if len(R_1[co2]) != 0 else print("Average \t","empty\t\t empty")
            print("Sub-Average \t", R_2[co2].size, "\t\t", R_2_labels[co2].size) if len(R_2[co2]) != 0 else print("Sub-Average \t","empty\t\t empty")
            print("Rare \t\t", R_3[co2].size, "\t\t", R_3_labels[co2].size) if len(R_3[co2]) != 0 else print("Rare \t\t","empty\t\t empty")

        co2 += 1 #end of the loop, increase the counter by 1

    return R_1, R_2, R_3, R_1_labels, R_2_labels, R_3_labels, R_1_original, R_2_original, R_3_original

def showSplit(R_1, R_2, R_3, cluster_no):
    '''Plotting splitted Datasets'''
    #Plotting splitted datasets (cluster regions) histograms without KDE, with Borders
    print("\n************************************************************************************************************\n")
    print(f">>>Cluster {cluster_no} Splitted datasets:::<<<\n")
    print("Average \t", R_1[cluster_no].size) if len(R_1[cluster_no]) != 0 else print("Average \t","empty")
    print("Sub-Average \t", R_2[cluster_no].size) if len(R_2[cluster_no]) != 0 else print("Sub-Average \t","empty")
    print("Rare \t\t", R_3[cluster_no].size) if len(R_3[cluster_no]) != 0 else print("Rare \t\t","empty")

    plt.figure(figsize=(12,7))

    #Plotting Region1 data
    sns.distplot(R_1[cluster_no], kde=False, hist=True, rug=True, bins=5, color='green', hist_kws = {'color':'#35dab6', 'edgecolor':'#d6c417',
                        'linewidth':1, 'linestyle':'--', 'alpha':0.9}, norm_hist= True);
    #Plotting Region2 data
    sns.distplot(R_2[cluster_no], kde=False, hist=True, rug=True, bins=5, color='blue', hist_kws = {'color':'#0096fe', 'edgecolor':'#e1f7d5',
                        'linewidth':1, 'linestyle':'--', 'alpha':0.9}, norm_hist= True);
    #Plotting Region3 data
    sns.distplot(R_3[cluster_no], kde=False, hist=True, rug=True, bins=5, color='red', hist_kws = {'color':'#ff0000', 'edgecolor':'#e6dbdc',
                        'linewidth':2, 'linestyle':'--', 'alpha':0.9}, norm_hist= True);

    print("\n************************************************************************************************************\n")

    return None

def showSamples(R_1_original, R_2_original, R_3_original, R_1_labels, R_2_labels, R_3_labels):
    '''Show samples from the split regions for each cluster'''
    global ds_dim

    cluster_list = [R_1_original, R_2_original, R_3_original]
    cluster_labels = [R_1_labels, R_2_labels, R_3_labels]
    fig, ax = plt.subplots(nrows=3, ncols=5 , figsize=(10,7) )
    fig.tight_layout(pad=1.5)
    for i_row , row in enumerate(ax):
        for i_col , col in enumerate(row):
            randm_num = random.choice(range(len(cluster_list[i_row])))
            img = np.array(cluster_list[i_row][randm_num])
            label = cluster_labels[i_row][randm_num]
            if len(ds_dim) == 2:
                img = img.reshape(ds_dim[0], ds_dim[1])
            elif len(ds_dim) == 3:
                img = img.reshape(ds_dim[0], ds_dim[1], ds_dim[2])
            col.imshow(img , interpolation='None')
            col.set_title(" Label : {}".format(str(label)))
            col.set_xlabel("Region : {}".format(str(i_row+1)))

    return None

def prepareData(no_cluster, R_1_original, R_1_labels, R_2_original, R_2_labels, R_3_original, R_3_labels, Xtest, Ytest, stratify = False, is_colored = False):
    '''Data Preparation and Train, Test, Validate splitting for training'''
    x_test, y_test = Xtest, Ytest
    x_R01, y_R01 = [], []
    x_R02, y_R02 = [], []
    x_R03, y_R03 = [], []
    for z in range(no_cluster):
        x_R01.extend(R_1_original[z])
        y_R01.extend(R_1_labels[z])
        x_R02.extend(R_2_original[z])
        y_R02.extend(R_2_labels[z])
        x_R03.extend(R_3_original[z])
        y_R03.extend(R_3_labels[z])

    x_train, y_train, x_valid, y_valid = TrainValidSplit(x_R01, y_R01, x_R02, y_R02, x_R03, y_R03, stratify)

    if is_colored:
        x_train = reshapeColored(x_train)
        x_valid = reshapeColored(x_valid)
        x_test = reshapeColored(x_test)

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def TrainValidSplit(x_Region01, y_Region01, x_Region02, y_Region02, x_Region03, y_Region03, stratify):
    '''Train & Valid data splitting'''
    y_stratify = y_Region01 if stratify else None
    x_train01, x_valid01, y_train01, y_valid01 = train_test_split(x_Region01, y_Region01, train_size= 0.7, test_size= 0.3, stratify = y_stratify)

    y_stratify = y_Region02 if stratify else None
    x_train02, x_valid02, y_train02, y_valid02 = train_test_split(x_Region02, y_Region02, train_size= 0.7, test_size= 0.3, stratify = y_stratify)

    y_stratify = y_Region03 if stratify else None
    x_train03, x_valid03, y_train03, y_valid03 = train_test_split(x_Region03, y_Region03, train_size= 0.7, test_size= 0.3, stratify = y_stratify)

    x_train = np.vstack((x_train01, x_train02, x_train03))
    y_train = np.concatenate((y_train01, y_train02, y_train03))
    x_valid = np.vstack((x_valid01, x_valid02, x_valid03))
    y_valid = np.concatenate((y_valid01, y_valid02, y_valid03))

    return x_train, y_train, x_valid, y_valid


def prepareDataSpecial(no_cluster, R_1_original, R_1_labels, R_2_original, R_2_labels, R_3_original, R_3_labels, Xtest, Ytest, train = 'na', is_colored = False):
    '''Special Experiments Data Preparation - Train, Validate splitting for training'''
    x_test, y_test = Xtest, Ytest
    x_R01, y_R01 = [], []
    x_R02, y_R02 = [], []
    x_R03, y_R03 = [], []
    for z in range(no_cluster):
        x_R01.extend(R_1_original[z])
        y_R01.extend(R_1_labels[z])
        x_R02.extend(R_2_original[z])
        y_R02.extend(R_2_labels[z])
        x_R03.extend(R_3_original[z])
        y_R03.extend(R_3_labels[z])

    x_R01, y_R01 = np.array(x_R01), np.array(y_R01)
    x_R02, y_R02 = np.array(x_R02), np.array(y_R02)
    x_R03, y_R03 = np.array(x_R03), np.array(y_R03)

    if train.lower() == 'avg':
        x_train, y_train = x_R01, y_R01
        x_valid = np.vstack((x_R02, x_R03))
        y_valid = np.concatenate((y_R02, y_R03))
    elif train.lower() == 'sub-avg':
        x_train, y_train = x_R02, y_R02
        x_valid = np.vstack((x_R01, x_R03))
        y_valid = np.concatenate((y_R01, y_R03))
    elif train.lower() == 'rare':
        x_train, y_train = x_R03, y_R03
        x_valid = np.vstack((x_R01, x_R02))
        y_valid = np.concatenate((y_R01, y_R02))
    else:
        print('Error: Please specify the data groups correctly')
        return

    x_train, y_train = shuffleData(x_train, y_train)
    x_valid, y_valid = shuffleData(x_valid, y_valid)

    if is_colored:
        x_train = reshapeColored(x_train)
        x_valid = reshapeColored(x_valid)
        x_test = reshapeColored(x_test)

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def prepareDataExtreme(no_cluster, R_1_original, R_1_labels, R_2_original, R_2_labels, R_3_original, R_3_labels, exp_idx = '', is_colored = False):
    '''Special Experiments Data Preparation - Train, Validate, Testing splitting'''
    if not exp_idx:
        print('Please provide expirement index')
        return
    else:
        exp_idx = int(exp_idx)
    if exp_idx not in [20, 21, 22, 23, 24, 25]:
        print('Please enter valid expirement index [20, 21, 22, 23, 24, 25]')
        return

    x_R01, y_R01 = [], []
    x_R02, y_R02 = [], []
    x_R03, y_R03 = [], []
    for z in range(no_cluster):
        x_R01.extend(R_1_original[z])
        y_R01.extend(R_1_labels[z])
        x_R02.extend(R_2_original[z])
        y_R02.extend(R_2_labels[z])
        x_R03.extend(R_3_original[z])
        y_R03.extend(R_3_labels[z])

    x_R01, y_R01 = np.array(x_R01), np.array(y_R01)
    x_R02, y_R02 = np.array(x_R02), np.array(y_R02)
    x_R03, y_R03 = np.array(x_R03), np.array(y_R03)

    extreme_exp = getExtremeExp(exp_idx)
    train, valid = extreme_exp[0], extreme_exp[1]
    if train == 'Average':
        x_train, y_train = x_R01, y_R01
        if valid == 'Sub-Average':
            x_valid, y_valid = x_R02, y_R02
            x_test, y_test = x_R03, y_R03
        elif valid == 'Rare':
            x_valid, y_valid = x_R03, y_R03
            x_test, y_test = x_R02, y_R02
    elif train == 'Sub-Average':
        x_train, y_train = x_R02, y_R02
        if valid == 'Average':
            x_valid, y_valid = x_R01, y_R01
            x_test, y_test = x_R03, y_R03
        elif valid == 'Rare':
            x_valid, y_valid = x_R03, y_R03
            x_test, y_test = x_R01, y_R01
    elif train == 'Rare':
        x_train, y_train = x_R03, y_R03
        if valid == 'Average':
            x_valid, y_valid = x_R01, y_R01
            x_test, y_test = x_R02, y_R02
        elif valid == 'Sub-Average':
            x_valid, y_valid = x_R02, y_R02
            x_test, y_test = x_R01, y_R01

    x_train, y_train = shuffleData(x_train, y_train)
    x_valid, y_valid = shuffleData(x_valid, y_valid)
    x_test, y_test = shuffleData(x_test, y_test)

    if is_colored:
        x_train = reshapeColored(x_train)
        x_valid = reshapeColored(x_valid)
        x_test = reshapeColored(x_test)

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def shuffleData(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def reshapeColored(data):
    img_dim = int(math.sqrt(data.shape[1]/3))
    data = data.reshape(-1, img_dim, img_dim, 3)
    return data

def prepareDataGT(x_data, x_test, y_data, y_test, stratify=None):
    '''Prepare Data for Ground Truth Experiments'''

    Y_st = y_data if stratify else None

    x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, train_size= 0.7, test_size= 0.3, stratify = Y_st)

    print(x_train.shape, y_train.shape)
    print(x_valid.shape, y_valid.shape)
    print(x_test.shape, y_test.shape)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

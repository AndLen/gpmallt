import lzma

import pandas as pd
from sklearn import preprocessing


def delim_map(delim):
    switch = {
        "comma": ",",
        "space": " "
    }
    return switch.get(delim)


def read_data(filename):
    filename1 = filename.with_suffix('').with_suffix('.data')
    if filename1.exists():
        with open(filename1,mode='rt') as f:
            first_line = f.readline()
            config = first_line.strip().split(",")
    else:
        filename2 = filename.with_suffix('.data.xz')
        if filename2.exists():
            print('compressed .data')
            with lzma.open(filename2,mode='rt') as f:
                first_line = f.readline()
                config = first_line.strip().split(",")
            filename = filename2

        else:
            filename3 = filename.with_suffix('').with_suffix('.csv')
            if filename3.exists():
                print('raw csv. Assuming class at end.')
                with open(filename3, mode='rt') as f:
                    first_line = f.readline()
                    config = ['classLast',first_line.count(','),None,'comma']
                filename = filename3
            else:
                raise ValueError(filename)


    classPos = config[0]
    num_feat = int(config[1])

    feat_labels = ['f' + str(x) for x in range(num_feat)]
    if classPos == "classFirst":
        feat_labels.insert(0, "class")
    elif classPos == "classLast":
        feat_labels.append("class")
    else:
        raise ValueError(classPos)

    delim = delim_map(config[3])

    rawData = pd.read_csv(filename, delimiter=delim, skiprows=1, header=None, names=feat_labels)
    labels = rawData['class']
    data = rawData.drop('class', axis=1)
    min_max_scaler = preprocessing.MinMaxScaler()
    data = min_max_scaler.fit_transform(data)

#    data[data.columns] = min_max_scaler.fit_transform(data[data.columns])
    return {"data": data, "labels":labels}
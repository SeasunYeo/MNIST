import numpy as np
import struct
Data_Path = './mnist_datase/t10k-images-idx3-ubyte'
Label_Path = './mnist_datase/t10k-labels-idx1-ubyte'
with open(Data_Path, 'rb') as Data_File:
    Data_Buf = Data_File.read()
Data_Index = 0
Data_Magic, Data_Num, Data_Rows, Data_Cols = struct.unpack_from(
    '>IIII', Data_Buf, Data_Index)
print(Data_Num)
Data_Index += struct.calcsize('>IIII')
Data = []
for x in range(Data_Num):
    Image = struct.unpack_from('>784B', Data_Buf, Data_Index)
    Data_Index += struct.calcsize('>784B')
    Image = list(Image)
    for item in range(len(Image)):
        if Image[item] > 1:
            Image[item] = 1
        Image[item]=Image[item]/1
    Image=np.reshape(Image,(28,28))
    Image.resize((Image.size,1))
    Data.append(Image)
with open(Label_Path,'rb') as Label_File:
    Label_Buf=Label_File.read()
Label_Index=0
Label_Magic,Label_Num=struct.unpack_from('>II',Label_Buf,Label_Index)
print(Label_Num)
Label_Index+=struct.calcsize('>II')
Label=[]
for x in range(Label_Num):
    Num=struct.unpack_from('>1B',Label_Buf,Label_Index)
    Label_Index+=struct.calcsize('1B')
    Num=Num[0]
    Label.append(Num)
np.save('./data/test.npy',[Data,Label])

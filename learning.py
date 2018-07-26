from layers import *


def main():
    datalayer1 = Data('./data/train.npy', 600)
    datalayer2 = Data('./data/test.npy', 10000)
    inner_layers = []
    inner_layers.append(FullyConnect(28*28, 10))
    inner_layers.append(Sigmoid())
    losslayer = QuadraticLoss()
    accuracy = Accuracy()
    for layer in inner_layers:
        layer.lr = 20.0
    epochs =100 
    for i in range(epochs):
        print('epochs:', i)
        losssum = 0
        iters = 0
        while True:
            data, pos = datalayer1.forward()
            x, label = data
            for layer in inner_layers:
                x = layer.forward(x)
            loss = losslayer.forward(x, label)
            losssum += loss
            iters += 1
            d = losslayer.backward()
            for layer in inner_layers[::-1]:
                d = layer.backward(d)
            if pos == 0:
                data, _ = datalayer2.forward()
                x, label = data
                for layer in inner_layers:
                    x = layer.forward(x)
                accu = accuracy.forward(x, label)
                print("loss:", losssum/iters)
                print("accuracy:", accu)
                break


if __name__ == '__main__':
    main()

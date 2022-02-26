

from model import GestureDatasetFromFile

if __name__ == '__main__':
    dataset = GestureDatasetFromFile('data')

    print(len(dataset))

    print(dataset[0])
    x, y = dataset[100]

    print(x.shape)
    print(y)

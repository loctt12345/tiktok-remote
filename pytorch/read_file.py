
def read_txt(path):
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            x = line[:-1]
            data.append(x)

    for i in range(len(data)):
        arr = data[i].split(', ')
        for j in range(len(arr)):
            try:
                arr[j] = float(arr[j])
            except:
                if j == 0:
                    arr[j] = float(arr[j][1:])
                else :
                    arr[j] = float(arr[j][0:-1])
        data[i] = arr
    return data

def split_data(data, step):
    new_data = []
    for i in range(len(data)):
        x = []
        for j in range(0, len(data[i]) - 1, step):
            x.append(data[i][j : j + step])
        
        new_data.append(x)
    return new_data

def process_data(path, step):
    data = read_txt(path)
    data = split_data(data, step)
    return data

def main():
    # data = read_txt('../data/UP.txt')
    # data = split_data(data, 84)
    data = process_data('../data/LIKE.txt', 84)
    print(len(data))

if __name__ == '__main__':
    main()

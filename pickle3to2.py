import argparse
import pickle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', type=str)
    args = parser.parse_args()

    with open(args.fname, 'rb') as file:
        obj = pickle.load(file)
    with open(args.fname, 'wb') as file:
        pickle.dump(obj, file, protocol=2)


if __name__ == '__main__':
    main()

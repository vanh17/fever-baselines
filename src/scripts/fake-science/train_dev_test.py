import sys
import io


def main():
    if len(sys.argv) != 5:
        exit("[usage] python3 train_dev_test.py path_to_splitting_file portion_of_training portion_of_dev path_to_output_folder")
    with io.open(sys.argv[1]) as input, io.open(sys.argv[4]+"/train.jsonl", "w+") as train, io.open(sys.argv[4]+"/dev.jsonl", "w+") as dev, io.open(sys.argv[4]+"/test.jsonl", "w+") as test:
        portion_train = int(sys.argv[2])
        portion_dev = int(sys.argv[3])
        lines = input.read().splitlines()
        length = len(lines)
        train_list =  lines[:int(portion_train*length/100)]
        dev_list = lines[int(portion_train*length/100):int((portion_train + portion_dev)*length/100)]
        test_list = lines[int((portion_train + portion_dev)*length/100):]
        train.write("\n".join(train_list))
        dev.write("\n".join(dev_list))
        test.write("\n".join(test_list))


if __name__ == '__main__':
    main()

import argparse


def combine(combined, file1, file2, multiply):
    with open(combined, "a+", encoding='utf-8') as c, open(file1, "r", encoding='utf-8') as f1, \
            open(file2, "r", encoding='utf-8') as f2:
        for line in f1.readlines():
            c.write(line)
        for line in f2.readlines():
            i = 0
            while i < multiply:
                i = i+1
                c.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('combined', type=str, help='/path/to/combined_folder/')
    # pass the value to target. This is the folder where you want to store newly 3 dev, train, test set.
    parser.add_argument('target', type=str, help='/path/to/target_folder/')
    # pass the value to source. This is the folder where you want to store newly 3 dev, train, test set.
    parser.add_argument('source',type=str, help='/path/to/source_folder/')
    parser.add_argument('multiply', type=int, help='multiplying factor')
    args = parser.parse_args()
    # code for creating train, dev, test respectively
    combine(args.combined + "train_" + str(args.multiply) + ".jsonl",
            args.target + "train_fever.jsonl", args.source + "train.jsonl", args.multiply)
    combine(args.combined + "dev_" + str(args.multiply) + ".jsonl",
            args.target + "dev_fever.jsonl", args.source + "dev.jsonl", args.multiply)
    combine(args.combined + "test_" + str(args.multiply) + ".jsonl",
            args.target + "test_fever.jsonl", args.source + "test.jsonl", args.multiply)
    # combine('train.ns.pages.p1.jsonl', 'fake-science-db.jsonl')


import argparse


def combine(file1, file2, multiply):
    with open(file1, "a+", encoding='utf-8') as f1, open(file2, "r", encoding='utf-8') as f2:
        for line in f2.readlines():
            i = 0
            while i < multiply:
                i = i+1
                f1.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='/path/to/target/file.jsonl')
    parser.add_argument('source',type=str, help='/path/to/source/file.jsonl')
    parser.add_argument('multiply', type=int, help='multiplying factor')
    args = parser.parse_args()
    combine(args.target, args.source, args.multiply)
    # combine('train.ns.pages.p1.jsonl', 'fake-science-db.jsonl')


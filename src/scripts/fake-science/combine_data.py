import argparse


def combine(combined, file1, file2, multiply):
    with open(combined, "a+", encoding='utf-8') as c, open(file1, "r", encoding='utf-8') as f1, open(file2, "r", encoding='utf-8') as f2:
        for line in f1.readlines():
            c.write(line)
        for line in f2.readlines():
            i = 0
            while i < multiply:
                i = i+1
                c.write(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('combined', type=str, help='/path/to/combined/file.jsonl')
    parser.add_argument('target', type=str, help='/path/to/target/file.jsonl')
    parser.add_argument('source',type=str, help='/path/to/source/file.jsonl')
    parser.add_argument('multiply', type=int, help='multiplying factor')
    args = parser.parse_args()
    combine(args.combined, args.target, args.source, args.multiply)
    # combine('train.ns.pages.p1.jsonl', 'fake-science-db.jsonl')


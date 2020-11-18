import sys

from ruwordnet_reader import RuWordnet

if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise Exception("Required arguments: <input-dir> <output-path>")

    input_dir = sys.argv[1]
    out_dir = sys.argv[2]

    rwn = RuWordnet(db_path=out_dir, ruwordnet_path=input_dir, with_lemmas=False)

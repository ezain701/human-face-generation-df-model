import argparse

from cleanfid import fid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_dir", type=str, required=True)
    parser.add_argument("--fake_dir", type=str, required=True)
    args = parser.parse_args()

    score = fid.compute_fid(args.real_dir, args.fake_dir)
    print(f"FID: {score:.4f}")


if __name__ == "__main__":
    main()

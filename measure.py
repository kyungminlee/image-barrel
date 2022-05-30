import imagebarrel as IB
import tqdm
import argparse
import cv2
import json
import os.path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    parser.add_argument('--stable', type=int, default=4, help="Number of frames required for stability check")
    parser.add_argument('--rescale', type=int, default=8, help="Rescaling factor for stability check")

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"file path: {args.filepath}")
    print(f"number of frames: {length}")
    print(f"frames per second: {fps}")
    
    output_filepath = f'{args.filepath}.measure.csv'
    with open(output_filepath, 'w') as outf:
        outf.write(f'# filename: {os.path.basename(args.filepath)}\n')
        outf.write(f'# stable: {args.stable}\n')
        outf.write(f'# rescale: {args.rescale}\n')
        outf.flush()

        barrel = IB.ImageBarrel(args.stable)

        target_shape = None
        iframe = 0
        bar = tqdm.tqdm(total=length)
        while True:
            success, img = cap.read()
            if not success:
                break
            if target_shape is None:
                target_shape = tuple(x // args.rescale for x in img.shape[:2])
            img = cv2.resize(img, target_shape)

            barrel.append(img)
            
            r = barrel.range(args.stable)
            outf.write(f'{iframe},{r:.18e}\n')
            iframe += 1
            bar.update()
        bar.close()
    cap.release()


if __name__=='__main__':
    main()

import imagebarrel as IB
import tqdm
import argparse
import cv2
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('filepath', type=str)
    parser.add_argument('--aggregate', type=int, default=128, help="Maximum number of frames to aggregate for a single output slide")
    parser.add_argument('--stable', type=int, default=4, help="Number of frames required for stability check")
    parser.add_argument('--rescale', type=int, default=8, help="Rescaling factor for stability check")
    parser.add_argument('--threshold', type=float, required=True, help="Threshold for stability check")
    parser.add_argument('--prefix', type=str, default='export', help="Prefix for the output slide images")

    args = parser.parse_args()

    cap = cv2.VideoCapture(args.filepath)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"file path: {args.filepath}")
    print(f"number of frames: {length}")
    print(f"frames per second: {fps}")
    
    img_list = IB.ImageAggregate(args.aggregate)
    mbarrel = IB.ImageBarrel(args.stable)

    target_shape = None
    iframe = 0
    bar = tqdm.tqdm(total=length)
    export_img_count = 0
    while True:
        success, img = cap.read()
        if not success:
            break
        if target_shape is None:
            target_shape = tuple(x // args.rescale for x in img.shape[:2])
        small_img = cv2.resize(img, target_shape)

        mbarrel.append(small_img)
        
        if mbarrel.range(args.stable) <= args.threshold:
            img_list.append(img)
        else:
            if len(img_list) > 0:
                output_img = img_list.median()
                cv2.imwrite(f'{args.prefix}-{export_img_count:04d}.png', output_img)
                img_list.clear()
                export_img_count += 1
        iframe += 1
        bar.update()

    if len(img_list) > 0:
        output_img = img_list.median()
        cv2.imwrite(f'{args.prefix}-{export_img_count:04d}.png', output_img)
        img_list.clear()
        export_img_count += 1

    bar.close()
    cap.release()



if __name__=='__main__':
    main()

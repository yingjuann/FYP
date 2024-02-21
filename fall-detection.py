import cv2
import time
import torch
import argparse
import numpy as np
from utils.datasets import letterbox
from utils.torch_utils import select_device
from models.experimental import attempt_load
from utils.plots import output_to_keypoint, plot_skeleton_kpts
from utils.general import non_max_suppression_kpt, strip_optimizer
from torchvision import transforms


def falling_alarm(image, bbox):
    x_min, y_min, x_max, y_max = bbox
    cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), color=(0, 0, 255),
                  thickness=5, lineType=cv2.LINE_AA)
    cv2.putText(image, 'Person Fell down', (11, 100), 0, 1, [0, 0, 255], thickness=3, lineType=cv2.LINE_AA)


@torch.no_grad()
def run(poseweights='yolov7-w6-pose.pt', source='video_1.mp4', device='cpu'):

    path = source
    ext = path.split('/')[-1].split('.')[-1].strip().lower()
    if ext not in ["mp4", "webm", "avi"] and not ext.isnumeric():
        print("Unsupported video format. Please provide a valid video file.")
        return

    input_path = int(path) if path.isnumeric() else path
    device = select_device(device)
    half = device.type != 'cpu'
    model = attempt_load(poseweights, map_location=device)
    _ = model.eval()

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        print('Error while trying to read video. Please check path again')
        return

    frame_width, frame_height = int(cap.get(3)), int(cap.get(4))

    vid_write_image = letterbox(
        cap.read()[1], (frame_width), stride=64, auto=True)[0]
    resize_height, resize_width = vid_write_image.shape[:2]
    out_video_name = "output" if path.isnumeric else f"{input_path.split('/')[-1].split('.')[0]}"
    out = cv2.VideoWriter(
        f"{out_video_name}_result4.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 30, (resize_width, resize_height))

    frame_count, total_fps = 0, 0

    while cap.isOpened:

        print(f"Frame {frame_count} Processing")
        ret, frame = cap.read()
        if ret:
            orig_image = frame

            # preprocess image
            image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
            image = letterbox(image, (frame_width), stride=64, auto=True)[0]
            image_ = image.copy()
            image = transforms.ToTensor()(image)
            image = torch.tensor(np.array([image.numpy()]))

            image = image.to(device)
            image = image.float()
            start_time = time.time()

            with torch.no_grad():
                output, _ = model(image)

            output = non_max_suppression_kpt(output, 0.25, 0.65, nc=model.yaml['nc'], nkpt=model.yaml['nkpt'], kpt_label=True)
            output = output_to_keypoint(output)
            img = image[0].permute(1, 2, 0) * 255
            img = img.cpu().numpy().astype(np.uint8)

            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            for idx in range(output.shape[0]):
                plot_skeleton_kpts(img, output[idx, 7:].T, 3)

            # Detect falling action based on human height < human width
            for pose in output:
                xmin, ymin = (pose[2] - pose[4] / 2), (pose[3] - pose[5] / 2)
                xmax, ymax = (pose[2] + pose[4] / 2), (pose[3] + pose[5] / 2)
                height = ymax - ymin
                width = xmax - xmin

                if height < width:
                    falling_alarm(img, (xmin, ymin, xmax, ymax))

            cv2.imshow("Detection", img)
            key = cv2.waitKey(1)
            if key == ord('c'):
                break

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            total_fps += fps
            frame_count += 1
            out.write(img)
        else:
            break

    cap.release()
    out.release()
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--poseweights', nargs='+', type=str, default='yolov7-w6-pose.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default='input_video.mp4', help='path to video or 0 for webcam')
    parser.add_argument('--device', type=str, default='cpu', help='cpu/0,1,2,3(gpu)')
    opt = parser.parse_args()
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    strip_optimizer(opt.device, opt.poseweights)
    main(opt)

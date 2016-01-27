import time
import numpy as np
try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import GObject, Gst
    GObject.threads_init()
    Gst.init(None)
    gstreamer_available = True
except ImportError:
    gstreamer_available = False

try:
    import cv2
    opencv_available = True
except ImportError:
    opencv_available = False


class Cv2VideoCapture(object):
    def __init__(self, device=None, size=None, fps=None, sync=False):
        self.device = device or 0
        self.size = size or (480, 640)
        fps = fps or 30

        self.cap = cv2.VideoCapture(self.device)
        cap_height, cap_width = self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT), self.cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        if cap_height != self.size[0]:
            self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, self.size[0])
        if cap_width != self.size[1]:
            self.cap.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, self.size[1])
        cap_fps = self.cap.get(cv2.cv.CV_CAP_PROP_FPS)
        if cap_fps != fps:
            self.cap.set(cv2.cv.CV_CAP_PROP_FPS, fps)
        if sync:
            raise ValueError("sync not supported")

    def get(self):
        ret, image = self.cap.read()
        if not ret:
            raise RuntimeError("Could not read the image from the capture device")
        return image, self.get_time() # TODO: use actual time of the frame

    def get_time(self):
        return time.time()

    def release(self):
        self.cap.release()


class GstVideoCapture(object):
    def __init__(self, device=None, size=None, fps=None, sync=False):
        self.device = '/dev/video%d'%(device or 0)
        self.size = size or (480, 640)
        fps = fps or 30
        self._create_main_pipeline(self.device, self.size, fps, sync)
        self._start()

    def _create_main_pipeline(self, device, size, fps, sync):
        self.pipeline = Gst.Pipeline()

        self.source = Gst.ElementFactory.make('v4l2src', 'source')
        self.source.set_property('device', device)
        self.source.set_property('do-timestamp', 'true')
        # run 'v4l2-ctl --list-ctrls' for full list of controls
        struct, _ = Gst.structure_from_string('name,\
                                               white_balance_temperature_auto=(bool){0},\
                                               backlight_compensation=(int){0},\
                                               exposure_auto=0,\
                                               focus_auto=(bool){0}')
        self.source.set_property('extra-controls', struct)

        caps = Gst.caps_from_string('video/x-raw,format=(string){BGR},width=%d,height=%d,framerate=%d/1'%(size[1], size[0], fps))
        self.sink = Gst.ElementFactory.make('appsink', 'sink')
        self.sink.set_property('emit-signals', True)
        self.sink.set_property('sync', sync)
        self.sink.set_property('drop', True)
        self.sink.set_property('max-buffers', 1)
        self.sink.set_property('caps', caps)
        self.sink.emit('pull-preroll')

        self.pipeline.add(self.source)
        self.pipeline.add(self.sink)

        Gst.Element.link(self.source, self.sink)

    def get(self):
        sample = self.sink.emit("pull-sample")
        if sample is None:
            raise RuntimeError("Could not read the image from the capture device")
        buf = sample.get_buffer()
        image = np.ndarray(shape=(self.size[0], self.size[1], 3),
                           buffer=buf.extract_dup(0, buf.get_size()),
                           dtype=np.uint8)
        return image, self.pipeline.base_time + buf.pts - buf.duration

    def get_time(self):
        return self.pipeline.clock.get_time()

    def _start(self):
        state_change_return = self.pipeline.set_state(Gst.State.PLAYING)
        if state_change_return == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Failed to start capture device %s'%self.device)

    def release(self):
        self.pipeline.set_state(Gst.State.NULL)


class VideoCapture(object):
    def __init__(self, device=None, size=None, fps=None, sync=False, backend=None):
        if backend is None:
            # select backend that is available
            if gstreamer_available:
                self.cap = GstVideoCapture(device, size, sync)
            elif opencv_available:
                self.cap = Cv2VideoCapture(device, size)
            else:
                # if no backend available, raise exception
                self.video = GstVideoCapture(device, size, sync)
        elif backend == "gstreamer":
            self.cap = GstVideoCapture(device, size, sync)
        elif backend == "opencv":
            self.video = Cv2VideoCapture(device, size)
        else:
            raise ValueError("Unknown backend: %s", backend)

    def get(self):
        return self.cap.get()

    def get_time(self):
        return self.cap.get_time()

    def release(self):
        self.cap.release()


def main():
    import cv2
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--size', nargs=2, type=int, default=None, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--sync', type=int, default=1)
    parser.add_argument('--backend', type=str, default=None)

    args = parser.parse_args()

    cap = VideoCapture(device=args.device, size=args.size, fps=args.fps, sync=args.sync, backend=args.backend)
    while True:
        try:
            image, _ = cap.get()
            cv2.imshow("Image window", image)
            key = cv2.waitKey(100)
            key &= 255
            if key == 27 or key == ord('q'):
                print "Pressed ESC or q, exiting"
                break
        except KeyboardInterrupt:
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

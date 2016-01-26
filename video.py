import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst

GObject.threads_init()
Gst.init(None)

class GstVideoCapture(object):
    def __init__(self, device=None, size=None, fps=None, sync=False):
        self.device = device or '/dev/video0'
        self.size = size or (480, 640)
        fps = fps or 30
        self._create_main_pipeline(self.device, self.size, fps, sync)

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

    def start(self):
        state_change_return = self.pipeline.set_state(Gst.State.PLAYING)
        if state_change_return == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError('Failed to start capture device %s'%self.device)

    def stop(self):
        self.pipeline.set_state(Gst.State.NULL)


def main():
    import cv2
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--size', nargs=2, type=int, default=None, metavar=('HEIGHT', 'WIDTH'))
    parser.add_argument('--fps', type=int, default=None)
    parser.add_argument('--sync', type=int, default=1)

    args = parser.parse_args()

    cap = GstVideoCapture(device=args.device, size=args.size, fps=args.fps, sync=args.sync)
    cap.start()
    while True:
        try:
            image = cap.get()
            cv2.imshow("Image window", image)
            key = cv2.waitKey(100)
            key &= 255
            if key == 27 or key == ord('q'):
                print "Pressed ESC or q, exiting"
                break
        except KeyboardInterrupt:
            break
    cap.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

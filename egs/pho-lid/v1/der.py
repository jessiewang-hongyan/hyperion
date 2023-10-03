from pyannote.core import Segment, Timeline, Annotation
from pyannote.metrics.diarization import DiarizationErrorRate

class DER():
    def __init__(self):
        self.metric = DiarizationErrorRate()

    def get_seg_from_seq(self, lab_seq):
        annotation = Annotation()

        seg_start = 0
        seg_end = 0
        seg_lab = lab_seq[0]
        for lab in lab_seq:
            if lab == seg_lab:
                seg_end += 1
            else:
                # save previous segment
                segment = Segment(seg_start, seg_end + 1)
                annotation[segment] = seg_lab

                # init for new segment
                seg_start = seg_end + 1
                seg_lab = lab

        # store the last segment
        # if lab_seq[seg_start] == lab_seq[-1]:
        last_segment = Segment(seg_start, len(lab_seq))
        annotation[last_segment] = lab_seq[-1]

        return annotation

    def get_DER(self, truth_seq, pred_seq):
        reference = self.get_seg_from_seq(truth_seq)
        hypothesis = self.get_seg_from_seq(pred_seq)

        return self.metric(reference, hypothesis) 

    def get_global_DER(self):
        global_value = abs(self.metric)
        mean, (lower, upper) = self.metric.confidence_interval() 
        return global_value, mean, lower, upper

    def clear_DER():
        self.metric = DiarizationErrorRate()


if __name__ == "__main__":
    ref = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0]
    hyp = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0]

    print(get_seg_from_seq(ref))
    print(get_seg_from_seq(hyp))
    print(get_DER(ref, hyp))



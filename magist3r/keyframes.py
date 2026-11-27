def make_windows(num_frames, win_len, stride):
    span = win_len - 1
    if num_frames - 1 < span:
        return [list(range(num_frames))]
    windows, start = [], 0
    while start + span <= num_frames - 1:
        windows.append(list(range(start, start + win_len)))
        start += stride
    if windows[-1][-1] < num_frames - 1:
        s = num_frames - 1 - span
        windows.append(list(range(s, s + win_len)))
    return windows


class KeyframeDatabase:
    def __init__(self):
        self.entries = {}

    def __len__(self):
        return len(self.entries)

    def add_or_update(self, frame_id, entry, score):
        entry = dict(entry, score=float(score))
        if frame_id not in self.entries:
            self.entries[frame_id] = entry
            return
        old = self.entries[frame_id]
        if score <= old['score']:
            return
        for k in ('vggt_tokens', 'pose_tokens', 'salad_tokens'):
            entry[k] = old[k]
        self.entries[frame_id] = entry

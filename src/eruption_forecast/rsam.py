from obspy import Stream


class RSAM:
    def __init__(self, stream: Stream):
        self.stream = stream
        self.len_stream: int = len(self.stream)

    def calculate(self):
        stream: Stream = self.stream
        if self.len_stream > 1:
            stream = self.stream.merge()

import os
import re

from flask import Flask, Response, request

from cv import gen_video, PROJECT_DIR

app = Flask(__name__)


@app.after_request
def after_request(response):
    response.headers.add('Accept-Ranges', 'bytes')
    return response


def get_chunk(filename, byte1=None, byte2=None):
    filepath = os.path.join(PROJECT_DIR, filename)
    file_size = os.stat(filepath).st_size
    start = 0
    length = 102400

    if byte1 < file_size:
        start = byte1
    if byte2:
        length = byte2 + 1 - byte1
    else:
        length = file_size - start

    with open(filepath, 'rb') as f:
        f.seek(start)
        chunk = f.read(length)
    return chunk, start, length, file_size


@app.route('/video/<filename>')
def get_file(filename):
    range_header = request.headers.get('Range', None)
    byte1, byte2 = 0, None
    if range_header:
        match = re.search(r'(\d+)-(\d*)', range_header)
        groups = match.groups()

        if groups[0]:
            byte1 = int(groups[0])
        if groups[1]:
            byte2 = int(groups[1])

    chunk, start, length, file_size = get_chunk(filename, byte1, byte2)
    resp = Response(chunk, 206, mimetype='video/mp4',
                    content_type='video/mp4', direct_passthrough=True)
    resp.headers.add('Content-Range', 'bytes {0}-{1}/{2}'.format(start, start + length - 1, file_size))
    return resp


@app.route('/camera')
def get_video():
    try:
        return Response(gen_video(),
                        mimetype="multipart/x-mixed-replace; boundary=frame")
    except Exception as e:
        print(e)
        return "Internal server error", 500


if __name__ == '__main__':
    app.run()

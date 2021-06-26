# imports
import midi
import itertools
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

# this function is needed to test if the Note is effectively ON
# some Note On events could correspond to a velocity of 0, therefore not actually being "on"
def is_note_on(event):
    velocity = event.data[1]
    return event.name == "Note On" and velocity > 0

# check the maximum time in the track
def get_minmax_t(note_tracks):
    max_t = 0
    min_t = 9999
    for track in note_tracks:
        if any(track) == True:
            for notes in track:
                for note in notes:
                    if note> max_t:
                        max_t = note
                    elif note<min_t:
                        min_t = note
    return min_t,max_t

# convert MIDI file to piano-roll representation
def create_note_tracks(pattern, bpm, pixels_per_second):
    pattern.make_ticks_abs()
    resolution = pattern.resolution

    ticks_per_pixel = int((resolution * (bpm / 60)) / pixels_per_second)

    note_tracks = []
    for track_index, track in enumerate(pattern):
        notes = [[] for i in range(128)]  # create a list of lists for pitch classes

        for msg in track:
            if msg.name in ["Note On", "Note Off"]:
                if is_note_on(msg):
                    name = 'On'
                else:
                    name = 'Off'
                msg_ticks = msg.tick
                pitch = msg.get_pitch()
                notes[pitch].append(msg_ticks // ticks_per_pixel)

        note_tracks.append(notes)

    return note_tracks

# convert the piano-roll to image

def create_image(note_tracks, note_height):
    # get the min and max song duration (in pixels)
    min_t, max_t = get_minmax_t(note_tracks)

    color = (255)  # set to white - could vary with note velocity
    img_ydim = 128 * note_height  # image height is the number of pitch classes * height of each note
    img_xdim = max_t  # width is the song duration in seconds/how many seconds per pixel
    img = np.zeros((img_ydim, img_xdim), np.uint8)  # create initial black image from np array of 0s

    for track in note_tracks:  # note_tracks contains the notes for multiple tracks
        if any(track) == True:  # check whether the track is empty

            y = 0
            for notes in reversed(track):  # start with the highest pitch

                if notes:  # if the list is not empty
                    it = iter(notes)
                    coordinates = list(zip(it, it))

                    for coord in coordinates:

                        x1 = coord[0]
                        x2 = coord[1] - 2  # remove one pixel from the second coordinate

                        if x1 != x2:
                            c1 = (x1, y)  # top left corner
                            c2 = (x2, y)  # bottom right corner
                            cv.rectangle(img, c1, c2, color, -1)  # add a rectangle i.e note, -1 means FILLED

                y += note_height

    img = img[:, min_t:max_t]

    return img

# read the midi file
def parse_midifile(f):
    try:
        midi_pattern = midi.read_midifile(f)
        relevant_keys = ['Set Tempo', 'Time Signature']
        msg_dict = {key: [] for key in relevant_keys}
        for track in midi_pattern:
            for msg in track:
                key = msg.name
                msg_dict.setdefault(key, [])
                msg_dict[key].append(msg)

        tempo = [msg.get_bpm() for msg in msg_dict['Set Tempo']]
        tempo = max(tempo)

        time_sig = [(msg.get_numerator(), msg.get_denominator()) for msg in msg_dict['Time Signature']]
        time_sig = list(set(time_sig))

    except:
        midi_pattern = midi.Pattern()
        tempo = 0
        time_sig = [(0, 0)]

    return midi_pattern, tempo, time_sig

# function that combines other functions to convert midi file to images
def midi_2_image(root_dir, save_dir, dataset_name, note_height, pixels_per_second, track_duration):
    for root, dirs, files in os.walk(root_dir + dataset_name + '/'):
        for file in files:

            f = os.path.join(root, file)
            pattern, tempo, time_sig = parse_midifile(f)
            song_name = file[:-4]

            # make sure the time signature is (4,4)
            if len(time_sig) == 1 and time_sig[0] == (4, 4):
                note_tracks = create_note_tracks(pattern, tempo, pixels_per_second)
                img = create_image(note_tracks, note_height)

                crop = track_duration * pixels_per_second
                image = img[:, :crop]

                if image.shape[1] == crop:

                    save2path = save_dir + dataset_name + '/'
                    status = cv.imwrite('{}{}.png'.format(save2path, song_name), image)
                    print('{} saved successfully: {}'.format(song_name, status))
                else:
                    print('Track was too short')
import midi
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image

from sklearn.metrics.pairwise import euclidean_distances
import pandas as pd
from math import log
import pandas as pd
import numpy as np
import scipy.linalg
import scipy.stats

from scipy.stats import entropy


# custom note class
class Note(object):
    def __init__(
            self,
            velocity=60,
            pitch=0,
            start_time=0,  # time in pixels (time-steps)
            end_time=0,
            duration=0,
            pitch_class=None):
        self.velocity = velocity
        self.pitch = pitch
        self.start_time = start_time
        self.end_time = end_time
        self.duration = duration
        self.pitch_class = pitch_class

    def calculate_duration(self):
        self.duration = self.end_time - self.start_time + 1  # add 1 to compensate for subtracted 1 in midi2image

    def assign_pitch_class(self):

        pitch_classes = ['A', 'A#/Bb', 'B', 'C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab']
        pitch_class_array = np.array([21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32])
        pitch_class_matrix = pitch_class_array
        for i in range(8):  # number of octaves
            pitch_class_matrix = np.vstack((pitch_class_matrix, pitch_class_array + (i + 1) * 12))

        for i in range(12):  # number of pitch classes
            p = pitch_classes[i]
            if self.pitch in pitch_class_matrix.T[i]:
                self.pitch_class = p


# create the piano roll format for evaluation
def create_pianoroll(image):
    pitch_range = image.shape[0]

    piano_roll = {k: [] for k in range(pitch_range)}

    for p in range(pitch_range):

        pitch_track = image[p]
        actual_p = pitch_range - p - 1  # first row in image is the highest pitch
        note_track = []

        if any(pitch_track):

            # subtract by vector shifted by 1, find indices of when the result is 1 to find end/start times
            start_times = np.where(pitch_track - np.pad(pitch_track, 1)[:-2] == 1)[0]
            end_times = np.where(pitch_track - np.pad(pitch_track, 1)[2:] == 1)[0]
            n_notes = len(start_times)

            for i in range(n_notes):
                start = start_times[i]
                end = end_times[i]
                n = Note(velocity=60, pitch=actual_p)  # create note class
                n.start_time = start
                n.end_time = end
                n.calculate_duration()
                n.assign_pitch_class()
                note_track.append(n)

        piano_roll[actual_p] = note_track

    return piano_roll


def statistics(array):
    low = array.min()
    high = array.max()
    mean = array.mean()

    return low, high, mean


def extract_features_image(image):
    non_empty_timesteps = 0
    occupied_t = 0
    polyphony = 0

    for row in image.T:
        tot = np.sum(row)

        if tot > 0:
            non_empty_timesteps += 1
            occupied_t += tot
        if tot > 1:
            polyphony += 1

    empty_ratio = (image.shape[1] - non_empty_timesteps) / image.shape[1]
    occupation_rate = occupied_t / (image.shape[1])  # notes per time step
    polyphonic_rate = polyphony / non_empty_timesteps

    return [empty_ratio, occupation_rate, polyphonic_rate]


def extract_features_pianoroll(piano_roll):
    number_of_notes = 1
    qualified_notes = 1
    qualified_rhythm = 1

    durations = []
    pitches = []

    for key, val in piano_roll.items():

        number_of_notes += len(val)

        for n in val:

            d = n.duration
            durations.append(d)
            pitches.append(n.pitch)
            if d > 3:
                qualified_notes += 1

            if (d + 1) % 2 == 0:
                qualified_rhythm += 1

    QN = qualified_notes / number_of_notes
    QR = qualified_rhythm / number_of_notes

    min_d, max_d, mean_d = statistics(np.array(durations))
    min_p, max_p, mean_p = statistics(np.array(pitches))
    pitch_range = max_p - min_p

    return [QN, QR, mean_d, pitch_range]


def extract_features(image, pianoroll):
    A = extract_features_image(image)
    B = extract_features_pianoroll(pianoroll)
    feature_vector = A + B
    return feature_vector


def evaluation_metrics(img):
    pianoroll = create_pianoroll(img)
    signature_v = extract_features(img, pianoroll)
    max_pearson = Krumhansl_Schmuckler(pianoroll)[2][0]
    IR = get_information_rate(img)
    feature_vector = signature_v + [max_pearson] + [IR]
    return feature_vector


def get_evaluation_metrics(images):
    feature_matrix = []
    for img in images:
        if img.any():
            fv = evaluation_metrics(img)
            feature_matrix.append(fv)

    return np.array(feature_matrix)


def Krumhansl_Schmuckler(pianoroll):
    pitch_classes = np.asarray(['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B'])

    durations_dict = {p: [] for p in pitch_classes}  # create a dictionary for all pitch classes as keys

    for key, val in pianoroll.items():
        for note in val:
            # add the duration of each note to the relevant pitch class
            durations_dict.setdefault(note.pitch_class, []).append(note.duration)

    pitch_distribution = np.array([sum(v) for k, v in durations_dict.items()])

    # profiles:
    major = np.asarray([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    minor = np.asarray([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

    scores_dict = {}

    for i in range(12):  # for each pitch class:

        current_pitch = pitch_classes[i]
        temp = np.roll(pitch_distribution, -i)  # shift the distribution to start with the current pitch

        R = scipy.stats.pearsonr(major, temp)[0]  # obtain pearson R for each of the profiles
        Rm = scipy.stats.pearsonr(minor, temp)[0]

        scores_dict[current_pitch + '_major'] = R
        scores_dict[current_pitch + '_minor'] = Rm

    ranked_scores_keys = [k for k, v in sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)]
    ranked_scores_values = [v for k, v in sorted(scores_dict.items(), key=lambda item: item[1], reverse=True)]

    # return(pitch_distribution)
    return scores_dict, ranked_scores_keys, ranked_scores_values


def find_longest_pause(image):
    a = np.sum(image, axis=0)
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(a, 0).view(np.int8), [0]))
    if any(iszero):
        edges = np.diff(iszero)
        start = np.where(edges == 1)[0]
        end = np.where(edges == -1)[0]
        pauses = end - start
        max_pause = max(pauses)
    else:
        max_pause = 0

    return max_pause


def reshape(img, n):
    y = img.shape[1]
    div = y // n
    inds = [div * i for i in range(1, n)]

    splits = np.hsplit(img, inds)

    reshaped_img = splits[0]
    for i in range(1, len(splits)):
        reshaped_img = np.vstack((reshaped_img, splits[i]))

    return reshaped_img


def reshape_rgb(img):
    y = img.shape[1]
    div = y // 3
    print(div)
    inds = [div * i for i in range(1, 3)]

    splits = np.hsplit(img, inds)
    channels = [x for x in splits]

    rgb_img = np.array(channels)
    rgb_img = np.transpose(rgb_img, (1, 2, 0))
    return rgb_img


def get_pitch_dict():
    # Get the pitch class dictionary (equate a pitch class to the corresponding midi note number)
    MIDI_note = 12  # index 12 = C0
    pitch_classes = ['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B']
    pitch_dict = {}
    for i in range(10):  # 9 octaves

        for p in pitch_classes:
            pitch_dict[MIDI_note] = p + str(i)
            MIDI_note += 1
            if MIDI_note == 128:
                break
    return pitch_dict


def extract_sequence(image):
    pitch_dict = get_pitch_dict()
    sequence = ['START']
    for row in image.T:
        if any(row):
            event = ''
            for n in np.where(row > 0)[0]:
                event += str(pitch_dict[128 - n - 1]) + '-'

            sequence.append(event[:-1])
        else:
            sequence.append('REST')

    return sequence


def calculate_IR(sequence):
    headings = sorted(list(set(sequence)), key=len)
    zero_matrix = np.zeros((len(headings), len(headings)))
    note_counts = pd.DataFrame(np.zeros(len(headings)), headings)
    transition_matrix = pd.DataFrame(zero_matrix, headings, headings)

    information_rates = []

    for i in range(1, len(sequence)):
        current_note = sequence[i]  # note at current timestep
        prev_note = sequence[i - 1]  # note at previous timestep

        if i == 1:  # start of sequence
            p_notes = np.array([1 / len(headings) for i in range(len(headings))])  # maximum uncertainty
            p_prev_note = 1 / len(headings)
        else:
            p_prev_note = note_counts.loc[prev_note][0] / (i - 1)
            p_notes = note_counts / (i - 1)  # [0]

        if transition_matrix.loc[prev_note].sum() == 0:
            p_transition = np.array([1 / len(headings) for i in range(len(headings))])  # maximum uncertainty
        else:
            p_transition = transition_matrix.loc[prev_note] / transition_matrix.loc[prev_note].sum()

        # calculate the entropy & IR
        notes_H = entropy(p_notes, base=2)
        transition_H = entropy(p_transition, base=2)  # *p_prev_note

        IR = max((notes_H - transition_H), 0)
        information_rates.append(IR)

        # update the counts
        note_counts.loc[current_note] += 1  # add 1 for the note at current timestep
        transition_matrix.loc[prev_note, current_note] += 1

    total_IR = sum(information_rates) / len(information_rates)

    return total_IR[0]


def get_information_rate(image):
    seq = extract_sequence(image)
    IR = calculate_IR(seq)
    return IR


def generate_random_sequence(seq_length):
    values = np.asarray(['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B', 'REST'])
    sequence = np.array(['START'])
    random_seq = np.array([values[np.random.randint(0, len(values))] for i in range(seq_length)])

    sequence = np.concatenate((sequence, random_seq))
    return sequence


def generate_repetitive_sequence(seq_length, rep_length):
    values = np.asarray(['C', 'C#/Db', 'D', 'D#/Eb', 'E', 'F', 'F#/Gb', 'G', 'G#/Ab', 'A', 'A#/Bb', 'B', 'REST'])
    sequence = np.array(['START'])
    random_seq = np.array([values[np.random.randint(0, len(values))] for i in range(rep_length)])
    for i in range(int(seq_length // rep_length)):
        sequence = np.concatenate((sequence, random_seq))

    return sequence


# for visualisation in report

def generate_random_image(height, width):  # CROPPED height (within 80 range)
    k = np.random.randint(400, 600)
    arr = np.zeros(height * width)
    arr[:k] = 1
    np.random.shuffle(arr)
    arr = np.reshape(arr, (height, width))
    rand_image = np.vstack((np.zeros(((128 - height) // 2, width)), arr, np.zeros(((128 - height) // 2, width))))
    return rand_image


def generate_repetitive_image(height, width, r):
    k = int((np.random.randint(1600, 3000) / width) * r)
    arr = np.zeros(height * r)
    arr[:k] = 1
    np.random.shuffle(arr)
    arr = np.reshape(arr, (height, r))
    repetitive_image_section = np.vstack((np.zeros(((128 - height) // 2, r)), arr, np.zeros(((128 - height) // 2, r))))
    plt.imshow(repetitive_image_section)
    repetitive_image = np.tile(repetitive_image_section, width // r)
    return repetitive_image


def get_distances_hist(matrix1, matrix2):
    n_features = matrix1.shape[1]
    histogram_matrix = []
    for i in range(n_features):
        print('doing feature {}'.format(i))
        row1 = matrix1.T[i].reshape(-1, 1)  # transposed to compare all the samples of m1 to all of m2 for any feature
        # row1 = row1.astype('float64')
        row2 = matrix2.T[i].reshape(-1, 1)  # features = ROWS
        # row2 = row2.astype('float64')

        distances_matrix = euclidean_distances(row1, row2)  # n by n cross-validation matrix

        # if the dimensions are not even, keep lower triangle when there are more rows than columns
        if distances_matrix.shape[0] > distances_matrix.shape[1]:
            tri = np.tril(distances_matrix)
        else:
            tri = np.triu(distances_matrix)

        distances = np.reshape(tri, -1)  # flatten array
        distances = distances[distances != 0]  # remove zero values
        histogram_matrix.append(distances)

    return histogram_matrix


def get_pdf_per_feature(histograms):
    pdf_per_feature = []

    for hist in histograms:
        hist_range = abs(max(hist) - min(hist))
        kde = kernel_density_estimation(hist, 1000, min(hist) - hist_range / 2, max(hist) + hist_range / 2)
        pdf_per_feature.append(kde)

    return pdf_per_feature


def Scott_bandwidth(X):
    n = len(X)
    h = 1.06 * abs(np.std(X)) * n ** (-1 / 5)
    h = max(0.1, h)

    return h


def kernel_density_estimation(X, N, lower_r, upper_r):
    X = X.reshape(-1, 1)

    # upper_r = -1
    # lower_r = 1

    X_plot = np.linspace(lower_r, upper_r, N)[:, np.newaxis]

    kde = KernelDensity(kernel='gaussian', bandwidth=Scott_bandwidth(X)).fit(X)
    log_density = kde.score_samples(X_plot)
    pdf = np.exp(log_density)
    return [pdf, X_plot]


def get_KL_divergence(sample1, sample2, N, plot):  # samples are HIST

    min1 = min(sample1)
    min2 = min(sample2)
    max1 = max(sample1)
    max2 = max(sample2)

    lower = min(min1, min2)
    upper = max(max1, max2)
    tot_range = abs(upper - lower)
    lower_bound = lower - tot_range * 0.2
    upper_bound = upper + tot_range * 0.2

    kde1 = kernel_density_estimation(sample1, N, lower_bound, upper_bound)
    kde2 = kernel_density_estimation(sample2, N, lower_bound, upper_bound)

    kl = entropy(kde1[0], kde2[0])
    if plot == True:
        plt.plot(kde1[1], kde1[0])
        plt.plot(kde2[1], kde2[0])
        plt.show()
    return kl


def get_KL_divergence_per_feature(h1, h2, plot):
    KL_per_feature = []

    for i in range(len(h1)):
        min_len = min(len(h1[i]), len(h2[i])) - 1
        KL = get_KL_divergence(h1[i][:min_len], h2[i][:min_len], 1000, plot)
        KL_per_feature.append(KL)

    return KL_per_feature
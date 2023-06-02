# 1. Install latest Python
# 2. Write in command prompt: pip install matplotlib,scipy
# 3. First launch client.py and press button 1 or 2 and crash 33 (this interaction create cyclic dep of server)
# 4. Now launch server.py and optionally close (first launch creates cyclic dep of client)
# 5. Now launch client.py and use normally. From now on client and server are launchable and closable in ANY order,
# ANYTIME, run only one instance of each though.
# 5a. button 1 is for converting .mat to .txt, then moving it in root and listening to serverout.txt regen'd by server
# 5b. button 2 is just moving an already ASCII .txt to root and otherwise same
# 5c. button 3 is just listening to serverout.txt
import numpy  # inner graph workings logic and more
import tkinter  # for graph creation logic and more
import matplotlib.pyplot as plt  # later for graph creation logic
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # later for graph creation logic
from tkinter import filedialog  # for buttons 1 and 2
import os  # later for buttons 1, 2 and 3
import scipy.io as sio  # later for button 1
from scipy.ndimage import gaussian_filter  # for button 3
from matplotlib.animation import FuncAnimation  # later for button 3

# slider consts and graph materialisation consts
SIGMA0 = 32.0
SDX_SKEW0 = 2.3
SENSIT0 = 0.8
PIDW0 = 1.5  # s

ROW_RATE = 256  # rows / s
LATENCY = 0.5  # s
CACHE_SIZE = 10  # s

# app aesthetics, buttons aesthetic, sliders aesthetic
APP_NAME = "IMPO App v0.8"
DEFAULT_FONT = "TkDefaultFont"
DEFAULT_FONT_SIZE = 18
SPINBOX_START = 0.05
SPINBOX_END = 500.0
SPINBOX_WIDTH = 7


########################################################################################################################
# inner graph workings logic
# processed globals are better kept within sane range
def sfloat(s):
    try:
        if not (SPINBOX_START <= float(s) <= SPINBOX_END):
            return 1.0
        else:
            return float(s)
    except ValueError:
        return 1.0


# multiplies smoothened differentiated x ndarray so that equivalent eye movement qualifies same absolute value
def skew(arr):
    global sdx_skew
    new_arr = arr * sfloat(sdx_skew.get())
    return new_arr


# absolute signal value below noise threshold or dominated by ref signal is muted, else its polarity is noted
def alt_sign(arr, arr_ref):
    global sensit
    new_arr = numpy.empty(0)
    for val, val_ref in zip(arr, arr_ref):
        if abs(val) < sfloat(sensit.get()) or abs(val) < abs(val_ref):
            new_val = 0
        else:
            new_val = numpy.sign(val)
        new_arr = numpy.append(new_arr, new_val)
    return new_arr


# for intent of telling apart directions peaks for pattern identifier, same polarity is squashed in ndarray
def squash_same_polarity(arr):
    new_arr = numpy.empty(0)
    if len(arr) == 0:
        return new_arr
    start = 0
    while arr[start] == 0:
        start += 1
    new_arr = numpy.append(new_arr, numpy.sign(arr[start]))
    for arr_it in range(start + 1, len(arr)):
        if numpy.sign(new_arr[-1]) == -numpy.sign(arr[arr_it]):
            new_arr = numpy.append(new_arr, numpy.sign(arr[arr_it]))
    return new_arr


# for intent of pattern identifier, count up unique pairings of opposing poles
def count_polar_pairs(arr):
    pos_cnt = numpy.sum(arr == 1)
    neg_cnt = numpy.sum(arr == -1)
    return min(pos_cnt, neg_cnt)


# identifies gestures by priority TBL > DBL > stronger direction signal > inaction (-3, -2, 1234, 0)
def pattern_id(arr_x, arr_y):
    global pidw
    samples = int(sfloat(pidw.get()) * ROW_RATE)

    new_arr = numpy.zeros(len(arr_x))
    arr_it = len(arr_x)
    while arr_it >= 0:
        mv_arr_it = max(0, arr_it - samples)
        xslice = arr_x[mv_arr_it:arr_it]
        yslice = arr_y[mv_arr_it:arr_it]
        bl_count = count_polar_pairs(xslice)

        if bl_count >= 3:
            new_arr[mv_arr_it:arr_it] = -3
        elif bl_count == 2:
            new_arr[mv_arr_it:arr_it] = -2
        elif bl_count == 1:
            pass
        else:  # +-x and +-y logic
            if -1 in xslice:
                new_arr[mv_arr_it:arr_it] = 1
            if 1 in xslice:
                new_arr[mv_arr_it:arr_it] = 2
            if -1 in yslice:
                new_arr[mv_arr_it:arr_it] = 3
            if 1 in yslice:
                new_arr[mv_arr_it:arr_it] = 4
        arr_it -= samples
    return new_arr


########################################################################################################################
# graph creation logic
def start_graph(root_frame):
    graph_frame = tkinter.Frame(root_frame)
    graph_frame.grid(column=0, row=1, columnspan=6)
    func_fig = plt.figure(figsize=(8, 8))
    func_fig.subplots_adjust(top=0.97, bottom=0.03, left=0.07, right=1, hspace=0.4)

    canvas = FigureCanvasTkAgg(func_fig, master=graph_frame)
    canvas.get_tk_widget().pack()

    func_ax = func_fig.subplots(5, 2)
    func_X = numpy.linspace(-CACHE_SIZE, 0, CACHE_SIZE * ROW_RATE)
    func_ax[0, 0].set_title("x")
    func_ax[0, 1].set_title("y")
    func_ax[1, 0].set_title("differentiate->x")
    func_ax[1, 1].set_title("differentiate->y")
    func_ax[2, 0].set_title("skew->smoothen->dx")
    func_ax[2, 1].set_title("smoothen->dy")
    func_ax[3, 0].set_title("alt_sign->ssdx")
    func_ax[3, 1].set_title("alt_sign->sdy")
    func_ax[4, 0].set_title("pattern_id->assdx,asdy")
    func_ax[4, 0].set_yticks([-3, -2, 0, 1, 2, 3, 4])
    func_ax[4, 0].set_yticklabels(["TBL", "DBL", "NOACT", "LEFT", "RIGHT", "DOWN", "UP"])
    func_ax[4, 0].set_ylim([-3 * 1.1, 4 * 1.1])
    func_ax[4, 1].set_visible(False)

    func_Y = [None] * 9
    for i in range(2):
        func_Y[i] = numpy.zeros(CACHE_SIZE * ROW_RATE)
    for i in range(7):
        func_Y[i + 2] = numpy.zeros(CACHE_SIZE * ROW_RATE - 1)

    func_line = [None] * 9
    func_line[0], = func_ax[0, 0].plot(func_X, func_Y[0])
    func_line[1], = func_ax[0, 1].plot(func_X, func_Y[1])
    func_X = func_X[:-1]
    func_line[2], = func_ax[1, 0].plot(func_X, func_Y[2])
    func_line[3], = func_ax[1, 1].plot(func_X, func_Y[3])
    func_line[4], = func_ax[2, 0].plot(func_X, func_Y[4])
    func_line[5], = func_ax[2, 1].plot(func_X, func_Y[5])
    func_line[6], = func_ax[3, 0].plot(func_X, func_Y[6])
    func_line[7], = func_ax[3, 1].plot(func_X, func_Y[7])
    func_line[8], = func_ax[4, 0].plot(func_X, func_Y[8])
    return func_fig, func_ax, func_line, func_Y


# 1st button logic
def on_button1_click():
    file_path = filedialog.askopenfilename(filetypes=[('MAT files', '*.mat')])
    if file_path:
        file_name = os.path.basename(file_path)
        button1.config(text=f'gotten {file_name}&listening')
        button1.config(state=tkinter.DISABLED)
        button2.config(state=tkinter.DISABLED)

        mat_contents = sio.loadmat(file_path)
        reading = mat_contents['Biotrace'][2:]
        py_reading = numpy.transpose(reading)
        numpy.savetxt("serverin.txt", py_reading, delimiter='\t', fmt='%.10e')
        button3.invoke()


# 2nd button logic
def on_button2_click():
    file_path = filedialog.askopenfilename(filetypes=[('Text files', '*.txt')])
    if file_path:
        file_name = os.path.basename(file_path)
        button2.config(text=f'gotten {file_name}&listening')
        button1.config(state=tkinter.DISABLED)
        button2.config(state=tkinter.DISABLED)

        txt_contents = numpy.loadtxt(file_path)
        numpy.savetxt("serverin.txt", txt_contents, delimiter='\t', fmt='%.10e')
        button3.invoke()


# 3rd button logic
def on_button3_click(func_fig, func_ax, func_line, func_Y):
    button3.config(text='listening')
    button1.config(state=tkinter.DISABLED)
    button2.config(state=tkinter.DISABLED)
    button3.config(state=tkinter.DISABLED)

    def update_graph(frame):
        global sigma, total_fr

        func_fig.canvas.draw()
        file_name = "serverout.txt"
        file_lock_name = "lock"
        with open(file_lock_name, "w"):  # lock up serverout.txt for server.py's changes of it
            pass
        try:
            new_Y = numpy.genfromtxt(file_name, delimiter='\t')  # try generating from it, it may not even exist
        except FileNotFoundError:
            os.remove(file_lock_name)  # safely delete lock
            exit(33)
        os.remove(file_lock_name)  # delete lock when done properly using it
        new_Y = new_Y.T

        func_Y[0] = func_Y[0][len(new_Y[0]):]
        func_Y[1] = func_Y[1][len(new_Y[1]):]
        func_Y[0] = numpy.concatenate((func_Y[0], new_Y[0]))
        func_Y[1] = numpy.concatenate((func_Y[1], new_Y[1]))
        func_Y[2] = numpy.diff(func_Y[0])
        func_Y[3] = numpy.diff(func_Y[1])
        func_Y[4] = skew(gaussian_filter(func_Y[2], sigma=sfloat(sigma.get())))
        func_Y[5] = gaussian_filter(func_Y[3], sigma=sfloat(sigma.get()))
        func_Y[6] = alt_sign(func_Y[4], func_Y[5])
        func_Y[7] = alt_sign(func_Y[5], func_Y[4])
        func_Y[8] = pattern_id(func_Y[6], func_Y[7])

        for i in range(8):
            func_line[i].set_ydata(func_Y[i])
        lat_ratio = int(sfloat(pidw.get()) / LATENCY)
        if lat_ratio == 0:
            lat_ratio = 1
        if total_fr % lat_ratio == 0:
            func_line[8].set_ydata(func_Y[8])
            parse_instr(func_Y[8][-1])
        total_fr += 1
        for i in range(4):
            for j in range(2):
                func_ax[i, j].relim()
                func_ax[i, j].autoscale_view()
        return func_line

    ani = FuncAnimation(func_fig, update_graph, frames=None, interval=int(LATENCY * 1000), blit=True,
                        cache_frame_data=False)
    func_fig.ani = ani


########################################################################################################################
# app starts now
shown_frame = tkinter.Tk()
shown_frame.title(APP_NAME)
# create graph
total_fr = 0
fig, ax, line, Y = start_graph(shown_frame)

# create buttons
button1 = tkinter.Button(shown_frame, text='.mat->ASCII&mv&listen', command=on_button1_click,
                         font=(DEFAULT_FONT, DEFAULT_FONT_SIZE))
button2 = tkinter.Button(shown_frame, text='mvASCII&listen', command=on_button2_click,
                         font=(DEFAULT_FONT, DEFAULT_FONT_SIZE))
button3 = tkinter.Button(shown_frame, text='listen', command=lambda: on_button3_click(fig, ax, line, Y),
                         font=(DEFAULT_FONT, DEFAULT_FONT_SIZE))
button1.grid(row=0, column=0, columnspan=2)
button2.grid(row=0, column=2, columnspan=2)
button3.grid(row=0, column=4, columnspan=2)

# create spinbox labels and spinboxes
sigma_label = tkinter.Label(shown_frame, text='sigma')
sdx_skew_label = tkinter.Label(shown_frame, text='sdx_skew')
sensit_label = tkinter.Label(shown_frame, text='sensit (µV/sample)')
pidw_label = tkinter.Label(shown_frame, text='pidw (s)')
sigma = tkinter.Spinbox(shown_frame, from_=SPINBOX_START, to=SPINBOX_END, increment=SPINBOX_START, format="%.2f",
                        textvariable=tkinter.DoubleVar(value=SIGMA0),
                        font=(DEFAULT_FONT, DEFAULT_FONT_SIZE), width=SPINBOX_WIDTH)
sdx_skew = tkinter.Spinbox(shown_frame, from_=SPINBOX_START, to=SPINBOX_END, increment=SPINBOX_START, format="%.2f",
                           textvariable=tkinter.DoubleVar(value=SDX_SKEW0),
                           font=(DEFAULT_FONT, DEFAULT_FONT_SIZE), width=SPINBOX_WIDTH)
sensit = tkinter.Spinbox(shown_frame, from_=SPINBOX_START, to=SPINBOX_END, increment=SPINBOX_START, format="%.2f",
                         textvariable=tkinter.DoubleVar(value=SENSIT0),
                         font=(DEFAULT_FONT, DEFAULT_FONT_SIZE), width=SPINBOX_WIDTH)
pidw = tkinter.Spinbox(shown_frame, from_=SPINBOX_START, to=SPINBOX_END, increment=SPINBOX_START, format="%.2f",
                       textvariable=tkinter.DoubleVar(value=PIDW0),
                       font=(DEFAULT_FONT, DEFAULT_FONT_SIZE), width=SPINBOX_WIDTH)
sigma_label.grid(row=2, column=0)
sdx_skew_label.grid(row=2, column=1)
sensit_label.grid(row=2, column=2)
pidw_label.grid(row=2, column=5)
sigma.grid(row=3, column=0)
sdx_skew.grid(row=3, column=1)
sensit.grid(row=3, column=2)
pidw.grid(row=3, column=5)

# create keyboard
keys = numpy.array([
    ["ζζζ", "ζζζ", "zZ'", ",;=", ".:?", "βββ", "βββ"],
    ["ζζζ", "aA1", "bB2", "cC3", "dD4", "eE5", "βββ"],
    ["ααα", "fF6", "gG7", "hH8", "iI9", "jJ0", "λλλ"],
    ["ααα", "kK*", "lL+", "mM#", "nN-", "oO_", "λλλ"],
    ["ααα", "pP(", "qQ)", "rR&", "sS!", "tT£", "λλλ"],
    ["σσσ", "uU$", "vV€", "wW/", "xX\\", "yY”", "^^^"],
    ["σσσ", "σσσ", "  @", "  @", "<<<", "↓↓↓", ">>>"]
])
mode = 0
curs = [3, 3]


def update_captions():
    for i in range(7):
        for j in range(7):
            buttons[i][j].config(text=keys[i][j][mode], bg='SystemButtonFace')
            if i == curs[0] and j == curs[1]:
                buttons[i][j].config(bg='red')


def parse_letter(gotten):
    global mode
    curr_row, curr_col = text_box.index(tkinter.INSERT).split('.')
    curr_row, curr_col = int(curr_row), int(curr_col)
    if gotten not in 'λβζασ<↓>^':
        text_box.insert(tkinter.INSERT, gotten)
    if gotten == 'λ':
        text_box.insert(tkinter.INSERT, '\n')
    if gotten == 'β':
        text = text_box.get(1.0, tkinter.END)
        text_box.delete(1.0, tkinter.END)
        text_box.insert(tkinter.END, text[:-2])  # TODO: instead of deleting last letter, delete last letter before ][
    if gotten == 'ζ':
        mode = 0
    if gotten == 'α':
        mode = 2
    if gotten == 'σ':
        mode = 1
    if gotten == '<':
        text_box.mark_set(tkinter.INSERT, f'{curr_row}.{curr_col - 1}')
    if gotten == '↓':
        text_box.mark_set(tkinter.INSERT, f'{curr_row + 1}.{curr_col}')
    if gotten == '>':
        text_box.mark_set(tkinter.INSERT, f'{curr_row}.{curr_col + 1}')
    if gotten == '^':
        text_box.mark_set(tkinter.INSERT, f'{curr_row - 1}.{curr_col}')
    update_captions()
    text_box.see(tkinter.INSERT)


entire_right = tkinter.Frame(shown_frame)
entire_right.grid(row=0, column=6, rowspan=4)
buttons = []
for _ in range(7):
    btn_row = []
    for __ in range(7):
        btn = tkinter.Button(entire_right, command=lambda i=_, j=__: parse_letter(keys[i][j][mode]),
                             font=(DEFAULT_FONT, DEFAULT_FONT_SIZE * 2), width=3)
        btn.grid(row=_, column=__)
        btn_row.append(btn)
    buttons.append(btn_row)
update_captions()


def parse_instr(instr):
    if instr == -3:
        parse_letter('β')
    if instr == -2:
        buttons[curs[0]][curs[1]].invoke()
    if instr == 1 and curs[1] > 0:
        curs[1] -= 1
    if instr == 2 and curs[1] < 6:
        curs[1] += 1
    if instr == 3 and curs[0] < 6:
        curs[0] += 1
    if instr == 4 and curs[0] > 0:
        curs[0] -= 1
    update_captions()
    return instr


text_box = tkinter.Text(entire_right, height=4, width=24, font=(DEFAULT_FONT, DEFAULT_FONT_SIZE * 2))
text_box.focus_set()
text_box.grid(row=7, column=0, columnspan=7)
shown_frame.mainloop()

# How to get IMPO App running?
# 1. Install latest Python
# 2. Write in command prompt: pip install scipy, after that pip install numpy
# 3. First launch client.py and press button 1 or 2 with prepared .mat or .txt substrate
# 4. Env is set up, button 3 is just for reading the old substrate, 1 and 2 are for replacing substrate
import tkinter
from tkinter import filedialog
import turtle
import os
import time
import scipy
import numpy as np

# graph math consts and pg compress factor
SIGMA0 = 32.0
GDX_GAIN0 = 1.3
THRESHOLD0 = 0.8 # µV/sample
PIDM0 = 3
PG_COMPR0 = 16
PG_USEAVERAGING = False

# communication consts
ROWS_PER_SECOND = 256
SECONDS_PER_PACKET = 0.5
ROWS_PER_PACKET = int(ROWS_PER_SECOND * SECONDS_PER_PACKET)
SECONDS_PER_BUFFER = 10
ROWS_PER_BUFFER = int(ROWS_PER_SECOND * SECONDS_PER_BUFFER)

# app name, common font, spinbox and turtle consts
APP_OBJECT_NAME = 'IMPO App v0.976'
COMMON_FONT = 'TkDefaultFont'
COMMON_FONT_BASE_SIZE = 20
SPINBOX_MIN_VAL = 0.01
SPINBOX_MAX_VAL = 500
SPINBOX_INCR_VAL = 0.1
SPINBOX_WIDTH = 10
SPINBOX_FALLBACK_VAL = 1
TURTLE_WIDTH = 1300
TURTLE_HEIGHT = 890
TURTLE_PAD = 30
GRAPH_CALLSIGNS = [
    ['x','y'],['diffentiate->x','differentiate->y'],['amplify->gaussify->dx','gaussify->dy'],
    ['keep ab. th.->agdx','keep ab. th.->gdy'],['strong sign->kagdx','strong sign->kgdy'],['pattern id.->skagdx,skgdy,kagdx','unused']
    ]
GRAPH_COLS = len(GRAPH_CALLSIGNS[0])
GRAPH_ROWS = len(GRAPH_CALLSIGNS)
GRAPH_WIDTH = TURTLE_WIDTH / GRAPH_COLS - TURTLE_PAD
GRAPH_HEIGHT = TURTLE_HEIGHT / GRAPH_ROWS - TURTLE_PAD
UNUSED_GRAPHS = [[5,1]]
YSECTIONS = 7
STATIC_TICK_GRAPHS = {(4,0): [-1,1,3], (4,1): [-1,1,3], (5,0): [-3,4,8]}

####################################################################################################
# first half of screen aspect 1/3: graph maths
# in: entered string Q number in either spinbox
# out: if parseable and within range converted float Q, if not fallback float Q
def sfloat(s):
    try:
        f = float(s)
        if SPINBOX_MIN_VAL <= f <= SPINBOX_MAX_VAL:
            return f
        else:
            return SPINBOX_FALLBACK_VAL
    except ValueError:
        return SPINBOX_FALLBACK_VAL
# in: np array of float Q
# out: np array smoothened using method Gaussian blur
# why: gain information about overall tendency
def gaussify(arr):
    return scipy.ndimage.gaussian_filter(arr, sigma=sfloat(sigma_spinbox.get()))
# in: np array of float Q
# out: np array multiplied by scalar
# why: equivalent eye mv must qualify same abs value
def amplify(arr):
    return arr * sfloat(gdx_gain_spinbox.get())
# in: np array of float Q
# out: np array of 0's where sig is noise and sig where it isn't
# why: get rid of noise
def keep_above_threshold(arr):
    return np.where(np.abs(arr) < sfloat(threshold_spinbox.get()), 0, arr)
# in: np array of float Q
# out: np array of 0's where sig isn't dominant x vs. y or y vs. x
# and -1's=l/d and +1's=r/u in respective array where that's dominant signal
# why: limit movement to 90 degree grid, pick strongest signal
def strong_sign(arr, arr_ref):
    return np.where(np.abs(arr) < np.abs(arr_ref), 0, np.sign(arr))
# in: np array of float {-1, 0, 1}
# out: np array of float {-1, 0, 1}, but the consecutives are merged to single
# why: atomise trend on the graph
def squash_same_polarity(arr):
    new_arr = np.empty(len(arr))
    if len(arr) == 0:
        return new_arr
    new_arr[0] = arr[0]
    cnt = 1
    for i in range(1, len(arr)):
        if np.sign(arr[i - 1]) != np.sign(arr[i]):
            new_arr[cnt] = np.sign(arr[i])
            cnt += 1
    new_arr.resize(cnt)
    return new_arr
# in: 2 np arrays of float {-1, 0, 1}, 1 np array of float Q
# out: np array of courses of action var 'samples' wide on the whole buffer
# float {-3, -2, 0, 1, 2, 3, 4}
def count_polar_pairs(arr):
    pos_cnt = np.sum(arr == 1)
    neg_cnt = np.sum(arr == -1)
    return min(pos_cnt, neg_cnt)
def pid(arr_x, arr_y, arr_bl):
    new_arr = np.zeros(len(arr_x))
    samples = max(1,int(sfloat(pidm_spinbox.get())))*ROWS_PER_PACKET
    arr_it = len(arr_x)
    while arr_it >= 0:
        mv_arr_it = max(0, arr_it - samples)
        xslice = arr_x[mv_arr_it:arr_it]
        yslice = arr_y[mv_arr_it:arr_it]
        bl_count = count_polar_pairs(squash_same_polarity(np.sign(arr_bl[mv_arr_it:arr_it])))

        if bl_count >= 3:
            new_arr[mv_arr_it:arr_it] = -3
        elif bl_count == 2:
            new_arr[mv_arr_it:arr_it] = -2
        elif bl_count == 1:
            pass
        else:
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
####################################################################################################
# first half of screen aspect 2/3: graph materialisation
def get_t_home(i, j):
    x = -TURTLE_WIDTH/2 + j * (GRAPH_WIDTH + TURTLE_PAD)
    y = TURTLE_HEIGHT/2 - (i+1) * (GRAPH_HEIGHT + TURTLE_PAD) + TURTLE_PAD*4/5
    return x, y
def t_line(t, x1, y1, x2, y2):
    t.setpos(x1, y1)
    t.pendown()
    t.setpos(x2, y2)
    t.penup()
def t_axes(t, x, y, capt):
    t_line(t, x, y, x + GRAPH_WIDTH, y)
    t_line(t, x + GRAPH_WIDTH, y, x + GRAPH_WIDTH, y + GRAPH_HEIGHT)
    t.setpos(x + GRAPH_WIDTH/2, y + GRAPH_HEIGHT - TURTLE_PAD/3)
    t.write(capt)
def t_static_tick(t, x, y, is_xtick, capt):
    if is_xtick:
        t_line(t, x, y - TURTLE_PAD/3, x, y + TURTLE_PAD/3)
        t.setpos(x, y - 2*TURTLE_PAD/3)
        t.write(capt)
    else:    
        t_line(t, x - TURTLE_PAD/3, y, x + TURTLE_PAD/3, y)
        #t.setpos(x - GRAPH_WIDTH/SECONDS_PER_BUFFER, y - TURTLE_PAD/3)
        #t.write(capt)
def t_awt(t, i, j, capt):
    home_x, home_y = get_t_home(i, j)
    t_axes(t, home_x, home_y, capt)
    xit, yit = 0, 0
    for xit in range(SECONDS_PER_BUFFER + 1):
        t_static_tick(t, home_x + GRAPH_WIDTH * xit/SECONDS_PER_BUFFER, home_y, True, xit-SECONDS_PER_BUFFER)
    for yit in range(YSECTIONS + 1):
        t_static_tick(t, home_x + GRAPH_WIDTH, home_y + GRAPH_HEIGHT * yit/YSECTIONS, False, 'unused')
def t_awts(t):
    for i in range(GRAPH_ROWS):
        for j in range(GRAPH_COLS):
            if [i,j] in UNUSED_GRAPHS:
                continue
            t_awt(t, i, j, GRAPH_CALLSIGNS[i][j])
def map_data(i, j, data, dit):
    home_x, home_y = get_t_home(i, j)
    xbounds1, xbounds2 = [0, len(data) - 1], [home_x, home_x + GRAPH_WIDTH]
    if (i,j) in STATIC_TICK_GRAPHS:
        ybounds1 = [STATIC_TICK_GRAPHS[(i,j)][0], STATIC_TICK_GRAPHS[(i,j)][1]]
    else:
        ybounds1 = [min(data), max(data)]
    ybounds2 = [home_y, home_y + GRAPH_HEIGHT]
    x = np.interp(dit, xbounds1, xbounds2)
    y = np.interp(data[dit], ybounds1, ybounds2)
    return x, y
def t_pop_graph(t, i, j, data):
    pc = max(4,int(sfloat(pg_compr_spinbox.get())))
    if PG_USEAVERAGING:
        while len(data)%pc != 0:
            data = data[:-1]
        data = np.mean(data.reshape(-1, pc), axis = 1)
    else:
        data = data[::pc]
    for dit in range(len(data) - 1):
        x1, y1 = map_data(i, j, data, dit)
        x2, y2 = map_data(i, j, data, dit + 1)
        t_line(t, x1, y1, x2, y2)
    #optimisations won't be needed in pygame
def t_pop_graphs(t, d2da):
    for i in range(GRAPH_ROWS):
        for j in range(GRAPH_COLS):
            if [i,j] in UNUSED_GRAPHS:
                continue
            t_pop_graph(t, i, j, d2da[i, j])
def t_pop_ytick(t, i, j, data):
    home_x, home_y = get_t_home(i, j)
    if (i,j) in STATIC_TICK_GRAPHS:
        yticksinfo = np.linspace(STATIC_TICK_GRAPHS[(i,j)][0], STATIC_TICK_GRAPHS[(i,j)][1], STATIC_TICK_GRAPHS[(i,j)][2])
    else:
        yticksinfo = np.percentile([min(data), max(data)], np.linspace(0, 100, YSECTIONS+1))
    for ytiit in range(len(yticksinfo)):
        x = home_x + GRAPH_WIDTH
        y = home_y + GRAPH_HEIGHT * ytiit/(len(yticksinfo)-1)
        t.setpos(x - GRAPH_WIDTH/SECONDS_PER_BUFFER, y - TURTLE_PAD/3)
        t.write(round(yticksinfo[ytiit], 1))
def t_pop_yticks(t, d2da):
    for i in range(GRAPH_ROWS):
        for j in range(GRAPH_COLS):
            if [i,j] in UNUSED_GRAPHS:
                continue
            t_pop_ytick(t, i, j, d2da[i, j])
def init_z2da(rows, cols, cnt_zeros):
    z2da = np.empty((rows, cols), dtype=object)
    for i in range(rows):
        for j in range(cols):
            z2da[i, j] = np.zeros(cnt_zeros)
    return z2da
def init_graphs():
    global turtle_screen, t_professor, t_tickscribe, serverin_name, d2da, bankit, play, ot
    tkinter_canvas = tkinter.Canvas(app_object, width=TURTLE_WIDTH, height=TURTLE_HEIGHT)
    tkinter_canvas.grid(row=1, column=0, columnspan=6)
    
    turtle_screen = turtle.TurtleScreen(tkinter_canvas)
    turtle_screen.tracer(False)
    t_grunt = turtle.RawTurtle(turtle_screen)
    t_grunt.penup()
    t_grunt.hideturtle()
    t_awts(t_grunt)
    
    t_professor = turtle.RawTurtle(turtle_screen)
    t_professor.penup()
    t_professor.hideturtle()
    t_professor.pencolor('blue')
    
    t_tickscribe = turtle.RawTurtle(turtle_screen)
    t_tickscribe.penup()
    t_tickscribe.hideturtle()
    
    serverin_name = 'serverin.txt'
    if not os.path.exists(serverin_name):
        with open(serverin_name, "w") as f:
            f.write('0.0\t0.0\n'*ROWS_PER_BUFFER)
    d2da = init_z2da(GRAPH_ROWS, GRAPH_COLS, ROWS_PER_BUFFER)
    bankit = 0
    play = True
    ot = time.time()
def update_graphs():
    global bankit, ot
    if bankit <= -1 or bankit >= len(bank):
        bankit = 0
    start = bankit
    stop = bankit + ROWS_PER_PACKET
    banklet = bank[start:stop].T
    if play:
        d2da[0][0] = np.concatenate((d2da[0][0][ROWS_PER_PACKET:], banklet[0]))
        d2da[0][1] = np.concatenate((d2da[0][1][ROWS_PER_PACKET:], banklet[1]))
    d2da[1][0] = np.diff(d2da[0][0])
    d2da[1][1] = np.diff(d2da[0][1])
    d2da[2][0] = amplify(gaussify(d2da[1][0]))
    d2da[2][1] = gaussify(d2da[1][1])
    d2da[3][0] = keep_above_threshold(d2da[2][0])
    d2da[3][1] = keep_above_threshold(d2da[2][1])
    d2da[4][0] = strong_sign(d2da[3][0], d2da[3][1])
    d2da[4][1] = strong_sign(d2da[3][1], d2da[3][0])
    d2da[5][0] = pid(d2da[4][0], d2da[4][1], d2da[3][0])
    pm = max(1,int(sfloat(pidm_spinbox.get())))
    if play and bankit % (ROWS_PER_PACKET*pm) == 0:
        on_pid_instr(d2da[5][0][-1])
    t_professor.clear()
    t_pop_graphs(t_professor, d2da)
    t_tickscribe.clear()
    t_pop_yticks(t_tickscribe, d2da)
    turtle_screen.update()
    
    if play:
        bankit += ROWS_PER_PACKET
    nt = time.time()-ot
    ot = time.time()
    syncper = max(0,int(1000*(SECONDS_PER_PACKET - nt)))
    home_x, home_y = get_t_home(0,0)
    t_tickscribe.setpos(home_x, home_y + GRAPH_HEIGHT - TURTLE_PAD/3)
    t_tickscribe.write(f'{bankit/ROWS_PER_SECOND} {syncper} '+'rt'*(syncper!=0))
    turtle_screen.update()
    app_object.after(syncper, update_graphs)
####################################################################################################
# first half of screen aspect 3/3: first 3 buttons logic
def on_button1_click():
    file_path = filedialog.askopenfilename(filetypes=[('MATLAB .mat File', '*.mat')])
    if file_path:
        file_name = os.path.basename(file_path)
        button1.config(text=f'((👂)){file_name}')

        mat_contents = scipy.io.loadmat(file_path)
        reading = mat_contents['Biotrace'][2:4]
        py_reading = np.transpose(reading)
        np.savetxt(serverin_name, py_reading, delimiter='\t', fmt='%.10e')
        button3.invoke()
def on_button2_click():
    file_path = filedialog.askopenfilename(filetypes=[('Text File', '*.txt')])
    if file_path:
        file_name = os.path.basename(file_path)
        button2.config(text=f'((👂)){file_name}')

        txt_contents = np.loadtxt(file_path)
        np.savetxt(serverin_name, txt_contents, delimiter='\t', fmt='%.10e')
        button3.invoke()
def on_button3_click():
    global bank
    button3.config(text=f'((👂))')
    button1.config(state=tkinter.DISABLED)
    button2.config(state=tkinter.DISABLED)
    button3.config(state=tkinter.DISABLED)
    
    bank = np.genfromtxt(serverin_name, delimiter='\t')
    update_graphs()
def on_button4_click():
    global bankit
    bankit -= ROWS_PER_BUFFER
def on_button5_click():
    global play
    play = not play
####################################################################################################
# first half of screen pointcut
app_object = tkinter.Tk()
app_object.title(APP_OBJECT_NAME)
app_object.state('zoomed')
app_object.protocol("WM_DELETE_WINDOW", app_object.quit)
button1 = tkinter.Button(app_object, text='m→t→✈→👂', command=on_button1_click,
                        font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
button2 = tkinter.Button(app_object, text='t→✈→👂', command=on_button2_click,
                        font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
button3 = tkinter.Button(app_object, text='👂', command=on_button3_click,
                        font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
button4 = tkinter.Button(app_object, text=f'{-SECONDS_PER_BUFFER}s', command=on_button4_click,
                        font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
button5 = tkinter.Button(app_object, text='⏯', command=on_button5_click,
                        font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
button1.grid(row=0, column=0)
button2.grid(row=0, column=1)
button3.grid(row=0, column=2)
button4.grid(row=0, column=3)
button5.grid(row=0, column=4)
sigma_spinbox_label = tkinter.Label(app_object, text='sigma')
gdx_gain_spinbox_label = tkinter.Label(app_object, text='gdx gain')
threshold_spinbox_label = tkinter.Label(app_object, text='threshold (µV/sample)')
pidm_spinbox_label = tkinter.Label(app_object, text='pid multiplier - N')
pg_compr_spinbox_label = tkinter.Label(app_object, text='pg compress factor - N \\ <1,3>')
sigma_spinbox_label.grid(row=2, column=0)
gdx_gain_spinbox_label.grid(row=2, column=1)
threshold_spinbox_label.grid(row=2, column=2)
pidm_spinbox_label.grid(row=2, column=3)
pg_compr_spinbox_label.grid(row=2, column=4)
sigma_spinbox=tkinter.Spinbox(app_object, textvariable=tkinter.DoubleVar(value=SIGMA0),
                                from_=SPINBOX_MIN_VAL, to=SPINBOX_MAX_VAL, increment=SPINBOX_INCR_VAL, width=SPINBOX_WIDTH, format='%.2f',
                                font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
gdx_gain_spinbox=tkinter.Spinbox(app_object, textvariable=tkinter.DoubleVar(value=GDX_GAIN0),
                                from_=SPINBOX_MIN_VAL, to=SPINBOX_MAX_VAL, increment=SPINBOX_INCR_VAL, width=SPINBOX_WIDTH, format='%.2f',
                                font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
threshold_spinbox=tkinter.Spinbox(app_object, textvariable=tkinter.DoubleVar(value=THRESHOLD0),
                                from_=SPINBOX_MIN_VAL, to=SPINBOX_MAX_VAL, increment=SPINBOX_INCR_VAL, width=SPINBOX_WIDTH, format='%.2f',
                                font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
pidm_spinbox=tkinter.Spinbox(app_object, textvariable=tkinter.DoubleVar(value=PIDM0),
                                from_=SPINBOX_MIN_VAL, to=SPINBOX_MAX_VAL, increment=SPINBOX_INCR_VAL, width=SPINBOX_WIDTH, format='%.2f',
                                font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
pg_compr_spinbox=tkinter.Spinbox(app_object, textvariable=tkinter.DoubleVar(value=PG_COMPR0),
                                from_=SPINBOX_MIN_VAL, to=SPINBOX_MAX_VAL, increment=SPINBOX_INCR_VAL, width=SPINBOX_WIDTH, format='%.2f',
                                font=(COMMON_FONT, COMMON_FONT_BASE_SIZE))
sigma_spinbox.grid(row=3, column=0)
gdx_gain_spinbox.grid(row=3, column=1)
threshold_spinbox.grid(row=3, column=2)
pidm_spinbox.grid(row=3, column=3)
pg_compr_spinbox.grid(row=3, column=4)

init_graphs()
####################################################################################################
# second half of screen aspect
def on_vk_press(vkc):
    if len(vkc) == 1 and vkc in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ':
        text_widget.insert('insert', vkc)
    if vkc == vk_captions[0][5]:
        vk_msg = text_widget.get(1.0, 'end')
        text_widget.delete(1.0, 'end')
        vk_msg = vk_msg[:-1]
        if vk_msg and vk_msg[-1] == '~':
            vk_msg = vk_msg[:-1]
            while vk_msg and vk_msg[-1] != '~':
                vk_msg = vk_msg[:-1]
        text_widget.insert('end', vk_msg[:-1])
    if vkc == vk_captions[5][0]:
        text_widget.insert('insert', '~Áno.~')
    if vkc == vk_captions[5][1]:
        text_widget.insert('insert', '~Nie.~')
    if vkc == vk_captions[5][2]:
        text_widget.insert('insert', '~Začať znova.~')
    if vkc == vk_captions[5][3]:
        text_widget.insert('insert', '~Ďalšie slovo.~')
    if vkc == vk_captions[5][4]:
        text_widget.insert('insert', '~To nie je správne.~')
    if vkc == vk_captions[5][5]:
        text_widget.insert('insert', '~Ukončiť.~')
    text_widget.see('insert')
def update_vks_colour():
    for i in range(len(vk_captions)):
        for j in range(len(vk_captions[0])):
            if i == curs[0] and j == curs[1]:
                vks[i][j].config(bg='red')
            else:
                vks[i][j].config(bg='SystemButtonFace')
def init_virtkeys():
    global vk_captions, curs, vks
    vk_captions = [
        ['A', 'B', 'C', 'D', '', '⌫'],
        ['E', 'F', 'G', 'H', '', ''],
        ['I', 'J', 'K', 'L', 'M', 'N'],
        ['O', 'P', 'Q', 'R', 'S', 'T'],
        ['U', 'V', 'W', 'X', 'Y', 'Z'],
        ['✅', '❌', '🔄', '🆕', '⋆.ೃ࿔*:･', '🚪']
    ]
    curs = [2,2]
    vks = []
    for _ in range(len(vk_captions)):
        vks_row = []
        for __ in range(len(vk_captions[0])):
            vk = tkinter.Button(large_right_frame, text = vk_captions[_][__], command=lambda i=_, j=__: on_vk_press(vk_captions[i][j]),
                                 font=(COMMON_FONT, COMMON_FONT_BASE_SIZE * 2), width=3)
            vk.grid(row=_, column=__)
            vks_row.append(vk)
        vks.append(vks_row)
    update_vks_colour()
def on_pid_instr(instr):
    if instr == -3:
        on_vk_press(vk_captions[0][5])
    if instr == -2:
        vks[curs[0]][curs[1]].invoke()
    if instr == 1 and curs[1] >= 1:
        curs[1] -= 1
    if instr == 2 and curs[1] <= len(vk_captions[0])-2:
        curs[1] += 1
    if instr == 3 and curs[0] <= len(vk_captions)-2:
        curs[0] += 1
    if instr == 4 and curs[0] >= 1:
        curs[0] -= 1
    update_vks_colour()
####################################################################################################
# second half of screen pointcut
large_right_frame = tkinter.Frame(app_object)
large_right_frame.grid(row=0, column=6, rowspan=4)
init_virtkeys()
text_widget = tkinter.Text(large_right_frame, height=6, width=21, font=(COMMON_FONT, COMMON_FONT_BASE_SIZE * 2))
text_widget.grid(row=6, column=0, columnspan=len(vk_captions[0]))
app_object.mainloop()
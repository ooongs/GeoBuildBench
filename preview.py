import os
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from src.core.random_constr import Construction
from src.utils import get_project_root, get_examples_dir

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False

class DisplayWindow:

    def __init__(self, width, height, datadir):
        self.datadir = datadir
        self.fnames = sorted([
            fname for fname in os.listdir(datadir)
            if fname.endswith(".txt")
        ])
        self.index = 0
        self.construction = Construction(display_size=(width, height))
        self.load_construction()

        # Create matplotlib figure and axes
        self.fig, self.ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        self.fig.canvas.manager.set_window_title("Display")
        
        # Connect keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key_press)
        
        # Initial draw
        self.draw()

    def load_construction(self):
        if self.index == len(self.fnames): 
            self.index = 0
        elif self.index < 0: 
            self.index += len(self.fnames)
        fname = self.fnames[self.index]
        print(self.index, fname)
        self.construction.load(os.path.join(self.datadir, fname))
        self.construction.generate(max_attempts=0)

    def draw(self):
        self.construction.render(self.ax)
        self.fig.canvas.draw()

    def on_key_press(self, event):
        regenerate = False
        
        if event.key in ('up', 'down', 'left', 'right'):
            regenerate = True
            if event.key in ('up', 'left'): 
                self.index -= 1
            else: 
                self.index += 1
            self.load_construction()
            
        elif event.key == ' ':  # space
            regenerate = True
        elif event.key == 'escape':
            plt.close(self.fig)
            return
        else:
            return

        if regenerate:
            self.construction.generate(max_attempts=0)
            self.draw()

if __name__ == "__main__":
    # Use examples directory by default, or specify another directory
    default_dir = str(get_examples_dir())
    import sys
    datadir = sys.argv[1] if len(sys.argv) > 1 else default_dir
    window = DisplayWindow(400, 300, datadir)
    plt.show()

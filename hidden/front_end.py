import pandas as pd
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from tkinter.filedialog import askopenfilename
from back_end import *


W_PIXELS = 1920
H_PIXELS = 1080
MFRAME_W = 1200
MFRAME_H = 675
LIGHT_GRAY = "#d9d3d2"
DARK_PINK = "#5c384f"


class Window:
    def __init__(self, main):
        self.root = main
        # Test variables for checking for errors
        self.check_network = 0
        self.check_training = 0
        self.check_testing = 0
        self.check_input_loaded = 0
        self.check_output_loaded = 0
        self.check_test_loaded = 0
        self.check_reset = 0
        # Ttk for themed buttons
        style = Style()
        style.configure('top_bar.TButton', font =
                ('Helvetica', 12, 'bold'),
                foreground = DARK_PINK, relief = 'raised', width = 36, height = 8)
        style.configure('side_button.TButton', font =
                ('Helvetica', 12, 'bold'),
                foreground = DARK_PINK, relief = 'raised', width = 32)
        style.configure('choice.TCombobox', relief = 'reised', width = 32)

    def error_handler(self, error_code):
        # Switch case kept giving errors, so I had to resort to this
        if(error_code == 1):
            tk.messagebox.showerror(title = "Error #001", message = "Both Input (features) and Output (labels) layers must have 1 or more nodes")
        if(error_code == 2):
            tk.messagebox.showerror(title = "Error #002", message = "Must have at least 1 hidden layer")
        if(error_code == 3):
            tk.messagebox.showerror(title = "Error #003", message = "Create a Neural Network before this step")
        if(error_code == 4):
            tk.messagebox.showerror(title = "Error #004", message = "Training features (input layer) not loaded")
        if(error_code == 5):
            tk.messagebox.showerror(title = "Error #005", message = "Training labels (output layer) not loaded")
        if(error_code == 6):
            tk.messagebox.showerror(title = "Error #006", message = "Network must first be trained. Either start training the network or wait for network to finish training if you've already started the process")
        if(error_code == 7):
            tk.messagebox.showerror(title = "Error #007", message = "Reset the network first")
        if(error_code == 8):
            tk.messagebox.showerror(title = "Error #008", message = "No dataset file loaded for testing")

    def setup_elements(self):
        # Frames for different sections
        self.main_frame = Frame(self.root, width = MFRAME_W, height = MFRAME_H)
        self.top_bar = Frame(self.root)
        self.left_panel = Frame(self.root, width = 544, height = H_PIXELS)
        self.top_bar.pack(side = TOP, pady = 8)
        self.left_panel.pack(side = LEFT, padx = 16)
        self.main_frame.pack(side = RIGHT, padx = 16, pady = 16 )

        self.page1 = Frame(self.left_panel)
        self.page1.pack()
        self.page2 = Frame(self.left_panel)
        self.page3 = Frame(self.left_panel)


        # Top bar for chaging between pages
        self.build_page_button = Button(self.top_bar, text = "Build", style = 'top_bar.TButton', command = self.goto_buildpage)
        self.train_page_button = Button(self.top_bar, text = "Train", style = 'top_bar.TButton', command = self.goto_trainpage)
        self.test_page_button = Button(self.top_bar, text = "Test", style = 'top_bar.TButton', command = self.goto_testpage)
        self.reset_button = Button(self.top_bar, text = "Reset", style = 'top_bar.TButton', command = self.start_over)

        self.build_page_button.pack(side = LEFT, padx = 4)
        self.train_page_button.pack(side = LEFT, padx = 4)
        self.test_page_button.pack(side = LEFT, padx = 4)
        self.reset_button.pack(side = LEFT, padx = 4)


        # Page 1 for building the Neural Network 
        self.inputLabel = Label(self.page1, text = "Input layer")
        self.input_choice = Combobox(self.page1, style = 'choice.TCombobox' ,values = [1, 2, 3, 4, 5, 6])
        self.input_choice.current(3)

        self.output_label = Label(self.page1, text = "Output layer")
        self.output_choice = Combobox(self.page1, values = [1, 2, 3, 4, 5, 6])
        self.output_choice.current(3)

        self.hid_layer_label1 = Label(self.page1, text = "Hidden layer 1")
        self.hlayer1_choice = Combobox(self.page1, values = [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.hlayer1_choice.current(6)

        self.hid_layer_label2 = Label(self.page1, text = "Hidden layer 2")
        self.hlayer2_choice = Combobox(self.page1, values = [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.hlayer2_choice.current(0)

        self.hid_layer_label3 = Label(self.page1, text = "Hidden layer 3")
        self.hlayer3_choice = Combobox(self.page1, values = [0, 1, 2, 3, 4, 5, 6, 7, 8])
        self.hlayer3_choice.current(0)

        self.add_layer2_button = Button(self.page1, text = "Add Hidden layer", style = 'side_button.TButton', command = self.add_layer2)
        self.add_layer3_button = Button(self.page1, text = "Add Hidden layer", style = 'side_button.TButton', command = self.add_layer3)
        self.build_network_button = Button(self.page1, text = "Build the network", style = 'side_button.TButton', command = self.buildNetwork)

        #.grid() wasn't adapting to changing pages quite as well
        self.inputLabel.pack()
        self.input_choice.pack()
        self.output_label.pack()
        self.output_choice.pack()
        self.hid_layer_label1.pack()
        self.hlayer1_choice.pack()
        self.add_layer2_button.pack(pady = 8)
        self.build_network_button.pack(pady = 8)


        # Page 2 for training the dataset on current Network
        self.epoch_label = Label(self.page2, text = "Number of Epochs")
        self.epoch_choice = Combobox(self.page2, values = [10, 100, 1000, 10000, 100000])
        self.epoch_choice.current(1)

        self.batch_label = Label(self.page2, text = "Batch size")
        self.batch_choice = Combobox(self.page2, values = [10, 25, 50, 100, 250, 500])
        self.batch_choice.current(1)

        self.learn_rate_label = Label(self.page2, text = "Learning rate")
        self.learn_rate_choice = Combobox(self.page2, values = [0.1, 0.01, 0.001, 0.0001])
        self.learn_rate_choice.current(2)

        self.features_button = Button(self.page2, text = "Load training features\n(input layer)", style = 'side_button.TButton', command = self.load_features)
        self.features_file_name = Label(self.page2, text = "\t")

        self.labels_button = Button(self.page2, text = "Load training labels\n(output layer)", style = 'side_button.TButton', command = self.load_labels)
        self.labels_file_name = Label(self.page2, text = "\t")
        
        self.train_button = Button(self.page2, text = "Train", style = 'side_button.TButton', command = self.train_network)

        self.page2_box = tk.Text(self.page2, width = 40, height = 2)

        self.display_graph = Button(self.page2, text = "Accuracy and Loss graph", style = 'side_button.TButton', command = self.draw_graph)

        self.epoch_label.pack(pady = 8)
        self.epoch_choice.pack()
        self.batch_label.pack(pady = 8)
        self.batch_choice.pack()
        self.learn_rate_label.pack(pady = 8)
        self.learn_rate_choice.pack()
        self.features_button.pack(pady = 8)
        self.features_file_name.pack()
        self.labels_button.pack(pady = 8)
        self.labels_file_name.pack()
        self.train_button.pack(pady = 8)
        self.display_graph.pack()
        self.page2_box.pack(pady = 8)


        # Page 3 for testing the model that was built and trained
        self.load_test_page_button = Button(self.page3, text = "Load test dataset", style = 'side_button.TButton', command = self.load_test)
        self.test_file_name = Label(self.page3, text = "\t")
        self.test_button = Button(self.page3, text = "Predict", style = 'side_button.TButton', command = self.test_model)
        self.page3_box = Text(self.page3, width = 32, height = 24)

        self.load_test_page_button.pack(pady = 8)
        self.test_file_name.pack()
        self.test_button.pack(pady = 8)
        self.page3_box.pack()


        # Draw the canvas where the visual network lies
        self.canvas = Canvas(self.main_frame, bg =  LIGHT_GRAY, width = MFRAME_W, height = MFRAME_H, highlightbackground = DARK_PINK, highlightthickness = 1)
        self.canvas.pack()
        #self.scroller = Scrollbar(self.canvas, orient = 'horizontal')
        #self.scroller.pack(side = BOTTOM, fill = 'x')

        # Store chosen values to pass on to the back-end
        self.nn_choices = [self.input_choice, self.hlayer1_choice, self.hlayer2_choice, self.hlayer3_choice, self.output_choice]
        self.training_choices = [self.epoch_choice, self.batch_choice, self.learn_rate_choice]

    # Change to page 1
    def goto_buildpage(self):
        self.page1.pack()
        # Makes element invisible, doesn't delete element
        self.page2.pack_forget()
        self.page3.pack_forget()

    # Change to page 2 
    def goto_trainpage(self):
        self.page2.pack()
        self.page1.pack_forget()
        self.page3.pack_forget()

    # Change to page 3
    def goto_testpage(self):
        self.page3.pack()
        self.page1.pack_forget()
        self.page2.pack_forget()

    # Reset button fucntion
    def start_over(self):
        self.canvas.delete("all")
        self.page2_box.delete("all")
        self.page3_box.delete("all")
        self.features_file_name["text"] = "\t"
        self.labels_file_name["text"] = "\t"
        self.test_file_name["text"] = "\t"

        self.check_network = 0
        self.check_input_loaded = 0
        self.check_output_loaded = 0
        self.check_training = 0
        self.check_test_loaded = 0
        self.check_testing = 0
        self.check_reset = 1

        self.nn_choices[0] = self.input_choice
        self.nn_choice[1] = self.hlayer1_choice
        self.nn_choices[-1] = self.output_choice



    def add_layer2(self):
        self.add_layer2_button.pack_forget()
        self.hid_layer_label2.pack()
        self.hlayer2_choice.pack()
        self.add_layer3_button.pack(pady = 8)
        self.add_layer2_button.pack(pady = 8)
        self.add_layer2_button.configure(text = "Remove hidden layer", command = self.remove_layer2)
    def add_layer3(self):
        self.add_layer2_button.pack_forget()
        self.add_layer3_button.pack_forget()
        self.hid_layer_label3.pack()
        self.hlayer3_choice.pack()
        self.add_layer3_button.pack(pady = 8)
        self.add_layer3_button.configure(text = "Remove hidden layer", command = self.remove_layer3)
    def remove_layer2(self):
        self.hid_layer_label2.pack_forget()
        self.hlayer2_choice.pack_forget()
        self.hlayer2_choice.current(0)
        self.add_layer2_button.configure(text = "Add hidden layer", command = self.add_layer2)
        self.add_layer3_button.pack_forget()
    def remove_layer3(self):
        self.hid_layer_label3.pack_forget()
        self.hlayer3_choice.pack_forget()
        self.hlayer3_choice.current(0)
        self.add_layer3_button.pack_forget()
        self.add_layer2_button.pack(pady = 8)
        self.add_layer3_button.pack(pady = 8)
        self.add_layer3_button.configure(text = "Add hidden layer", command = self.add_layer3)

    # This function is a contribution from Sabina as I kept having trouble making it on my own
    def buildNetwork(self):
        # Array to store input values; zeros removed 
        self.dataCombobox2 = []
        # Arrays to store node coordinates for each layer
        self.inputCoord = []
        self.outputCoord = []
        self.hiddenCoord1 = []
        self.hiddenCoord2 = []
        self.hiddenCoord3 = []
        number_of_layers = 0
        if self.input_choice.get() == '0' or self.output_choice.get() == '0':
            self.error_handler(1)
        elif self.hlayer1_choice.get() == '0' and self.hlayer2_choice.get() == '0' and self.hlayer1_choice.get() == '0':
            self.error_handler(2)
        else:
            if self.check_network == 1: 
                self.canvas.delete("all")
            # Check if the input and output nodes number were changes to the ones in the data files
            if isinstance(self.nn_choices[0], str) and isinstance(self.nn_choices[-1], str):
                self.dataCombobox2.append(self.nn_choices[0])
                number_of_layers += 1
                for layer in self.nn_choices[1:-1]:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        number_of_layers += 1
                self.dataCombobox2.append(self.nn_choices[-1])
                number_of_layers += 1  
            # Check if only the input nodes number were changed 
            elif isinstance(self.nn_choices[0], str):
                self.dataCombobox2.append(self.nn_choices[0])
                number_of_layers += 1
                for layer in self.nn_choices[1:]:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        number_of_layers += 1
            # Check if only the output nodes number were changed 
            elif isinstance(self.nn_choices[-1], str):
                for layer in self.nn_choices[:-1]:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        number_of_layers += 1
                self.dataCombobox2.append(self.nn_choices[-1])
                number_of_layers += 1
            else:
                for layer in self.nn_choices:
                    if layer.get() > '0':
                        self.dataCombobox2.append(layer.get())
                        number_of_layers += 1
            print("Nodes in", number_of_layers, "layers:", self.dataCombobox2)
            self.createNode(number_of_layers)
            self.createArrows()
            

    # This function is Sabina's contribution as I kept having trouble positioning every node
    def createNode(self, number_of_layers):
        distX = MFRAME_W / (number_of_layers + 1) 
        i = 1        
        for nodes in self.dataCombobox2:
            for node in range(1,int(nodes) + 1):
                x = i * distX - 96
                y = node * (MFRAME_H / (int(nodes) + 1)) - 64
                self.canvas.create_oval(x, y, x + 48, y + 48, fill="#23597d")
                # Append input layer coordinates
                if i == 1: self.inputCoord.append([x,y])
                # Append first hidden layer coordinates
                elif i == 2: self.hiddenCoord1.append([x,y])
                # Append either output coordinates, or second hidden layer 
                elif i == 3: 
                    if i == number_of_layers:
                        self.outputCoord.append([x,y])
                    else: self.hiddenCoord2.append([x,y])
                # Append either output coordinates, or third hidden layer 
                elif i == 4: 
                    if i == number_of_layers:
                        self.outputCoord.append([x,y])
                    else: self.hiddenCoord3.append([x,y])
                # Append output layer coordinates
                elif i == 5: self.outputCoord.append([x,y])
            i += 1
        self.check_network = 1
        # Add input values from the data file to the nodes
        if self.check_input_loaded:
            j = 0
            for coord in self.inputCoord:
                self.canvas.create_text(coord[0], coord[1], text=self.train[j][0], fill="black", tag="drag")
                j += 1
        # Add output values from the data file to the nodes
        if self.check_output_loaded:
            k = 0
            for coord in self.outputCoord:
                self.canvas.create_text(coord[0], coord[1], text=self.y_train[k][0], fill="black", tag="drag")
                k += 1

    # This function is Sabina's contribution as I kept having trouble trying to connect the lines
    def createArrows(self):
        shift = 48
        # Create arrow between each node pair
        for cin in self.inputCoord:
            for ch1 in self.hiddenCoord1:
                self.canvas.create_line(cin[0] + shift, cin[1] + shift/2, ch1[0], ch1[1] + shift/2, width=2)
                if self.hiddenCoord2: 
                    for ch2 in self.hiddenCoord2:
                        self.canvas.create_line(ch1[0] + shift, ch1[1] + shift/2, ch2[0], ch2[1] + shift/2, width=2)
                        if self.hiddenCoord3:
                            for ch3 in self.hiddenCoord3:
                                self.canvas.create_line(ch2[0] + shift, ch2[1] + shift/2, ch3[0], ch3[1] + shift/2, width=2)
        for cout in self.outputCoord:
            if self.hiddenCoord3:
                for ch3 in self.hiddenCoord3:
                    self.canvas.create_line(ch3[0] + shift, ch3[1] + shift/2, cout[0], cout[1] + shift/2, width=2)
            elif self.hiddenCoord2:
                for ch2 in self.hiddenCoord2:
                    self.canvas.create_line(ch2[0] + shift, ch2[1] + shift/2, cout[0], cout[1] + shift/2, width=2)
            else: 
                for ch1 in self.hiddenCoord1:
                    self.canvas.create_line(ch1[0] + shift, ch1[1] + shift/2, cout[0], cout[1] + shift/2, width=2)


    def load_features(self):
        if self.check_network == 1:
            self.store_input = askopenfilename(initialdir = "./sample_dataset", filetypes = [("CSV Files", "*.csv"), ("Text file", "*.txt"), ("Excel file", "*.xlxs")])
            self.train = pd.read_csv(self.store_input, sep = ',', header = None)
            local_file_name = self.store_input.split('/')[len(self.store_input.split('/'))-1]
            self.features_file_name["text"] = local_file_name
            self.nn_choices[0] = str(self.train.shape[1])
            # Must be called to revisualize table with input values on input layers
            self.buildNetwork()
            self.check_input_loaded = 1
        else:
            self.error_handler(3)

    def load_labels(self):
        if self.check_network == 1:
            self.store_output = askopenfilename(initialdir = "./sample_dataset", filetypes = [("CSV Files","*.csv"), ("Text file", "*.txt"), ("Excel file", "*.xlxs")])
            self.y_train = pd.read_csv(self.store_output, sep = ',', header = None)
            local_file_name = self.store_output.split('/')[len(self.store_output.split('/'))-1]
            self.labels_file_name["text"] = local_file_name
            self.nn_choices[-1] = str(self.y_train.shape[1])
            # Must be called to revisualize table with output values on output layers
            self.buildNetwork()
            self.check_output_loaded = 1
        else:
            self.error_handler(3)

    def train_network(self):
        if self.check_network == 0:
            self.error_handler(3)
        elif self.check_input_loaded == 0:
            self.error_handler(4)
        elif self.check_output_loaded == 0:
            self.error_handler(5)
        else:
            # Remove previous result
            if self.check_training == 1:
                self.page2_box.delete("1.0", "end")
            self.epoch = int(self.training_choices[0].get())
            self.batch_size = int(self.training_choices[1].get())
            self.learn_rate = float(self.training_choices[2].get())
            self.train = MachineLearning(self.train, self.y_train, self.dataCombobox2, self.batch_size, self.epoch, self.learn_rate)
            if self.check_reset == 0 or self.check_reset == 1:
                self.train.start_training()
            else:
                self.error_handler(7)
            self.check_reset = 2
            self.page2_box.insert(1.0, "Accuracy = {}".format(self.train.accuracy[1]))
            self.page2_box.insert("end", "\n")
            self.page2_box.insert("end", "Loss = {}".format(self.train.accuracy[0]))
            self.check_training = 1
    
    def draw_graph(self):
        if self.check_training == 1:
            self.train.plot_graph()
        else:
            self.error_handler(6)

    def load_test(self):
        if self.check_training == 1:
            self.store_test_date = askopenfilename(initialdir="./sample_dataset", filetypes=[("CSV Files","*.csv"), ("Text file", "*.txt"), ("Excel file", "*.xlxs")])
            self.test_file_container = pd.read_csv(self.store_test_date, sep = ',', header = None)
            local_file_name = self.store_test_date.split('/')[len(self.store_test_date.split('/'))-1]
            self.test_file_name["text"] = "{} loaded".format(local_file_name)
            self.check_test_loaded = 1         
        else:
            self.error_handler(8)

    def test_model(self):
        if self.check_network == 0:
            self.error_handler(3)
        elif self.check_training == 0:
            self.error_handler(6)
        elif self.check_test_loaded == 0:
            self.error_handler(8)
        else:
            # Remove previous predictions
            if self.check_testing == 1:
                self.page3_box.delete("1.0", "end")
            self.train.test_model(self.test_file_container)
            self.check_testing = 1
            # Post predictions in the box
            for element in self.train.predictions:
                self.page3_box.insert("end", element)
                self.page3_box.insert("end", "\n")


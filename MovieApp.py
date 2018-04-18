from tkinter import font as tkFont
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import tree
from collections import defaultdict
import random
import pydot
from io import StringIO
import pydotplus
from multiprocessing import Process
import tkinter as tk
from tkinter import filedialog
import kmedoid as kmd
from tkinter import messagebox

import pandas


class MovieApp:

    def __init__(self, master, dataframe, clusters,edit_rows=[]):
        """ master    : tK parent widget
        dataframe : pandas.DataFrame object"""
        self.root = master
        self.root.minsize(width=600, height=400)
        self.root.title('IMDb movie success predictor')

        self.main = tk.Frame(self.root)
        self.main.pack(fill=tk.BOTH, expand=True)

        self.lab_opt = {'background': 'darkgreen', 'foreground': 'white'}

#       the dataframe
        self.df = dataframe
        self.clusters = clusters
        self.dat_cols = list(self.df)
        if edit_rows:
            self.dat_rows = edit_rows
        else:
            self.dat_rows = range(len(self.df))
        self.rowmap = {i: row for i, row in enumerate(self.dat_rows)}

#       subset the data and convert to giant list of strings (rows) for viewing
        self.sub_data = self.df.ix[self.dat_rows, self.dat_cols]
        self.sub_datstring = self.sub_data.to_string(
            index=False, col_space=13).split('\n')
        self.title_string = self.sub_datstring[0]
        self.results = ""

# save the format of the lines, so we can update them without re-running
# df.to_string()
        self._get_line_format(self.title_string)

#       fill in the main frame
        self._fill()

#       updater for tracking changes to the database
        self.update_history = []

##################
# ADDING WIDGETS #
##################
    def _fill(self):
        self.canvas = tk.Canvas(self.main)
        self.canvas.pack(fill=tk.BOTH, expand=tk.YES)

        self._init_scroll()
        self._init_lb()
        self._pack_config_scroll()
        self._pack_bind_lb()
        self._fill_listbox()
        self._make_editor_frame()

##############
# SCROLLBARS #
##############
    def _init_scroll(self):
        self.scrollbar = tk.Scrollbar(self.canvas, orient="vertical")
        self.xscrollbar = tk.Scrollbar(self.canvas, orient="horizontal")

    def _pack_config_scroll(self):
        self.scrollbar.config(command=self.lb.yview)
        self.xscrollbar.config(command=self._xview)
        self.scrollbar.pack(side="right", fill="y")
        self.xscrollbar.pack(side="bottom", fill="x")

    def _onMouseWheel(self, event):
        self.title_lb.yview("scroll", event.delta, "units")
        self.lb.yview("scroll", event.delta, "units")
        return "break"

    def _xview(self, *args):
        """connect the yview action together"""
        self.lb.xview(*args)
        self.title_lb.xview(*args)

################
# MAIN LISTBOX #
################
    def _init_lb(self):
        self.title_lb = tk.Listbox(self.canvas, height=1,
                                   font=tkFont.Font(self.canvas,
                                                    family="Courier",
                                                    size=14),
                                   yscrollcommand=self.scrollbar.set,
                                   xscrollcommand=self.xscrollbar.set,
                                   exportselection=False)

        self.lb = tk.Listbox(self.canvas,
                             font=tkFont.Font(self.canvas,
                                              family="Courier",
                                              size=14),
                             yscrollcommand=self.scrollbar.set,
                             xscrollcommand=self.xscrollbar.set,
                             exportselection=False,
                             selectmode=tk.EXTENDED)

    def _pack_bind_lb(self):
        self.title_lb.pack(fill=tk.X)
        self.lb.pack(fill="both", expand=True)
        self.title_lb.bind("<MouseWheel>", self._onMouseWheel)
        self.lb.bind("<MouseWheel>", self._onMouseWheel)

    def _fill_listbox(self):
        """ fill the listbox with rows from the dataframe"""
        self.title_lb.insert(tk.END, self.title_string)
        for line in self.sub_datstring[1:]:
            self.lb.insert(tk.END, line)
            self.lb.bind('<ButtonRelease-1>', self._listbox_callback)
        self.lb.select_set(0)

    def _listbox_callback(self, event):
        """ when a listbox item is selected"""
        items = self.lb.curselection()
        if items:
            new_item = items[-1]
            dataVal = str(
                self.df.ix[
                    self.rowmap[new_item],
                    self.opt_var.get()])
            self.entry_box_old.config(state=tk.NORMAL)
            self.entry_box_old.delete(0, tk.END)
            self.entry_box_old.insert(0, dataVal)
            self.entry_box_old.config(state=tk.DISABLED)

#####################
# FRAME FOR EDITING #
#####################
    def _make_editor_frame(self):
        
        """ make a frame for editing dataframe rows"""
        self.editorFrame = tk.Frame(
            self.main, bd=2, padx=2, pady=2, relief=tk.GROOVE)
        self.editorFrame.pack(fill=tk.BOTH, side=tk.LEFT)

#       column editor
        self.col_sel_lab = tk.Label(
            self.editorFrame,
            text='Show Clusters:',
            **self.lab_opt)
        self.col_sel_lab.grid(row=0, columnspan=2, sticky=tk.W + tk.E)

        self.show_cluster = tk.Button(
            self.editorFrame,
            text='View Clusters',
            command=self.plot_graph)
        self.show_cluster.grid(row=0,column=3, columnspan=2, sticky=tk.W + tk.E)

        self.col_sel_lab = tk.Label(
            self.editorFrame,
            text='Process Clusters:',
            **self.lab_opt)
        self.col_sel_lab.grid(row=1, columnspan=2, sticky=tk.W + tk.E)

        self.show_cluster = tk.Button(
            self.editorFrame,
            text='Process',
            command=self.process_data_set)
        self.show_cluster.grid(row=1,column=3, columnspan=2, sticky=tk.W + tk.E)

        self.show_decision = tk.Button(
            self.editorFrame,
            text='Show Decision Tree Results',
            command=self.show_results)
        self.show_decision.grid(row=2,columnspan=2, sticky=tk.W + tk.E)

################
# SELECT MODES #
################
    def _sel_mode(self):
        
        self.mode_frame = tk.Frame(
            self.main, bd=2, padx=2, pady=2, relief=tk.GROOVE)
        self.mode_frame.pack(fill=tk.BOTH, side=tk.LEFT)

        tk.Label(self.mode_frame, text='Selection mode', **
                 self.lab_opt).pack(fill=tk.BOTH, expand=tk.YES)

        self.mode_lb = tk.Listbox(
            self.mode_frame,
            height=2,
            width=16,
            exportselection=False)
        self.mode_lb.pack(fill=tk.BOTH, expand=tk.YES)
        self.mode_lb.insert(tk.END, 'Multiple selection')
        self.mode_lb.bind('<ButtonRelease-1>', self._mode_lb_callback)
        self.mode_lb.insert(tk.END, 'Find and replace')
        self.mode_lb.bind('<ButtonRelease-1>', self._mode_lb_callback)
        self.mode_lb.select_set(0)

    def _mode_lb_callback(self, event):
        items = self.mode_lb.curselection()
        if items[0] == 0:
            self._swap_mode('multi')
        elif items[0] == 1:
            self._swap_mode('findrep')

    def _swap_mode(self, mode='multi'):
        """swap between modes of interaction with database"""
        self.lb.selection_clear(0, tk.END)
        self._swap_lab(mode)
        if mode == 'multi':
            self.lb.config(state=tk.NORMAL)
            self.entry_box_old.config(state=tk.DISABLED)
            self.update_b.config(
                command=self._updateDF_multi,
                text='Update multi selection')
        elif mode == 'findrep':
            self.lb.config(state=tk.DISABLED)
            self.entry_box_old.config(state=tk.NORMAL)
            self.update_b.config(
                command=self._updateDF_findrep,
                text='Find and replace')
        self.entry_box_new.delete(0, tk.END)
        self.entry_box_new.insert(0, "Enter new value")

    def _swap_lab(self, mode='multi'):
        """ alter the labels on the editor frame"""
        if mode == 'multi':
            self.old_val_lab.config(text='Old value:')
            self.new_val_lab.config(text='New value:')
        elif mode == 'findrep':
            self.old_val_lab.config(text='Find:')
            self.new_val_lab.config(text='Replace:')

#################
# EDIT COMMANDS #
#################
    def _updateDF_multi(self):
        """ command for updating via selection"""
        self.col = self.opt_var.get()
        items = self.lb.curselection()
        self._track_items(items)

    def _updateDF_findrep(self):
        """ command for updating via find/replace"""
        self.col = self.opt_var.get()
        old_val = self.entry_box_old.get()
        try:
            items = pandas.np.where(
                self.sub_data[
                    self.col].astype(str) == old_val)[0]
        except TypeError as err:
            self.errmsg(
                '%s: `%s` for column `%s`!' %
                (err, str(old_val), self.col))
            return
        if not items.size:
            self.errmsg(
                'Value`%s` not found in column `%s`!' %
                (str(old_val), self.col))
            return
        else:
            self._track_items(items)
            self.lb.config(state=tk.DISABLED)

    def _undo(self):
        if self.update_history:
            updated_vals = self.update_history.pop()
            for idx, val in updated_vals['vals'].items():
                self.row = self.rowmap[idx]
                self.idx = idx
                self.df.set_value(self.row, updated_vals['col'], val)
                self._rewrite()
            self.sync_subdata()

    def _undo(self):
        if self.update_history:
            updated_vals = self.update_history.pop()
            for idx, val in updated_vals['vals'].items():
                self.row = self.rowmap[idx]
                self.idx = idx
                self.df.set_value(self.row, updated_vals['col'], val)
                self._rewrite()
            self.sync_subdata()

####################
# HISTORY TRACKING #
####################
    def _track_items(self, items):
        """ this strings several functions together,
        updates database, tracks changes, and updates database viewer"""
        self._init_hist_tracker()
        for i in items:
            self.idx = i
            self.row = self.rowmap[i]
            self._track()
            self._setval()
            self._rewrite()
        self._update_hist_tracker()
#       update sub_data used w find and replace
        self.sync_subdata()

    def _setval(self):
        """ update database"""
        try:
            self.df.set_value(self.row, self.col, self.entry_box_new.get())
        except ValueError:
            self.errmsg(
                'Invalid entry `%s` for column `%s`!' %
                (self.entry_box_new.get(), self.col))

    def _init_hist_tracker(self):
        """ prepare to track a changes to the database"""
        self.prev_vals = {}
        self.prev_vals['col'] = self.col
        self.prev_vals['vals'] = {}

    def _track(self):
        """record a change to the database"""
        self.prev_vals['vals'][self.idx] = str(self.df.ix[self.row, self.col])

    def _update_hist_tracker(self):
        """ record latest changes to database"""
        self.update_history.append(self.prev_vals)

    def sync_subdata(self):
        """ syncs subdata with data"""
        self.sub_data = self.df.ix[self.dat_rows, self.dat_cols]

#################
# ERROR MESSAGE #
#################
    def errmsg(self, message):
        """ opens a simple error message"""
        errWin = tk.Toplevel()
        tk.Label(
            errWin,
            text=message,
            foreground='white',
            background='red').pack()
        tk.Button(errWin, text='Ok', command=errWin.destroy).pack()

##################
# UPDATING LINES #
##################
    def _rewrite(self):
        """ re-writing the dataframe string in the listbox"""
        new_col_vals = self.df.ix[self.row, self.dat_cols].astype(str).tolist()
        new_line = self._make_line(new_col_vals)
        if self.lb.cget('state') == tk.DISABLED:
            self.lb.config(state=tk.NORMAL)
            self.lb.delete(self.idx)
            self.lb.insert(self.idx, new_line)
            self.lb.config(state=tk.DISABLED)
        else:
            self.lb.delete(self.idx)
            self.lb.insert(self.idx, new_line)

    def _get_line_format(self, line):
        """ save the format of the title string, stores positions
            of the column breaks"""
        pos = [1 + line.find(' ' + n) + len(n) for n in self.dat_cols]
        self.entry_length = [pos[0]] + \
            [p2 - p1 for p1, p2 in zip(pos[:-1], pos[1:])]

    def _make_line(self, col_entries):
        """ add a new line to the database in the correct format"""
        new_line_entries = [('{0: >%d}' % self.entry_length[i]).format(entry)
                            for i, entry in enumerate(col_entries)]
        new_line = "".join(new_line_entries)
        return new_line
    
    def plot_graph(self):
        markers = ['bo', 'go', 'ro', 'c+', 'm+', 'y+']
        clusters = self.clusters
        for i in range(0, len(clusters.keys())):
            data = clusters.get(i)
            for j in range(0, len(data)):
                df = data[j]
                plt.plot(df[0], df[1], markers[i])
        plt.xlabel('IMDb Scores')
        plt.ylabel('Gross')
        plt.title('K-medoid clusters')
        plt.legend()
        plt.show()
    
    def assign_target(self,row):

        x = row['movie_title']
        clusters = self.clusters
        for i in range(0, len(clusters.keys())):
            data = clusters.get(i)
            for j in range(0, len(data)):
                df = data[j]
                if df[2] == x:
                    row['cluster'] = 'cluster'+str(i)

        return row

    def show_results(self):
        messagebox.showinfo("Results of the decision tree model",self.results)

    def process_data_set(self):
        
        #choosing features for decision tree
        columns = ['num_user_for_reviews', 'budget'
                    , 'num_critic_for_reviews', 'movie_title','movie_facebook_likes','num_voted_users','duration']
        
        df = self.df[columns]
        df = df.apply(self.assign_target, axis=1)
        df.drop(labels = ['movie_title'], axis = 1, inplace = True)

        #creating training and test sets
        splitSet = StratifiedShuffleSplit(
                n_splits=1, test_size=0.2, random_state=0)

        for train_index, test_index in splitSet.split(df, df['cluster']):
            train_set = df.loc[train_index]
            test_set = df.loc[test_index]

        Y_train = train_set.cluster
        X_train = train_set[train_set.columns.drop('cluster')]
        Y_test = test_set.cluster
        X_test = test_set[test_set.columns.drop('cluster')]

        #Creating decision tree 
        decision_tree = DecisionTreeClassifier()
        decision_tree.fit(X_train, Y_train)

        predictions = decision_tree.predict(X_test)

        output = 'Score of the decision tree='+str(decision_tree.score(X_test, Y_test))+('\n')

        output = output+'\nDecision Tree Confusion Matrix\n\n'+str(confusion_matrix(Y_test,predictions))+('\n')
       
        output = output+'\nDecision Tree Classification Report\n\n'+str(classification_report(Y_test,predictions))+('\n')

        #Applying random forest classifier
        rfc = RandomForestClassifier(n_estimators=2000)
        rfc.fit(X_train, Y_train)
        output = output+('Random Forest Statistics\n')s

        rfc_pred = rfc.predict(X_test)
        output = output+'\nRandom Forest Confusion Matrix\n\n'+str(confusion_matrix(Y_test,rfc_pred))+('\n')
        
        output = output+'\nRandom Forest Classification Report\n\n'+str(classification_report(Y_test,rfc_pred))+('\n')
       
        print(output)

        self.results = output

    
if __name__=='__main__':
    
    columns = ['movie_title','num_user_for_reviews', 'budget', 'num_critic_for_reviews','movie_facebook_likes',
    'num_voted_users','duration','gross', 'imdb_score']
    
    #loading dataset
    df = pd.read_csv('movie_metadata.csv').dropna(axis=0).reset_index(drop=True)

    dataset = df[['gross', 'imdb_score', 'movie_title']]
    dataset = dataset.values.tolist()

    clusters = kmd.kMedoids(dataset, 5, np.inf, 0)
    
    root = tk.Tk()
    editor = MovieApp(root, df[columns], clusters)

    root.mainloop()
    
    #Visualising the decision tree (runs only in Jupyter notebook)
    '''dot_data = StringIO()
    export_graphviz(decision_tree, out_file=dot_data,
                        filled=True, rounded=True,
                        special_characters=True, impurity=False, feature_names=train_set.columns.drop('cluster').drop('index'))
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png("dtree.png")'''
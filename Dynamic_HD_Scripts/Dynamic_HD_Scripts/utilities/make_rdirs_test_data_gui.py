import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import itertools
from Dynamic_HD_Scripts.tools.compute_catchments import compute_catchments_cpp as compute_catchments
from Dynamic_HD_Scripts.tools.flow_to_grid_cell import create_hypothetical_river_paths_map as compute_acc
import numpy as np
import tempfile
from matplotlib import pyplot as plt

class ConfigureCell:

  states = ["^","/`",">","\\.","v","./","<","`\\","o","~"]
  rdir_values = ["8","9","6","3","2","1","4","7","5","0"]

  def __init__(self,button,i,j,rdirs):
    self.i = i
    self.j = j
    self.button = button
    self.state_iter = itertools.cycle(self.states)
    self.state = next(self.state_iter)
    self.rdirs = rdirs
    self.button.configure(text=self.state)

  def __call__(self):
    self.state = next(self.state_iter)
    self.process_state_change()

  def process_state_change(self):
    self.button.configure(text=self.state)
    self.rdirs[self.i+1,self.j] = self.rdir_values[self.states.index(self.state)]

  def reverse(self,e):
    for _ in range(len(self.states)-2):
      next(self.state_iter)
    self.state = next(self.state_iter)
    self.process_state_change()

def update_output(rdirs,plots,figs,root):
    print(rdirs)
    loops_log = tempfile.NamedTemporaryFile(delete=True)
    catch = compute_catchments(rdirs,loops_log.name)  
    print(catch)
    plots["catch"].imshow(catch)
    with open(loops_log.name,"r") as f:
      ll_lines = f.readlines()
    if len(ll_lines) > 1:
      messagebox.showinfo(root,"Invalid River Directions")
    else:
      acc = compute_acc(rdirs,nlat=22,nlong=20,use_f2py_func=False)
      print(acc)
      plots["acc"].imshow(acc)
    figs["acc"].canvas.draw()
    figs["catch"].canvas.draw()

def main():
  root = tk.Tk()
  root.minsize(1000,600)
  root.geometry('1000x600+250+250')
  rdirs = np.ones((22,20))*8
  rdirs[0,:] = 0
  rdirs[21,:] = 0
  fig_c = plt.figure()
  ax_catch = fig_c.add_subplot(1,1,1)
  ax_catch.imshow(np.zeros((22,20)))
  fig_a = plt.figure()
  ax_acc = fig_a.add_subplot(1,1,1)
  ax_acc.imshow(np.zeros((22,20)))
  plt.show(block=False)
  plots = {"acc":ax_acc,"catch":ax_catch}
  figs = {"acc":fig_a,"catch":fig_c}
  button = ttk.Button(text="Update Diagnostics",command=lambda:update_output(rdirs,plots,figs,root))
  button.grid(column=0,row=0,sticky="nsew")
  frame = ttk.Frame(root)
  frame.grid(column=0,row=1,sticky="nsew")
  for i in range(20):
    frame.rowconfigure(i,weight=1)
  for i in range(20):
    row = ttk.Frame(frame)
    row.grid(column=0,row=i,sticky="nsew")
    for j in range(20):
        row.columnconfigure(j,weight=1)
    for j in range(20):
      button = ttk.Button(row,width=1)
      button.grid(column=j,row=0,sticky="nsew")
      cmd = ConfigureCell(button,i,j,rdirs)
      button.configure(command=cmd)
      button.bind("<Button-2>",cmd.reverse)
  update_output(rdirs,plots,figs,root)
  root.mainloop()

if __name__ == '__main__':
    main()

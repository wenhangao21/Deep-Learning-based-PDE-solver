# Deep-Learning-based-PDE-solver

The Raw Code folder contains only the raw code, run the code can get all the data. 

For simplicity, in the "Code by Category" folder, each sub-folder contains 
code sorted by category+ corresponding data generated + data produced by other code that's needed to run the code.
(same code in both folders).

One thing to note is to change directory to the directory of the code since the code generates files and read files

Order of running code in Code by Category:
1. Trainining: run main.py
2. Plot and Animation: visualization.py, semilog_plot.mlx
3. Make Prediction: get_result.py
4. Get Data for Plot and Animation: get_data_for_visualization.py

Order of running code in Raw Code:
1. run main.py to get the solution network parameters
2. now, we can run get_result.py, get_data_for_visualization.py, or semilog_plot.mlx, in any order
3. run visualization.py to get animation of solutions.


This project contains 7 python scripts:
1. main.py is the main script that trains the neural network, run this file, you will get a pytorch .pt file that contains trained parameter setting values for the solution network.
2. get_data_for_visualization.py creates a set of grid points in the domain, and correponding solutions(true and network), and will save them as .mat file
3. get_result.py is a script in which you can find the solution of the points you want in the domain. I put 2 points [0.5, 0.5, 0.5], [0.9, 0.3, 0.2] as an example. This is useful if you want the result for some specific points.
4. visualization.py is the script that creates animated image of the solutions for comparison purpose.
5. network_setting.py specifies the set up of the neural network. Used in other scripts
6. problem_setting.py contains the set up of the PDE, such as the differential operator, the bdy operator, the domain, spacial dimension, time interval etc.. Spatial dimension is set to be 2D. Used in other scripts
7. utilities.py contains all the useful tools such as calculating l2 error. Used in other scripts

This project also contains 1 matlab live script file
semilog_plot.mlx: plot the l2 error decay as the network learns(used matlab because it has LaTex interpreter for image tile and axis labels)
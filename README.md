# mmp-code
ACO for solving TSPs 

To run this project you need to install python3 (I used version 3.6.9) and have the libraries in the requirements.txt file. FFMPEG might need to be installed on the system in order for the tour improvement animators to work.

The testdata directory contains the testdata that I have download/created. testdata/created contains the testdata that I made. I ran my algorithms over the 100_node, 200_node and 300_node problems only.

More testdata can be found from:

http://www.math.uwaterloo.ca/tsp/vlsi/
http://www.math.uwaterloo.ca/tsp/world/countries.html

There are a few files that can be run:

<h3> TSP File Generator
    
</h3>

This is the generate_tsp.py file.
Edit the file to change the parameters and then execute it using python:
python3 generate_tsp.py.

<h3> The Main Program
    
</h3>

This file is controlled through command line parameters. These can be seen by using the -help command or by viewing the table in appendix D of the report.
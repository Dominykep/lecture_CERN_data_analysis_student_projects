# vln_hs_price_project Visualisation

## This is the second part of the vln_hs_price_project

The first part scrapes data and loads it into a .csv, this project takes that data and visualises it.

## Requirements
Libraries: pandas, matplotlib, numpy; 
Data: listings.csv (in the project directory)

## Usage
After running python main.py in the project directory, 
the user is prompted with a choice between 3 predetermined charts and a fourth option to create their own. 
If the user picks the fourth option, they can select between a scatter plot diagram and a bar chart. 
If the bar chart is selected, the x-axis is fixed to Neighbourhoods, the y-axis can be chosen by the user.
If the scatter plot is selected, both the x and y axes can be chosen by the user.
After picking the axes (or axis), the user chooses where to put the legend (top left, top right, bottom left, bottom right).
Once the user is done with the app, they can type 0 in the beginning portion of chart selection to end the program.

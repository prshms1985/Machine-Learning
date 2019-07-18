# -*- coding: utf-8 -*-
"""
Created on Feb 24 2018
Author: Prashamsh Takkalapally
Program - Machine Problem 3 – Course Planning using a Constraint Satisfaction Problem Formulation
"""

from constraint import *
import pandas as pd

# Defining list of terms to map numbers to term
all_terms = {1: 'Year 1 Fall 1', 2: 'Year 1 Fall 2', 3: 'Year 1 Spring 1', 
4: 'Year 1 Spring 2', 5: 'Year 1 Summer 1', 6: 'Year 1 Summer 2', 7: 'Year 2 Fall 1', 
8: 'Year 2 Fall 2', 9: 'Year 2 Spring 1', 10: 'Year 2 Spring 2', 11: 'Year 2 Summer 1', 
12: 'Year 2 Summer 2', 13: 'Year 3 Fall 1', 14: 'Year 3 Fall 2'}

#Reading excel data
csp_data = pd.read_excel("csp_course_rotations.xlsx", [0,1])

#Reading two available sheets in the excel, one containing courses and other pre-requisites
course = csp_data[0]
preq = csp_data[1]

#Capturing core,foundation in one variable and elective in another variable
core_foundation = course[(course['Type']=='foundation') | (course['Type']=='core')]['Course'].tolist()
electives = course[course['Type']=='elective']['Course'].tolist()

#Function to calculate the greater and lesser values
def func1(a,b):
    return a < b

# Function that takes few electives at a time and returns the possible solutions
# for the elective, core, foundation combination
def courses(elective1,elective2, elective3):
    problem = Problem()
    #Adding problem variables for courses.
    # Problem variables contain term #s that are available for each course
    for i in range(len(course)):
        course_options = []
        for j in range(2,len(course.columns)):
            k = 0
            if (course.iloc[i,j] == 1):
                course_options.append(j-1+k)
                k = k+6
                if (j-1+k)<=14:
                    course_options.append(j-1+k)
                    k=k+6
                if (j-1+k)<=14:
                    course_options.append(j-1+k)
        #Adding problem variables for core and foundation courses
        if (course.iloc[i,0] not in electives):                
            problem.addVariable(course.iloc[i,0], course_options)
        #Adding problem variables for electives
        if ((course.iloc[i,0]==elective1) or (course.iloc[i,0]==elective2)or (course.iloc[i,0]==elective3)):            
            problem.addVariable(course.iloc[i,0], course_options)
    problem.addVariable('', [6,12])
    
    #Constraint for different values
    problem.addConstraint(AllDifferentConstraint())
    # Constraints from the pre-requisites sheet for core, foundaiton and electives
    for i in range(len(preq)):
        if ((preq.iloc[i,0] not in electives)& (preq.iloc[i,1] not in electives)):
            problem.addConstraint(func1, [preq.iloc[i,0], preq.iloc[i,1]])
        if ((preq.iloc[i,1] == elective1) or (preq.iloc[i,1] == elective2)or (preq.iloc[i,1] == elective3)):
            problem.addConstraint(func1, [preq.iloc[i,0], preq.iloc[i,1]])
    
    solutions = problem.getSolutions()
    # Returning all possible solutions in each iteration
    return solutions

# Initializing an array variable to store the returned solutions        
total_solutions = []
#initializing variables for capturing prerequisites within electives
prec_elec = []
succ_elec = []
# capturing prerequisites within electives
for z in range(len(preq)):
    if ((preq.iloc[z,1] in electives)and(preq.iloc[z,0] in electives)):
        prec_elec.append(preq.iloc[z,0])
        succ_elec.append(preq.iloc[z,1])
#Passing the VALID elective combinations to the 'courses' function and appending the solutions to "total_solutions" array
for i in range(len(electives)):
    for j in range((i+1), (len(electives))):
        for k in range((j+1), (len(electives))):
            valid = "Yes"
            for q in range(len(prec_elec)):
                if (((electives[j]==succ_elec[q])and(electives[i]!=prec_elec[q]) and (electives[k]!=prec_elec[q]))or
                    ((electives[i]==succ_elec[q])and(electives[j]!=prec_elec[q]) and (electives[k]!=prec_elec[q]))or 
                    ((electives[k]==succ_elec[q])and(electives[i]!=prec_elec[q]) and (electives[j]!=prec_elec[q]))):
                    valid = "No"
            if(valid=="Yes"):
                solutions = courses(electives[i], electives[j],electives[k])
                for sol in solutions: 
                    total_solutions.append(sol)


print("CLASS: Artificial Intelligence, Lewis University")
print("NAME: Prashamsh Takkalapally")
print ('')
print('START TERM = Year 1 Fall 1')
print ('Number of Possible Degree Plans is ', len(total_solutions))
print ('')
print("Sample Degree Plan")

#Picking one out of 9488 solutions available. The # can be any between 0-9487
sample = total_solutions[9487]
#Capturing not_taken courses from electives in the selected sample output
not_taken = electives - sample.keys()
for cour in not_taken:
    # Printing "Not Taken" courses
    print("%-15s %15s" %('Not Taken', cour))
sam_sol = pd.Series(sample)
sol = sam_sol.sort_values()
for m in range(len(sol.keys())):
    if(sol.keys()[m]):
        #Printing terms and their associated courses in order
        print("%-15s %15s" %(all_terms[m+1], sol.keys()[m]))

    
    
    
 
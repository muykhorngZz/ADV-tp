# Advanced Programming for Data Science

Welcome to my repository for **Advanced Programming for Data Science**.

This repository is used to organize my learning materials, practical exercises, assignments, coding practice, datasets, notes, and projects throughout the course.

---

## Course Information

| Item | Description |
|---|---|
| **Course** | Advanced Programming for Data Science |
| **Main Language** | Python |
| **Area** | Programming + Data Science |
| **Main Focus** | Write better programs and apply them to real data problems |
| **Repository Purpose** | Keep my course work organized and track my progress |

The goal of this course is not only to learn more Python syntax.  
I also want to learn how to write code that is:

- Clear
- Reusable
- Efficient
- Easy to test
- Easy to debug
- Easy to maintain
- Suitable for Data Science projects

---

# My Learning Goals

By the end of this course, I want to be able to:

1. Write clean and readable Python code.
2. Understand advanced Python concepts.
3. Organize a Python project properly.
4. Create reusable functions, modules, and classes.
5. Apply Object-Oriented Programming.
6. Work with files and different data formats.
7. Process data with NumPy and Pandas.
8. Clean and transform datasets.
9. Visualize data clearly.
10. Debug programs systematically.
11. Test my code before using it in a project.
12. Compare different solutions based on performance.
13. Apply programming concepts to Data Science problems.
14. Build a complete small Data Science application independently.

---

# 🗂️ Repository Organization

```text
Advanced-Programming-for-Data-Science/
│
├── README.md
│
├── lessons/
│   ├── week_01/
│   ├── week_02/
│   ├── week_03/
│   └── ...
│
├── tp_labs/
│   ├── tp_01/
│   ├── tp_02/
│   └── ...
│
├── assignments/
│   ├── assignment_01/
│   ├── assignment_02/
│   └── ...
│
├── homework/
│   ├── hw_01/
│   ├── hw_02/
│   └── ...
│
├── practice/
│   ├── python/
│   ├── numpy/
│   ├── pandas/
│   ├── oop/
│   └── algorithms/
│
├── datasets/
│
├── projects/
│   ├── mini_project/
│   └── final_project/
│
├── notes/
│
└── exam_review/
```

---

# Main Topics I Expect to Study

## 1. Python Review

Before moving to advanced topics, I need to be comfortable with:

- Variables
- Data types
- Lists
- Tuples
- Dictionaries
- Sets
- Conditions
- Loops
- Functions
- Input and output

Example:

```python
def average(values):
    return sum(values) / len(values)

scores = [70, 80, 90]
print(average(scores))
```

Output:

```text
80.0
```

---

## 2. Advanced Python

Topics may include:

- List comprehensions
- Dictionary comprehensions
- Lambda functions
- Iterators
- Generators
- Decorators
- Modules
- Packages
- Context managers

My goal is to understand not only **how** these features work, but also **when** they are useful.

---

## 3. Object-Oriented Programming

Important concepts:

```text
Class
Object
Attribute
Method
Constructor
Encapsulation
Inheritance
Polymorphism
```

Example:

```python
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

    def show_result(self):
        print(f"{self.name}: {self.score}")


student = Student("Student A", 85)
student.show_result()
```

---

## 4. Functions, Modules, and Reusable Code

I want to avoid putting everything inside one large Python file.

Instead of:

```text
main.py
```

I should be able to organize a project like this:

```text
project/
│
├── main.py
├── data_loader.py
├── preprocessing.py
├── analysis.py
├── model.py
└── utils.py
```

This makes the project easier to understand and maintain.

---

## 5. File Handling

I will practice working with files such as:

- `.txt`
- `.csv`
- `.json`
- `.xlsx`

Example:

```python
with open("notes.txt", "r") as file:
    text = file.read()

print(text)
```

---

## 6. Exception Handling

Programs should be able to handle errors safely.

Important keywords:

```text
try
except
else
finally
raise
```

Example:

```python
try:
    value = int(input("Enter a number: "))
    print(100 / value)

except ValueError:
    print("Please enter a valid integer.")

except ZeroDivisionError:
    print("The number cannot be zero.")
```

---

# 🔢 NumPy

NumPy is important for numerical computing.

Topics I want to practice:

- Arrays
- Dimensions
- Shape
- Indexing
- Slicing
- Broadcasting
- Vector operations
- Matrix operations
- Statistical functions

Example:

```python
import numpy as np

values = np.array([5, 10, 15, 20])

print(np.mean(values))
print(np.max(values))
```

---

# 🐼 Pandas

Pandas is one of the main tools for working with structured data.

Topics:

- Series
- DataFrame
- Reading CSV files
- Selecting columns
- Filtering rows
- Missing values
- Sorting
- GroupBy
- Merge
- Join
- Aggregation

Example:

```python
import pandas as pd

df = pd.read_csv("data.csv")

print(df.head())
print(df.info())
print(df.describe())
```

---

# 🧹 Data Cleaning

A typical cleaning process may look like this:

```text
Raw Data
   ↓
Understand the Data
   ↓
Check Missing Values
   ↓
Remove Duplicates
   ↓
Fix Data Types
   ↓
Handle Outliers
   ↓
Clean Text / Categories
   ↓
Final Clean Dataset
```

Important tasks:

- Missing values
- Duplicate rows
- Wrong data types
- Outliers
- String cleaning
- Date conversion
- Category formatting

---

# 📊 Data Visualization

I will practice using tools such as:

```python
import matplotlib.pyplot as plt
```

Common visualizations:

- Line chart
- Bar chart
- Histogram
- Scatter plot
- Box plot

Before creating a graph, I should ask:

1. What question am I trying to answer?
2. Which variables do I need?
3. Which chart makes the result easiest to understand?

---

# 🐞 Debugging

When my program has an error, I should not immediately rewrite everything.

My debugging workflow:

```text
Read the error message
        ↓
Locate the problem
        ↓
Understand the error type
        ↓
Check variable values
        ↓
Fix one small part
        ↓
Run the program again
```

Common Python errors:

- `SyntaxError`
- `TypeError`
- `ValueError`
- `IndexError`
- `KeyError`
- `NameError`
- `FileNotFoundError`

---

# Testing

Testing helps confirm that my code behaves correctly.

Example:

```python
def multiply(a, b):
    return a * b

assert multiply(2, 3) == 6
assert multiply(5, 4) == 20
```

I want to build the habit of testing code instead of only checking whether it runs once.

---

# Code Performance

A correct solution is good, but I also want to understand whether it is efficient.

Things to compare:

- Execution time
- Memory usage
- Number of operations
- Code readability

Important concepts:

```text
Time Complexity
Space Complexity
Big O Notation
```

Common complexities:

```text
O(1)
O(log n)
O(n)
O(n log n)
O(n²)
```

---

# Data Science Programming Workflow

A simple workflow I can follow is:

```text
Define Problem
      ↓
Collect / Receive Data
      ↓
Load Data
      ↓
Explore Data
      ↓
Clean Data
      ↓
Transform Data
      ↓
Analyze Data
      ↓
Visualize Results
      ↓
Build Model if Needed
      ↓
Evaluate
      ↓
Explain Results
```

Programming supports every step of this workflow.

---

# TP / Lab Work

For every TP, I will try to record:

```text
TP:
Topic:
Objective:
Dataset:
Main Tasks:
Code:
Output:
Problems:
How I Fixed Them:
What I Learned:
```

### TP Checklist

- [ ] Read the instructions
- [ ] Understand the goal
- [ ] Write the code myself
- [ ] Run the program
- [ ] Check the output
- [ ] Fix errors
- [ ] Add useful comments
- [ ] Save the final version
- [ ] Review the TP later

---

# Assignments

For each assignment, I will organize my work using:

```text
Assignment:
Topic:
Deadline:
Requirements:
Dataset:
Approach:
Implementation:
Results:
Conclusion:
```

My assignment workflow:

```text
Understand the task
        ↓
Break it into smaller problems
        ↓
Prepare the data
        ↓
Write the solution
        ↓
Test the code
        ↓
Fix problems
        ↓
Interpret the result
        ↓
Write the explanation
        ↓
Submit
```

---

# Homework

For homework, I want to keep both my answer and my mistakes.

```text
Homework:
Date:
Topic:
Question:
My First Answer:
Correct Answer:
My Mistake:
What I Learned:
```

Mistakes are useful because they show me what I still need to review.

---

# My Study Method

For each new topic:

```text
1. Read the concept
2. Understand the idea
3. Look at a simple example
4. Write the code myself
5. Run it
6. Change the example
7. Create an error
8. Fix the error
9. Explain the code in my own words
10. Practice again
```

A topic is not fully learned just because the code runs.

I should be able to explain:

- What it does
- Why it works
- When to use it
- What can go wrong

---

# Weekly Routine

| Day | Main Activity |
|---|---|
| Monday | Review lecture notes |
| Tuesday | Practice examples |
| Wednesday | Work on TP / Lab |
| Thursday | Homework and revision |
| Friday | Assignment or project work |
| Saturday | Practice difficult topics |
| Sunday | Weekly review |

---

# Weekly Progress Checklist

At the end of each week:

- [ ] Lecture reviewed
- [ ] Notes updated
- [ ] Examples tested
- [ ] TP completed
- [ ] Homework completed
- [ ] Assignment progress checked
- [ ] Important errors recorded
- [ ] Difficult topics reviewed
- [ ] Extra practice completed

---

# Topic Note Template

For every important topic, I can use this format:

## Topic Name

### What is it?

Write a simple definition.

### Why is it useful?

Explain why programmers use it.

### When should I use it?

Give a real situation.

### Syntax

```python
# syntax
```

### Example

```python
# example
```

### Output

```text
# result
```

### Important Notes

Write the key points.

### Common Mistakes

Record errors I made.

### Practice

Create at least 2 small exercises.

---

# Progress Tracker

| Topic | Progress |
|---|---|
| Python Fundamentals | ⬜ Not Started |
| Advanced Python | ⬜ Not Started |
| NumPy | ⬜ Not Started |
| Pandas | ⬜ Not Started |
| OOP | ⬜ Not Started |
| File Handling | ⬜ Not Started |
| Exception Handling | ⬜ Not Started |
| Data Cleaning | ⬜ Not Started |
| Visualization | ⬜ Not Started |
| Debugging | ⬜ Not Started |
| Testing | ⬜ Not Started |
| Algorithms | ⬜ Not Started |
| Data Science Project | ⬜ Not Started |

Progress status:

```text
⬜ Not Started
🟡 In Progress
✅ Completed
```

---

# TP Tracker

| TP | Topic | Status | Reviewed |
|---|---|---|---|
| TP01 | TBD | ⬜ | ⬜ |
| TP02 | TBD | ⬜ | ⬜ |
| TP03 | TBD | ⬜ | ⬜ |
| TP04 | TBD | ⬜ | ⬜ |

---

# Assignment Tracker

| Assignment | Topic | Deadline | Status |
|---|---|---|---|
| Assignment 01 | TBD | TBD | ⬜ |
| Assignment 02 | TBD | TBD | ⬜ |
| Assignment 03 | TBD | TBD | ⬜ |

---

# Homework Tracker

| Homework | Topic | Status |
|---|---|---|
| HW01 | TBD | ⬜ |
| HW02 | TBD | ⬜ |
| HW03 | TBD | ⬜ |

---

# 📚 Exam Preparation

Before the exam, I should review both theory and coding.

## Theory

- Definitions
- Main concepts
- Differences between concepts
- Advantages and disadvantages
- When to use each technique

## Coding

Practice writing without copying:

```text
Conditions
Loops
Functions
Lists
Dictionaries
NumPy
Pandas
OOP
File Handling
Exception Handling
Data Cleaning
Testing
Algorithms
```

## Code Reading

I should practice answering:

```text
What does this code do?
What will be printed?
Why does this error happen?
How can this code be improved?
Is there a simpler solution?
```

---

# Final Goal

By the end of **Advanced Programming for Data Science**, I want to be able to take a real data problem and build a complete programming solution from beginning to end.

My target workflow is:

```text
Understand the Problem
        ↓
Organize the Project
        ↓
Write Clean Python Code
        ↓
Process the Data
        ↓
Analyze the Data
        ↓
Test and Debug
        ↓
Improve the Solution
        ↓
Explain the Result Clearly
```

> My final goal is not only to make the code work, but to understand the code well enough to build, improve, debug, and explain it independently.

---

## Personal Reminder

> Learn by writing code, making mistakes, fixing them, and explaining what I learned.


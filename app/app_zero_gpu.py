# git clone https://github.com/tuandung222/Small-Qwen-Coding-Multiple-Choice.git

import os
import sys

# run command to clone the repo
# os.system("git clone https://github.com/tuandung222/Small-Qwen-Coding-Multiple-Choice.git")

# run command to install the dependencies in the cloned repo / app/requirements.full.txt
# os.system("pip install -r Small-Qwen-Coding-Multiple-Choice/app/requirements.space.txt")

# https://github.com/tuandung222/Small-Qwen-Coding-Multiple-Choice/blob/main/app/requirements.space.txt
# run pip install from this link
os.system(
    "pip install -r https://github.com/tuandung222/Small-Qwen-Coding-Multiple-Choice/blob/main/app/requirements.space.txt"
)


# Add the parent directory to sys.path
# sys.path.append("Small-Qwen-Coding-Multiple-Choice")
# sys.path.append("Small-Qwen-Coding-Multiple-Choice/app")

import json
import os
import re
from typing import List, Optional, Union

import gradio as gr
import spaces
import torch

# import unsloth  # Import unsloth for optimized model loading
import yaml

# from examples import CODING_EXAMPLES, CODING_EXAMPLES_BY_CATEGORY

# from src.model.qwen_handler import QwenModelHandler
# from src.prompt_processors.prompt_creator import PromptCreator
# from src.prompt_processors.response_parser import ResponseParser


"""
Contains 200 example coding multiple choice questions for the demo application,
organized by category.
"""

# Define the examples by category
CODING_EXAMPLES_BY_CATEGORY = {
    "Python": [
        {
            "question": "Which of the following is NOT a valid way to iterate through a list in Python?",
            "choices": [
                "for item in my_list:",
                "for i in range(len(my_list)):",
                "for index, item in enumerate(my_list):",
                "for item from my_list:",
            ],
            "answer": "D",
        },
        {
            "question": "In Python, what does the `__str__` method do?",
            "choices": [
                "Returns a string representation of an object for developers",
                "Returns a string representation of an object for end users",
                "Converts a string to an object",
                "Checks if an object is a string",
            ],
            "answer": "B",
        },
        {
            "question": "What is the output of this Python code: `print(list(filter(lambda x: x % 2 == 0, range(10))))`?",
            "choices": [
                "[0, 2, 4, 6, 8]",
                "[1, 3, 5, 7, 9]",
                "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
                "[]",
            ],
            "answer": "A",
        },
        {
            "question": "What is the output of the following Python code?\nx = [1, 2, 3]\ny = x\ny.append(4)\nprint(x)",
            "choices": ["[1, 2, 3]", "[1, 2, 3, 4]", "[4, 1, 2, 3]", "Error"],
            "answer": "B",
        },
        {
            "question": "What is the correct way to check if two variables point to the same object in Python?",
            "choices": ["is", "==", "equals()", "==="],
            "answer": "A",
        },
        {
            "question": "What is the output of this Python code?\nprint(0.1 + 0.2 == 0.3)",
            "choices": ["False", "True", "Error", "None"],
            "answer": "A",
        },
        {
            "question": "In Python, what is a generator?",
            "choices": [
                "A function that returns an iterator",
                "A tool that creates new modules",
                "A class that generates random numbers",
                "A method for creating new objects",
            ],
            "answer": "A",
        },
        {
            "question": "In Python, what does the `*args` parameter do?",
            "choices": [
                "Allows a function to accept a variable number of positional arguments",
                "Makes arguments optional",
                "Multiplies all arguments",
                "Unpacks a dictionary into keyword arguments",
            ],
            "answer": "A",
        },
        {
            "question": "What is the output of `print(2 ** 3 ** 2)` in Python?",
            "choices": ["64", "36", "512", "None of the above"],
            "answer": "C",
        },
        {
            "question": "What does the `collections.Counter` class in Python do?",
            "choices": [
                "Counts occurrences of elements in an iterable",
                "Implements a countdown timer",
                "Tracks the number of function calls",
                "Counts the number of objects in memory",
            ],
            "answer": "A",
        },
        {
            "question": "What is the output of `print(list(zip([1, 2, 3], [4, 5, 6, 7])))`?",
            "choices": [
                "[(1, 4), (2, 5), (3, 6)]",
                "[(1, 4), (2, 5), (3, 6), (None, 7)]",
                "[(1, 4), (2, 5), (3, 6), (7, None)]",
                "Error",
            ],
            "answer": "A",
        },
        {
            "question": "What is a Python decorator?",
            "choices": [
                "A function that takes another function and extends its behavior",
                "A design pattern for creating objects",
                "A tool for formatting code",
                "A class for implementing UI elements",
            ],
            "answer": "A",
        },
        {
            "question": "What is the output of `print([i for i in range(10) if i % 2 == 0])`?",
            "choices": [
                "[0, 2, 4, 6, 8]",
                "[1, 3, 5, 7, 9]",
                "[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]",
                "[]",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of `__init__.py` files in Python packages?",
            "choices": [
                "To mark a directory as a Python package",
                "To initialize variables when a package is imported",
                "To document the package contents",
                "To define package-level constants",
            ],
            "answer": "A",
        },
        {
            "question": "What is the output of `print(sum(range(5)))`?",
            "choices": ["10", "15", "4", "Error"],
            "answer": "A",
        },
    ],
    "JavaScript": [
        {
            "question": "What is a closure in JavaScript?",
            "choices": [
                "A function that remembers its lexical scope",
                "A way to close browser windows",
                "A method to terminate functions",
                "A design pattern for security",
            ],
            "answer": "A",
        },
        {
            "question": "Which of these is NOT a JavaScript framework/library?",
            "choices": ["React", "Angular", "Django", "Vue"],
            "answer": "C",
        },
        {
            "question": "Which of these is a valid way to declare a constant in JavaScript?",
            "choices": [
                "const PI = 3.14",
                "constant PI = 3.14",
                "final PI = 3.14",
                "define PI = 3.14",
            ],
            "answer": "A",
        },
        {
            "question": 'What does the "this" keyword refer to in JavaScript?',
            "choices": [
                "The current object",
                "The parent object",
                "The global window object",
                "The function itself",
            ],
            "answer": "A",
        },
        {
            "question": "What is the difference between `==` and `===` in JavaScript?",
            "choices": [
                "`==` checks equality with type conversion, `===` checks equality without type conversion",
                "`==` checks reference equality, `===` checks value equality",
                "`==` is used for numbers, `===` is used for strings",
                "There is no difference",
            ],
            "answer": "A",
        },
        {
            "question": "What does the `async/await` feature do in JavaScript?",
            "choices": [
                "Simplifies asynchronous programming",
                "Creates multithreaded code",
                "Prevents memory leaks",
                "Optimizes rendering in browsers",
            ],
            "answer": "A",
        },
        {
            "question": "What is the main purpose of Redux in web development?",
            "choices": [
                "State management",
                "DOM manipulation",
                "Server-side rendering",
                "API communication",
            ],
            "answer": "A",
        },
        {
            "question": "Which of the following is NOT a primitive type in JavaScript?",
            "choices": ["number", "string", "boolean", "array"],
            "answer": "D",
        },
        {
            "question": "What is the output of `console.log(typeof null)` in JavaScript?",
            "choices": ["'object'", "'null'", "'undefined'", "Error"],
            "answer": "A",
        },
        {
            "question": "What is event bubbling in JavaScript?",
            "choices": [
                "When an event triggers on an element and then propagates up to parent elements",
                "When multiple events occur simultaneously",
                "When events are queued for later execution",
                "When events are canceled before execution",
            ],
            "answer": "A",
        },
        {
            "question": "What is the output of `console.log(1 + '2' + 3)` in JavaScript?",
            "choices": ["'123'", "6", "'33'", "Error"],
            "answer": "A",
        },
        {
            "question": "What is a JavaScript Promise?",
            "choices": [
                "An object representing a future completion or failure of an asynchronous operation",
                "A guarantee that a function will execute without errors",
                "A contract between different parts of an application",
                "A method for securing API endpoints",
            ],
            "answer": "A",
        },
    ],
    "SQL & Databases": [
        {
            "question": 'What does the SQL function "ROUND()" do?',
            "choices": [
                "Rounds a number to the nearest integer",
                "Concatenates two or more strings",
                "Converts a string to lowercase",
                "Returns the length of a string",
            ],
            "answer": "A",
        },
        {
            "question": "What does ACID stand for in database systems?",
            "choices": [
                "Atomicity, Consistency, Isolation, Durability",
                "Associativity, Commutativity, Identity, Distributivity",
                "Authentication, Cryptography, Integrity, Decentralization",
                "Availability, Consistency, Integration, Distribution",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of normalization in databases?",
            "choices": [
                "To reduce data redundancy and improve data integrity",
                "To improve query performance",
                "To encrypt sensitive data",
                "To compress data and save storage space",
            ],
            "answer": "A",
        },
        {
            "question": "Which SQL statement is used to retrieve data from a database?",
            "choices": ["SELECT", "FETCH", "GET", "RETRIEVE"],
            "answer": "A",
        },
        {
            "question": "What does the SQL command `GROUP BY` do?",
            "choices": [
                "Groups rows based on a column value",
                "Sorts rows in ascending order",
                "Filters rows based on a condition",
                "Joins two tables",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of the `HAVING` clause in SQL?",
            "choices": [
                "To filter groups that match a condition",
                "To join tables",
                "To select columns",
                "To sort results",
            ],
            "answer": "A",
        },
        {
            "question": "What is a primary key in a database?",
            "choices": [
                "A unique identifier for each record",
                "The first column in a table",
                "The fastest way to query data",
                "A mandatory field in every table",
            ],
            "answer": "A",
        },
        {
            "question": "What is the difference between `DELETE` and `TRUNCATE` in SQL?",
            "choices": [
                "`DELETE` can use WHERE condition, `TRUNCATE` removes all rows",
                "`DELETE` is faster, `TRUNCATE` is slower",
                "`DELETE` is for tables, `TRUNCATE` is for databases",
                "`DELETE` is permanent, `TRUNCATE` can be rolled back",
            ],
            "answer": "A",
        },
        {
            "question": "Which of these is a NoSQL database?",
            "choices": ["MongoDB", "MySQL", "PostgreSQL", "Oracle"],
            "answer": "A",
        },
        {
            "question": "What is a foreign key in a relational database?",
            "choices": [
                "A field that links to a primary key in another table",
                "A key used for encryption",
                "A key that must be unique across all tables",
                "A backup key used when the primary key fails",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of an index in a database?",
            "choices": [
                "To improve query performance",
                "To enforce data integrity",
                "To encrypt sensitive data",
                "To compress data storage",
            ],
            "answer": "A",
        },
    ],
    "Algorithms & Data Structures": [
        {
            "question": "What is the time complexity of binary search?",
            "choices": ["O(1)", "O(log n)", "O(n)", "O(n²)"],
            "answer": "B",
        },
        {
            "question": "Which data structure would be most efficient for implementing a priority queue?",
            "choices": ["Array", "Linked List", "Heap", "Stack"],
            "answer": "C",
        },
        {
            "question": "Which of these sorting algorithms has the worst worst-case time complexity?",
            "choices": ["Merge sort", "Quick sort", "Heap sort", "Bubble sort"],
            "answer": "D",
        },
        {
            "question": "In Big O notation, which of these is the most efficient?",
            "choices": ["O(n²)", "O(n log n)", "O(n)", "O(1)"],
            "answer": "D",
        },
        {
            "question": "What is a recursive function?",
            "choices": [
                "A function that calls itself",
                "A function that runs in the background",
                "A function that cannot be modified",
                "A function that returns multiple values",
            ],
            "answer": "A",
        },
        {
            "question": "Which algorithm is used for finding the shortest path in a weighted graph?",
            "choices": ["Dijkstra's algorithm", "Binary search", "Quicksort", "Depth-first search"],
            "answer": "A",
        },
        {
            "question": "Which of these sorting algorithms has the best average-case time complexity?",
            "choices": ["Merge Sort", "Bubble Sort", "Insertion Sort", "Selection Sort"],
            "answer": "A",
        },
        {
            "question": "Which data structure follows the LIFO (Last In First Out) principle?",
            "choices": ["Stack", "Queue", "Linked List", "Tree"],
            "answer": "A",
        },
        {
            "question": "What is the time complexity of inserting an element into a hash table?",
            "choices": ["O(1) average case", "O(log n)", "O(n)", "O(n²)"],
            "answer": "A",
        },
        {
            "question": "What is the space complexity of a recursive Fibonacci implementation?",
            "choices": ["O(n)", "O(log n)", "O(1)", "O(2^n)"],
            "answer": "A",
        },
        {
            "question": "What is the primary advantage of a B-tree over a binary search tree?",
            "choices": [
                "Better performance with disk-based storage",
                "Simpler implementation",
                "Lower memory usage",
                "Faster in-memory operations",
            ],
            "answer": "A",
        },
        {
            "question": "What is the worst-case time complexity of quicksort?",
            "choices": ["O(n²)", "O(n log n)", "O(n)", "O(log n)"],
            "answer": "A",
        },
        {
            "question": "Which data structure is most suitable for implementing a dictionary?",
            "choices": ["Hash Table", "Array", "Linked List", "Stack"],
            "answer": "A",
        },
        {
            "question": "What is the time complexity of breadth-first search on a graph?",
            "choices": ["O(V + E)", "O(V * E)", "O(log V)", "O(V²)"],
            "answer": "A",
        },
        {
            "question": "What is dynamic programming?",
            "choices": [
                "A method for solving complex problems by breaking them into simpler subproblems",
                "A programming paradigm that uses dynamic typing",
                "A technique for automatically allocating memory",
                "A method for optimizing code at runtime",
            ],
            "answer": "A",
        },
    ],
    "Web Development": [
        {
            "question": "Which of these is NOT a RESTful API method?",
            "choices": ["GET", "PUT", "SEARCH", "DELETE"],
            "answer": "C",
        },
        {
            "question": "What does CSS stand for?",
            "choices": [
                "Computer Style Sheets",
                "Creative Style System",
                "Cascading Style Sheets",
                "Colorful Style Sheets",
            ],
            "answer": "C",
        },
        {
            "question": "Which protocol is used for secure web browsing?",
            "choices": ["HTTP", "FTP", "HTTPS", "SMTP"],
            "answer": "C",
        },
        {
            "question": "In CSS, which property is used to change the text color of an element?",
            "choices": ["color", "text-color", "font-color", "text-style"],
            "answer": "A",
        },
        {
            "question": "What is the purpose of the `useState` hook in React?",
            "choices": [
                "To add state to functional components",
                "To create side effects in components",
                "To optimize rendering performance",
                "To handle form submissions",
            ],
            "answer": "A",
        },
        {
            "question": "What does API stand for?",
            "choices": [
                "Application Programming Interface",
                "Automated Program Interaction",
                "Application Process Integration",
                "Advanced Programming Implementation",
            ],
            "answer": "A",
        },
        {
            "question": "What is JWT used for?",
            "choices": [
                "Authentication and information exchange",
                "JavaScript web testing",
                "Java web toolkit",
                "JSON web transformation",
            ],
            "answer": "A",
        },
        {
            "question": "Which of these is NOT a valid HTTP status code?",
            "choices": [
                "200 OK",
                "404 Not Found",
                "500 Internal Server Error",
                "600 Server Timeout",
            ],
            "answer": "D",
        },
        {
            "question": "What is the purpose of CORS in web development?",
            "choices": [
                "To allow or restrict resources from being requested from another domain",
                "To optimize CSS rendering",
                "To compress HTTP responses",
                "To validate HTML syntax",
            ],
            "answer": "A",
        },
        {
            "question": "What is the difference between localStorage and sessionStorage?",
            "choices": [
                "localStorage persists after browser close, sessionStorage doesn't",
                "localStorage has a smaller storage limit than sessionStorage",
                "localStorage is encrypted, sessionStorage isn't",
                "localStorage is for text only, sessionStorage can store objects",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of a service worker in web development?",
            "choices": [
                "To enable offline functionality and background processing",
                "To manage server-side rendering",
                "To optimize database queries",
                "To handle user authentication",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of the `<meta viewport>` tag?",
            "choices": [
                "To control the viewport's size and scale for responsive design",
                "To improve SEO rankings",
                "To define metadata for social media sharing",
                "To specify the character encoding of the document",
            ],
            "answer": "A",
        },
    ],
    "Software Engineering & DevOps": [
        {
            "question": "Which design pattern is used when you need to create objects without specifying their concrete classes?",
            "choices": [
                "Observer Pattern",
                "Factory Pattern",
                "Singleton Pattern",
                "Decorator Pattern",
            ],
            "answer": "B",
        },
        {
            "question": 'In version control, what does "git rebase" do?',
            "choices": [
                "Integrates changes from one branch onto another",
                "Reverts all changes to the last commit",
                "Creates a new branch",
                "Deletes the remote repository",
            ],
            "answer": "A",
        },
        {
            "question": "What does the command `docker run` do?",
            "choices": [
                "Creates and starts a container",
                "Builds a Docker image",
                "Lists running containers",
                "Stops a running container",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of containerization technologies like Docker?",
            "choices": [
                "To package applications with all dependencies",
                "To create virtual machines",
                "To encrypt sensitive data",
                "To compress code for distribution",
            ],
            "answer": "A",
        },
        {
            "question": "What does MVC stand for in software architecture?",
            "choices": [
                "Model-View-Controller",
                "Multiple-Version-Control",
                "Most-Valuable-Component",
                "Managed-Virtual-Container",
            ],
            "answer": "A",
        },
        {
            "question": "What does the `git pull` command do?",
            "choices": [
                "Fetches changes from a remote repository and merges them",
                "Uploads local changes to a remote repository",
                "Creates a new branch",
                "Lists all commits",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of a load balancer?",
            "choices": [
                "Distributes network traffic across multiple servers",
                "Increases the speed of database queries",
                "Manages memory allocation in applications",
                "Compresses data before storage",
            ],
            "answer": "A",
        },
        {
            "question": "Which of these is NOT a principle of SOLID?",
            "choices": [
                "Single responsibility",
                "Open/closed",
                "Liskov substitution",
                "Dynamic typing",
            ],
            "answer": "D",
        },
        {
            "question": "What is the purpose of Continuous Integration (CI)?",
            "choices": [
                "To automatically build and test code changes",
                "To deploy applications to production",
                "To monitor application performance",
                "To manage database migrations",
            ],
            "answer": "A",
        },
        {
            "question": "What is the difference between CI and CD?",
            "choices": [
                "CI is about integration and testing; CD is about delivery or deployment",
                "CI is for code; CD is for databases",
                "CI is manual; CD is automated",
                "CI is for development; CD is for production only",
            ],
            "answer": "A",
        },
        {
            "question": "What is Infrastructure as Code (IaC)?",
            "choices": [
                "Managing infrastructure through code and automation",
                "Writing code that runs on multiple platforms",
                "Converting legacy systems to modern code",
                "Implementing code reviews for infrastructure teams",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of a Kubernetes pod?",
            "choices": [
                "The smallest deployable unit that can contain one or more containers",
                "A storage volume for container data",
                "A network interface for container communication",
                "A security policy for container isolation",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of a blue-green deployment?",
            "choices": [
                "To reduce downtime and risk by running two identical environments",
                "To separate development and production environments",
                "To implement color-coded security levels",
                "To optimize resource usage in cloud environments",
            ],
            "answer": "A",
        },
    ],
    "Programming Concepts": [
        {
            "question": "What is the result of `5 & 3` in binary operations?",
            "choices": ["1", "7", "8", "15"],
            "answer": "A",
        },
        {
            "question": "What is the purpose of the `static` keyword in Java?",
            "choices": [
                "It makes a variable or method belong to the class, not instances",
                "It prevents a class from being inherited",
                "It restricts access to a method or variable",
                "It makes a variable unchangeable",
            ],
            "answer": "A",
        },
        {
            "question": "In OOP, what is encapsulation?",
            "choices": [
                "The bundling of data and methods that operate on that data",
                "The ability of a class to inherit from multiple classes",
                "The hiding of implementation details",
                "The ability of objects to take different forms",
            ],
            "answer": "A",
        },
        {
            "question": "Which language is primarily used for iOS development?",
            "choices": ["Java", "Swift", "C#", "Kotlin"],
            "answer": "B",
        },
        {
            "question": "What is the difference between TCP and UDP?",
            "choices": [
                "TCP is connection-oriented; UDP is connectionless",
                "TCP is secure; UDP is not",
                "TCP is faster; UDP is more reliable",
                "TCP is for web; UDP is for email",
            ],
            "answer": "A",
        },
        {
            "question": "Which of these is NOT a primitive data type in Java?",
            "choices": ["int", "float", "String", "char"],
            "answer": "C",
        },
        {
            "question": "What is a memory leak?",
            "choices": [
                "Memory allocated that is never freed",
                "When a program uses too much memory",
                "When memory is corrupted by a virus",
                "When cache memory overflows",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of the `virtual` keyword in C++?",
            "choices": [
                "It allows a method to be overridden in derived classes",
                "It makes a class abstract",
                "It restricts a class from being instantiated",
                "It optimizes method calls at compile time",
            ],
            "answer": "A",
        },
        {
            "question": "What is the key feature of a blockchain?",
            "choices": [
                "Distributed immutable ledger",
                "Centralized data storage",
                "Fast transaction processing",
                "Unlimited scalability",
            ],
            "answer": "A",
        },
        {
            "question": "Which protocol is used for sending emails?",
            "choices": ["SMTP", "HTTP", "FTP", "SSH"],
            "answer": "A",
        },
        {
            "question": "What is the difference between a thread and a process?",
            "choices": [
                "Threads share memory space; processes have separate memory",
                "Threads run on multiple CPUs; processes run on a single CPU",
                "Threads are for I/O operations; processes are for computation",
                "Threads are managed by the application; processes by the OS",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of a mutex?",
            "choices": [
                "To ensure only one thread can access a resource at a time",
                "To speed up multi-threaded operations",
                "To allocate memory dynamically",
                "To compress data for transmission",
            ],
            "answer": "A",
        },
        {
            "question": "What is the difference between a stack and a heap in memory management?",
            "choices": [
                "Stack is for static memory allocation; heap is for dynamic allocation",
                "Stack is slower; heap is faster",
                "Stack is for global variables; heap is for local variables",
                "Stack is managed by the OS; heap by the application",
            ],
            "answer": "A",
        },
    ],
    "C & C++": [
        {
            "question": 'What is the output of this C code?\nint x = 5;\nprintf("%d", x++);\n',
            "choices": ["5", "6", "4", "Error"],
            "answer": "A",
        },
        {
            "question": "What is a pointer in C?",
            "choices": [
                "A variable that stores the address of another variable",
                "A variable that can point to multiple values",
                "A function that returns multiple values",
                "A special type of array",
            ],
            "answer": "A",
        },
        {
            "question": "What does the `const` keyword do in C++?",
            "choices": [
                "Declares that a variable or function cannot be modified",
                "Creates a constant expression",
                "Defines a compile-time constant",
                "All of the above",
            ],
            "answer": "D",
        },
        {
            "question": "What is the difference between `new` and `malloc` in C++?",
            "choices": [
                "`new` calls constructors, `malloc` doesn't",
                "`new` is faster than `malloc`",
                "`new` is for arrays, `malloc` is for single objects",
                "`new` is deprecated in modern C++",
            ],
            "answer": "A",
        },
        {
            "question": "What is the purpose of the `volatile` keyword in C?",
            "choices": [
                "Tells the compiler that a variable may change unexpectedly",
                "Makes a variable thread-safe",
                "Prevents a variable from being optimized",
                "Stores the variable in non-volatile memory",
            ],
            "answer": "A",
        },
        {
            "question": "What is a memory leak in C++?",
            "choices": [
                "When memory is allocated with `new` but not freed with `delete`",
                "When a program uses more memory than available",
                "When memory is corrupted by buffer overflow",
                "When memory is accessed after being freed",
            ],
            "answer": "A",
        },
        {
            "question": 'What is the output of this C code?\nint a = 10, b = 5;\nprintf("%d", a | b);\n',
            "choices": ["15", "0", "5", "10"],
            "answer": "A",
        },
        {
            "question": "What is the purpose of the `inline` keyword in C++?",
            "choices": [
                "Suggests that the compiler replace function calls with the function body",
                "Forces a function to be defined in a header file",
                "Makes a function thread-safe",
                "Prevents a function from being overridden",
            ],
            "answer": "A",
        },
        {
            "question": "What is the difference between a struct and a class in C++?",
            "choices": [
                "Members are public by default in struct, private in class",
                "Structs cannot have methods, classes can",
                "Structs are for POD types, classes for objects",
                "Structs cannot be inherited from, classes can",
            ],
            "answer": "A",
        },
    ],
}

# Flatten the examples for easy access by index
CODING_EXAMPLES = []
for category, examples in CODING_EXAMPLES_BY_CATEGORY.items():
    for example in examples:
        example["category"] = category
        CODING_EXAMPLES.append(example)


class PromptCreator:
    """
    Creates and formats prompts for multiple choice questions
    Supports different prompt styles for training and inference
    """

    # Prompt types
    BASIC = "basic"  # Simple answer-only format
    YAML_REASONING = "yaml"  # YAML formatted reasoning
    TEACHER_REASONED = (
        "teacher"  # Same YAML format as YAML_REASONING but using teacher completions for training
    )
    OPTIONS = "options"  # Includes only lettered options in prompt

    # Additional reasoning paradigms
    SOCRATIC = "socratic"  # Uses Socratic questioning to explore options
    SCIENTIST = "scientist"  # Uses scientific method and hypothesis testing
    LAWYER = "lawyer"  # Uses legal arguments and evidence evaluation
    DEBUGGER = "debugger"  # Uses programmer debugging methodology
    PHILOSOPHER = "philosopher"  # Uses philosophical analysis frameworks
    EXPERT_NOVICE = "expert_novice"  # Dialogues between expert and novice
    PROS_CONS = "pros_cons"  # Evaluates pros and cons for each option
    CODE_REVIEW = "code_review"  # Uses code review paradigm for code questions
    MATH_PROOF = "math_proof"  # Uses mathematical proof structure

    VALID_PROMPT_TYPES = [
        BASIC,
        YAML_REASONING,
        TEACHER_REASONED,
        OPTIONS,
        SOCRATIC,
        SCIENTIST,
        LAWYER,
        DEBUGGER,
        PHILOSOPHER,
        EXPERT_NOVICE,
        PROS_CONS,
        CODE_REVIEW,
        MATH_PROOF,
    ]

    def __init__(self, prompt_type: str = BASIC):
        """
        Initialize with specified prompt type

        Args:
            prompt_type: Type of prompt to use

        Raises:
            ValueError: If prompt_type is not one of the valid types
        """
        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. Must be one of {self.VALID_PROMPT_TYPES}"
            )

        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        # The difference only matters during training when using teacher completions
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING

        self.prompt_type = prompt_type
        # Store the original prompt type to track if we're using teacher mode
        self.original_type = prompt_type

    def format_choices(self, choices: Union[List[str], str]) -> str:
        """
        Format choices into a string

        Args:
            choices: List of choices or pre-formatted string

        Returns:
            Formatted string of choices

        Raises:
            ValueError: If choices is empty or invalid
        """
        if not choices:
            raise ValueError("Choices cannot be empty")

        if isinstance(choices, str):
            return choices

        if not isinstance(choices, list):
            raise ValueError(f"Choices must be a list or string, got {type(choices)}")

        if not all(isinstance(choice, str) for choice in choices):
            raise ValueError("All choices must be strings")

        return "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))

    def get_max_letter(self, choices: Union[List[str], str]) -> str:
        """
        Get the maximum letter for the given number of choices

        Args:
            choices: List of choices or pre-formatted string

        Returns:
            Maximum letter (A, B, C, etc.)

        Raises:
            ValueError: If choices is empty or invalid
        """
        if not choices:
            raise ValueError("Choices cannot be empty")

        if isinstance(choices, str):
            # Try to count the number of lines in the formatted string
            num_choices = len([line for line in choices.split("\n") if line.strip()])
            if num_choices == 0:
                raise ValueError("No valid choices found in string")
            return chr(64 + num_choices)

        if not isinstance(choices, list):
            raise ValueError(f"Choices must be a list or string, got {type(choices)}")

        if not all(isinstance(choice, str) for choice in choices):
            raise ValueError("All choices must be strings")

        return chr(64 + len(choices))

    def create_inference_prompt(self, question: str, choices: Union[List[str], str]) -> str:
        """
        Create a prompt for inference

        Args:
            question: The question text
            choices: List of choices or pre-formatted string

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If question or choices are empty or invalid
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        # Basic prompt types
        if self.prompt_type == self.BASIC:
            return self._create_basic_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type in [self.YAML_REASONING, self.TEACHER_REASONED]:
            return self._create_yaml_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.OPTIONS:
            return self._create_options_prompt(question, formatted_choices, max_letter)

        # Advanced reasoning paradigms
        elif self.prompt_type == self.SOCRATIC:
            return self._create_socratic_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.SCIENTIST:
            return self._create_scientist_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.LAWYER:
            return self._create_lawyer_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.DEBUGGER:
            return self._create_debugger_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.PHILOSOPHER:
            return self._create_philosopher_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.EXPERT_NOVICE:
            return self._create_expert_novice_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.PROS_CONS:
            return self._create_pros_cons_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.CODE_REVIEW:
            return self._create_code_review_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.MATH_PROOF:
            return self._create_math_proof_prompt(question, formatted_choices, max_letter)
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def _create_basic_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a basic prompt that only asks for the answer"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Answer with a single letter from A through {max_letter} without any additional explanation or commentary."""

    def _create_yaml_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a YAML-formatted prompt that asks for reasoning"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Your response MUST be in YAML format as follows:

understanding: |
  <your understanding of what the question is asking>
analysis: |
  <your analysis of each option>
reasoning: |
  <your step-by-step reasoning process>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>

The answer field MUST contain ONLY a single character letter."""

    def _create_options_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a prompt that focuses on lettered options"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Please select the best answer from the options above. Provide a brief explanation for your choice and clearly state the letter of your answer (A through {max_letter})."""

    def create_training_prompt(self, question: str, choices: Union[List[str], str]) -> str:
        """
        Create a prompt for training

        Args:
            question: The question text
            choices: List of choices or pre-formatted string

        Returns:
            Formatted prompt string

        Raises:
            ValueError: If question or choices are empty or invalid
        """
        if not question or not isinstance(question, str):
            raise ValueError("Question must be a non-empty string")

        formatted_choices = self.format_choices(choices)
        max_letter = self.get_max_letter(choices)

        # Basic prompt types
        if self.prompt_type == self.BASIC:
            return self._create_basic_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type in [self.YAML_REASONING, self.TEACHER_REASONED]:
            return self._create_yaml_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.OPTIONS:
            return self._create_options_training_prompt(question, formatted_choices, max_letter)

        # Advanced reasoning paradigms
        elif self.prompt_type == self.SOCRATIC:
            return self._create_socratic_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.SCIENTIST:
            return self._create_scientist_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.LAWYER:
            return self._create_lawyer_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.DEBUGGER:
            return self._create_debugger_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.PHILOSOPHER:
            return self._create_philosopher_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.EXPERT_NOVICE:
            return self._create_expert_novice_training_prompt(
                question, formatted_choices, max_letter
            )
        elif self.prompt_type == self.PROS_CONS:
            return self._create_pros_cons_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.CODE_REVIEW:
            return self._create_code_review_training_prompt(question, formatted_choices, max_letter)
        elif self.prompt_type == self.MATH_PROOF:
            return self._create_math_proof_training_prompt(question, formatted_choices, max_letter)
        else:
            raise ValueError(f"Unknown prompt type: {self.prompt_type}")

    def _create_basic_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a basic training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

The answer is a single letter (A, B, C, etc.). Only provide ONE character as your answer:"""

    def _create_yaml_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a YAML-formatted training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Analyze this question step-by-step and provide a detailed explanation.
Follow the YAML format in your response:

understanding: |
  <your understanding of the question>
analysis: |
  <your analysis of each option>
reasoning: |
  <your reasoning about the correct answer>
conclusion: |
  <your final conclusion>
answer: <single letter A through {max_letter}>"""

    def _create_options_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a training prompt for options format"""
        return f"""Question: {question}

Choices:
{formatted_choices}

Please select the best answer from the options above. Provide a brief explanation for your choice and clearly state the letter of your answer (A through {max_letter})."""

    def set_prompt_type(self, prompt_type: str) -> "PromptCreator":
        """
        Set the prompt type

        Args:
            prompt_type: Type of prompt to use (BASIC, YAML_REASONING, or TEACHER_REASONED)

        Returns:
            Self for method chaining

        Raises:
            ValueError: If prompt_type is not one of the valid types
        """
        if prompt_type not in self.VALID_PROMPT_TYPES:
            raise ValueError(
                f"Invalid prompt type: {prompt_type}. Must be one of {self.VALID_PROMPT_TYPES}"
            )

        # Store the original type
        self.original_type = prompt_type

        # For prompt formatting, teacher_reasoned is equivalent to yaml_reasoning
        if prompt_type == self.TEACHER_REASONED:
            prompt_type = self.YAML_REASONING

        self.prompt_type = prompt_type
        return self

    def is_teacher_mode(self) -> bool:
        """Check if using teacher-reasoned mode"""
        return self.original_type == self.TEACHER_REASONED

    # Advanced reasoning paradigm prompt methods

    def _create_socratic_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a Socratic dialogue prompt that leads through questioning"""
        return f"""Question: {question}

Choices:
{formatted_choices}

To solve this problem, I'll use Socratic questioning to examine each option:
1. What do I already know about this topic?
2. What assumptions am I making?
3. What evidence would prove or disprove each option?
4. What are the implications of each option?
5. Are there alternative perspectives I should consider?

After answering these questions for each option, I will conclude with my answer letter (A through {max_letter})."""

    def _create_socratic_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a Socratic dialogue training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

To solve this problem, I'll use Socratic questioning to examine each option:
1. What do I already know about this topic?
2. What assumptions am I making?
3. What evidence would prove or disprove each option?
4. What are the implications of each option?
5. Are there alternative perspectives I should consider?

After answering these questions for each option, I will conclude with my answer letter (A through {max_letter})."""

    def _create_scientist_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a scientific method prompt that tests each option as a hypothesis"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this using the scientific method:

1. Observation: Understanding what the question is asking
2. Hypothesis: Treating each option (A through {max_letter}) as a potential hypothesis
3. Testing: Analyzing the validity of each option
4. Analysis: Evaluating the evidence for each option
5. Conclusion: Determining which option is supported by the evidence

For my final answer, I'll clearly state which letter (A through {max_letter}) corresponds to the correct option."""

    def _create_scientist_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a scientific method training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this using the scientific method:

1. Observation: Understanding what the question is asking
2. Hypothesis: Treating each option (A through {max_letter}) as a potential hypothesis
3. Testing: Analyzing the validity of each option
4. Analysis: Evaluating the evidence for each option
5. Conclusion: Determining which option is supported by the evidence

For my final answer, I'll clearly state which letter (A through {max_letter}) corresponds to the correct option."""

    def _create_lawyer_prompt(self, question: str, formatted_choices: str, max_letter: str) -> str:
        """Create a legal argument prompt that evaluates evidence"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this question as a lawyer evaluating evidence:

FACTS:
- What are the key facts presented in the question?
- What established principles or knowledge apply to this situation?

ARGUMENTS:
- For each option (A through {max_letter}):
  - What arguments support this option?
  - What arguments oppose this option?
  - What is the strength of evidence for each?

RULING:
- Based on the weight of evidence, which option has the strongest case?
- Are there any reasonable doubts about my conclusion?

VERDICT:
My answer is option [letter A through {max_letter}]."""

    def _create_lawyer_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a legal argument training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this question as a lawyer evaluating evidence:

FACTS:
- What are the key facts presented in the question?
- What established principles or knowledge apply to this situation?

ARGUMENTS:
- For each option (A through {max_letter}):
  - What arguments support this option?
  - What arguments oppose this option?
  - What is the strength of evidence for each?

RULING:
- Based on the weight of evidence, which option has the strongest case?
- Are there any reasonable doubts about my conclusion?

VERDICT:
My answer is option [letter A through {max_letter}]."""

    def _create_debugger_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a debugging prompt that treats options as code paths"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll debug this problem systematically:

1. PROBLEM STATEMENT:
   - What is the core issue or question?
   - What is the expected correct behavior/answer?

2. TRACE EXECUTION:
   - For each option (A through {max_letter}):
     - If this option were correct, what logical steps would follow?
     - Are there any logical errors or edge cases in this option?

3. TEST CASES:
   - What examples can I think of to test each option?
   - Do any options fail under certain conditions?

4. ROOT CAUSE:
   - Which option correctly addresses the core problem?
   - Why do the other options fail?

5. FIX:
   - My answer is option [letter A through {max_letter}]
   - Explanation: [brief justification]"""

    def _create_debugger_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a debugging training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll debug this problem systematically:

1. PROBLEM STATEMENT:
   - What is the core issue or question?
   - What is the expected correct behavior/answer?

2. TRACE EXECUTION:
   - For each option (A through {max_letter}):
     - If this option were correct, what logical steps would follow?
     - Are there any logical errors or edge cases in this option?

3. TEST CASES:
   - What examples can I think of to test each option?
   - Do any options fail under certain conditions?

4. ROOT CAUSE:
   - Which option correctly addresses the core problem?
   - Why do the other options fail?

5. FIX:
   - My answer is option [letter A through {max_letter}]
   - Explanation: [brief justification]"""

    def _create_philosopher_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a philosophical analysis prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze this philosophically:

CONCEPTUAL ANALYSIS:
- What are the key concepts in this question?
- How might different philosophical frameworks interpret these concepts?

LOGICAL STRUCTURE:
- What is the logical form of each option?
- Are there any logical fallacies or contradictions in the options?

THOUGHT EXPERIMENT:
- What hypothetical scenarios could test the validity of each option?
- What would be the implications if each option were true?

SYNTHESIS:
- Which option best aligns with sound reasoning?
- What might opponents of this view argue?

CONCLUSION:
Therefore, the correct answer is option [letter A through {max_letter}]."""

    def _create_philosopher_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a philosophical analysis training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze this philosophically:

CONCEPTUAL ANALYSIS:
- What are the key concepts in this question?
- How might different philosophical frameworks interpret these concepts?

LOGICAL STRUCTURE:
- What is the logical form of each option?
- Are there any logical fallacies or contradictions in the options?

THOUGHT EXPERIMENT:
- What hypothetical scenarios could test the validity of each option?
- What would be the implications if each option were true?

SYNTHESIS:
- Which option best aligns with sound reasoning?
- What might opponents of this view argue?

CONCLUSION:
Therefore, the correct answer is option [letter A through {max_letter}]."""

    def _create_expert_novice_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a dialogue between expert and novice"""
        return f"""Question: {question}

Choices:
{formatted_choices}

NOVICE: I'm trying to solve this multiple-choice question but I'm not sure how to approach it. Can you help me?

EXPERT: Of course! Let's break it down step by step. First, let's understand what the question is asking.

NOVICE: Okay, so the question is asking about [{question}]. And there are {max_letter - 64} possible answers.

EXPERT: That's right. Let's analyze each option one by one:

[Analysis of each option]

NOVICE: That makes sense. So which option do you think is correct?

EXPERT: Based on our analysis, I believe the correct answer is option [letter A through {max_letter}] because [explanation].

NOVICE: Got it! So the answer is [letter A through {max_letter}]."""

    def _create_expert_novice_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create an expert-novice dialogue training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

NOVICE: I'm trying to solve this multiple-choice question but I'm not sure how to approach it. Can you help me?

EXPERT: Of course! Let's break it down step by step. First, let's understand what the question is asking.

NOVICE: Okay, so the question is asking about [{question}]. And there are {max_letter - 64} possible answers.

EXPERT: That's right. Let's analyze each option one by one:

[Analysis of each option]

NOVICE: That makes sense. So which option do you think is correct?

EXPERT: Based on our analysis, I believe the correct answer is option [letter A through {max_letter}] because [explanation].

NOVICE: Got it! So the answer is [letter A through {max_letter}]."""

    def _create_pros_cons_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a pros and cons analysis prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze each option by listing its pros and cons:

OPTION A:
- Pros: [list of advantages]
- Cons: [list of disadvantages]

[Continue for all options through {max_letter}]

DECISION MATRIX:
- Option with most pros: ?
- Option with fewest cons: ?
- Option with best overall balance: ?

CONCLUSION:
After weighing the pros and cons of each option, the answer is [letter A through {max_letter}]."""

    def _create_pros_cons_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a pros and cons analysis training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll analyze each option by listing its pros and cons:

OPTION A:
- Pros: [list of advantages]
- Cons: [list of disadvantages]

[Continue for all options through {max_letter}]

DECISION MATRIX:
- Option with most pros: ?
- Option with fewest cons: ?
- Option with best overall balance: ?

CONCLUSION:
After weighing the pros and cons of each option, the answer is [letter A through {max_letter}]."""

    def _create_code_review_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a code review prompt for programming questions"""
        return f"""Question: {question}

Choices:
{formatted_choices}

CODE REVIEW PROCESS:

REQUIREMENTS ANALYSIS:
- What is the expected behavior/output?
- What are the constraints or edge cases to consider?

CODE INSPECTION:
- For each option (A through {max_letter}):
  - Is the syntax correct?
  - Are there any potential bugs or edge cases?
  - Does it follow best practices?
  - Is it efficient and maintainable?

TESTING PERSPECTIVE:
- What test cases would validate or invalidate each option?
- How would each option handle those tests?

REVIEWER FEEDBACK:
Based on this review, option [letter A through {max_letter}] is the most correct because [explanation]."""

    def _create_code_review_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a code review training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

CODE REVIEW PROCESS:

REQUIREMENTS ANALYSIS:
- What is the expected behavior/output?
- What are the constraints or edge cases to consider?

CODE INSPECTION:
- For each option (A through {max_letter}):
  - Is the syntax correct?
  - Are there any potential bugs or edge cases?
  - Does it follow best practices?
  - Is it efficient and maintainable?

TESTING PERSPECTIVE:
- What test cases would validate or invalidate each option?
- How would each option handle those tests?

REVIEWER FEEDBACK:
Based on this review, option [letter A through {max_letter}] is the most correct because [explanation]."""

    def _create_math_proof_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a mathematical proof structure prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this as a mathematical proof:

GIVEN:
- Information provided in the question
- Mathematical principles relevant to this problem

PROVE:
- I need to determine which option (A through {max_letter}) is correct

PROOF:
1. First, I'll establish the key principles needed
2. For each option:
   - Assume the option is true
   - Determine if this leads to a valid result or a contradiction
   - Note any special cases or conditions

CONCLUSION:
Therefore, option [letter A through {max_letter}] is proven to be correct.
"""

    def _create_math_proof_training_prompt(
        self, question: str, formatted_choices: str, max_letter: str
    ) -> str:
        """Create a mathematical proof training prompt"""
        return f"""Question: {question}

Choices:
{formatted_choices}

I'll approach this as a mathematical proof:

GIVEN:
- Information provided in the question
- Mathematical principles relevant to this problem

PROVE:
- I need to determine which option (A through {max_letter}) is correct

PROOF:
1. First, I'll establish the key principles needed
2. For each option:
   - Assume the option is true
   - Determine if this leads to a valid result or a contradiction
   - Note any special cases or conditions

CONCLUSION:
Therefore, option [letter A through {max_letter}] is proven to be correct.
"""


import re
from typing import Any, Dict, Optional, Tuple

import yaml

try:
    from .prompt_creator import PromptCreator
except ImportError:
    pass


class ResponseParser:
    """
    Parser for model responses with support for different formats
    Extracts answers and reasoning from model outputs
    """

    # Parser modes
    BASIC = "basic"  # Extract single letter answer
    YAML = "yaml"  # Parse YAML formatted response with reasoning

    def __init__(self, parser_mode: str = BASIC):
        """
        Initialize with specified parser mode

        Args:
            parser_mode: Mode of parsing to use (BASIC or YAML)
        """
        if parser_mode not in [self.BASIC, self.YAML]:
            raise ValueError(f"Unknown parser mode: {parser_mode}")
        self.parser_mode = parser_mode

    def parse(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse the response text to extract answer and reasoning

        Args:
            response_text: Raw response text from the model

        Returns:
            Tuple of (answer, reasoning)
        """
        if not response_text:
            return None, None

        if self.parser_mode == self.BASIC:
            return self._parse_basic_response(response_text)
        elif self.parser_mode == self.YAML:
            return self._parse_yaml_response(response_text)
        else:
            raise ValueError(f"Unknown parser mode: {self.parser_mode}")

    def _parse_basic_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse basic response format (just the answer)"""
        # Clean up the response
        response_text = response_text.strip()

        # Try to find a single letter answer
        answer_match = re.search(r"^[A-Za-z]$", response_text)
        if answer_match:
            return answer_match.group(0).upper(), None

        # Try to find answer after "Answer:" or similar
        answer_match = re.search(r"(?:answer|Answer):\s*([A-Za-z])", response_text)
        if answer_match:
            return answer_match.group(1).upper(), None

        # Try to find any single letter in the response
        answer_match = re.search(r"[A-Za-z]", response_text)
        if answer_match:
            return answer_match.group(0).upper(), None

        return None, None

    def _parse_yaml_response(self, response_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse YAML-formatted response with reasoning"""
        # Clean up the response
        response_text = response_text.strip()

        # Remove any markdown code block markers
        response_text = re.sub(r"```yaml\s*", "", response_text)
        response_text = re.sub(r"```\s*", "", response_text)

        try:
            # Try to parse as YAML
            yaml_content = yaml.safe_load("---\n" + response_text)
            if isinstance(yaml_content, dict):
                answer = yaml_content.get("answer")
                reasoning = self._extract_reasoning_from_yaml(yaml_content)

                # Clean up answer if needed
                if answer:
                    answer = answer.strip().upper()
                    if len(answer) > 1:
                        # Extract first letter if multiple characters
                        answer = answer[0]

                return answer, reasoning
        except yaml.YAMLError:
            # If YAML parsing fails, try to extract using regex
            answer_match = re.search(r"answer:\s*([A-Za-z])", response_text)
            reasoning_match = re.search(r"reasoning:\s*\|\s*([\s\S]+?)(?:\n\w+:|$)", response_text)

            answer = answer_match.group(1).upper() if answer_match else None
            reasoning = reasoning_match.group(1).strip() if reasoning_match else None

            return answer, reasoning

        return None, None

    def _extract_reasoning_from_yaml(self, yaml_content: Dict[str, Any]) -> Optional[str]:
        """Extract and format reasoning from YAML content"""
        reasoning_parts = []

        # Add understanding if present
        if "understanding" in yaml_content:
            reasoning_parts.append(f"Understanding:\n{yaml_content['understanding']}")

        # Add analysis if present
        if "analysis" in yaml_content:
            reasoning_parts.append(f"Analysis:\n{yaml_content['analysis']}")

        # Add reasoning if present
        if "reasoning" in yaml_content:
            reasoning_parts.append(f"Reasoning:\n{yaml_content['reasoning']}")

        # Add conclusion if present
        if "conclusion" in yaml_content:
            reasoning_parts.append(f"Conclusion:\n{yaml_content['conclusion']}")

        return "\n\n".join(reasoning_parts) if reasoning_parts else None

    def set_parser_mode(self, parser_mode: str) -> "ResponseParser":
        """Set the parser mode"""
        if parser_mode not in [self.BASIC, self.YAML]:
            raise ValueError(f"Unknown parser mode: {parser_mode}")
        self.parser_mode = parser_mode
        return self

    @classmethod
    def from_prompt_type(cls, prompt_type: str) -> "ResponseParser":
        """
        Create a ResponseParser instance from a prompt type

        Args:
            prompt_type: Type of prompt (from PromptCreator)

        Returns:
            ResponseParser instance with appropriate mode
        """
        if prompt_type == PromptCreator.BASIC:
            return cls(cls.BASIC)
        elif prompt_type in [PromptCreator.YAML_REASONING, PromptCreator.TEACHER_REASONED]:
            return cls(cls.YAML)
        else:
            raise ValueError(f"Unknown prompt type: {prompt_type}")


import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

try:
    import unsloth  # Import unsloth first to apply all optimizations and avoid warnings
except ImportError:
    pass
from transformers import AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizer

try:
    from unsloth import FastLanguageModel
    from unsloth.chat_templates import train_on_responses_only
except ImportError:
    pass

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ModelSource(str, Enum):
    """Model source enumeration"""

    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"


@dataclass
class HubConfig:
    """Configuration for Hugging Face Hub integration"""

    model_id: str
    token: Optional[str] = None
    private: bool = False
    save_method: str = "lora"  # lora, merged_16bit, merged_4bit, gguf


class QwenModelHandler:
    """Handles loading, configuration, and inference with Qwen models"""

    HUGGINGFACE = "huggingface"
    UNSLOTH = "unsloth"

    def __init__(
        self,
        model_name: str,
        max_seq_length: int = 2048,
        quantization: Union[str, BitsAndBytesConfig] = "4bit",
        model_source: str = ModelSource.HUGGINGFACE,
        device_map: str = "auto",
        source_hub_config: Optional[HubConfig] = None,
        destination_hub_config: Optional[HubConfig] = None,
        attn_implementation: str = "default",
        force_attn_implementation: bool = False,
    ):
        """
        Initialize a Qwen model handler.

        Args:
            model_name: Name or path of the model to load
            max_seq_length: Maximum sequence length for tokenizer and model
            quantization: Quantization level (4bit, 8bit, or none) or BitsAndBytesConfig object
            model_source: Source of the model (huggingface or unsloth)
            device_map: Device mapping strategy for the model
            source_hub_config: Configuration for the source model on Hugging Face Hub
            destination_hub_config: Configuration for the destination model on Hugging Face Hub
            attn_implementation: Attention implementation to use (default, flash_attention_2, sdpa, eager, xformers)
            force_attn_implementation: Whether to force the attention implementation even if not optimal
        """
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.quantization = quantization
        self.model_source = model_source
        self.device_map = device_map
        self.source_hub_config = source_hub_config
        self.destination_hub_config = destination_hub_config
        self.attn_implementation = attn_implementation
        self.force_attn_implementation = force_attn_implementation

        # Initialize model and tokenizer
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        # Log model configuration
        logger.info(f"Loading {model_name} from {model_source}, max_seq_length={max_seq_length}")

        # Load the model based on the source
        self._load_model()

    def _check_attention_support(self):
        """Check if the specified attention implementation is supported on the current hardware"""
        # Check for Flash Attention 2 support
        has_flash_attn = False
        try:
            import flash_attn

            has_flash_attn = True
            logger.info("Flash Attention 2 is available (package flash-attn detected)")
            # Check flash_attn version
            try:
                logger.info(f"Flash Attention 2 version: {flash_attn.__version__}")
            except AttributeError:
                logger.info("Flash Attention 2 version information not available")
        except ImportError:
            logger.info("Flash Attention 2 is not available (package flash-attn not found)")
            if self.attn_implementation == "flash_attention_2":
                logger.info("To install: pip install flash-attn --no-build-isolation")

        # Check for xFormers support
        has_xformers = False
        try:
            import xformers

            has_xformers = True
            try:
                logger.info(f"xFormers is available (version: {xformers.__version__})")
            except AttributeError:
                logger.info("xFormers is available (version information not available)")
        except ImportError:
            logger.info("xFormers is not available (package not found)")
            if self.attn_implementation == "xformers":
                logger.info("To install: pip install xformers")

        # Check for CUDA availability for SDPA
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            try:
                cuda_version = torch.version.cuda
                logger.info(f"CUDA is available (version: {cuda_version})")
                # Check if CUDA version is sufficient for SDPA
                if self.attn_implementation == "sdpa" and cuda_version:
                    major, minor = map(int, cuda_version.split(".")[:2])
                    if major < 11 or (major == 11 and minor < 6):
                        logger.warning(f"SDPA works best with CUDA 11.6+, current: {cuda_version}")
            except:
                logger.info("CUDA is available (version information not available)")
        else:
            logger.info("CUDA is not available")

        # Check PyTorch version for SDPA
        if self.attn_implementation == "sdpa":
            from packaging import version

            torch_version = torch.__version__
            if version.parse(torch_version) < version.parse("2.0.0"):
                logger.warning(f"SDPA requires PyTorch 2.0+, current: {torch_version}")
                if not self.force_attn_implementation:
                    logger.warning("Falling back to default attention implementation")
                    return "default"

        # Return available implementations
        if self.attn_implementation == "flash_attention_2" and not has_flash_attn:
            if self.force_attn_implementation:
                logger.warning(
                    "Flash Attention 2 was requested but is not available. Forcing may cause errors."
                )
            else:
                logger.warning(
                    "Flash Attention 2 was requested but is not available. Falling back to default."
                )
                return "default"

        if self.attn_implementation == "xformers" and not has_xformers:
            if self.force_attn_implementation:
                logger.warning(
                    "xFormers was requested but is not available. Forcing may cause errors."
                )
            else:
                logger.warning(
                    "xFormers was requested but is not available. Falling back to default."
                )
                return "default"

        if self.attn_implementation == "sdpa" and not has_cuda:
            if self.force_attn_implementation:
                logger.warning(
                    "SDPA was requested but CUDA is not available. Forcing may cause errors."
                )
            else:
                logger.warning(
                    "SDPA was requested but CUDA is not available. Falling back to default."
                )
                return "default"

        logger.info(f"Using attention implementation: {self.attn_implementation}")
        return self.attn_implementation

    def _load_model(self):
        """Load the model and tokenizer based on the specified source"""
        try:
            if self.model_source == ModelSource.UNSLOTH:
                self._load_from_unsloth()
            else:
                self._load_from_huggingface()

            # Ensure tokenizer has pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Log model info
            logger.info(f"Model loaded successfully: {self.model_name}")
            if hasattr(self.model, "config"):
                logger.info(f"Model type: {self.model.config.model_type}")
                for key, value in self.model.config.to_dict().items():
                    if key in [
                        "hidden_size",
                        "intermediate_size",
                        "num_attention_heads",
                        "num_hidden_layers",
                        "torch_dtype",
                    ]:
                        logger.info(f"{key}: {value}")

        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def _load_from_huggingface(self):
        """Load model from HuggingFace Hub"""
        # Configure quantization
        quantization_config = None
        if isinstance(self.quantization, str):
            if self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            elif self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        elif isinstance(self.quantization, BitsAndBytesConfig):
            quantization_config = self.quantization

        # Check attention implementation
        attn_implementation = self._check_attention_support()

        model_kwargs = {
            "device_map": self.device_map,
            "token": self.source_hub_config.token if self.source_hub_config else None,
            "trust_remote_code": True,
        }

        # Add quantization config if specified
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        # Add attention implementation if not default
        if attn_implementation != "default":
            model_kwargs["attn_implementation"] = attn_implementation
            logger.info(f"Using attention implementation: {attn_implementation}")

        # Import AutoModelForCausalLM here to avoid early import
        from transformers import AutoModelForCausalLM

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=self.source_hub_config.token if self.source_hub_config else None,
            trust_remote_code=True,
            padding_side="right",
            model_max_length=self.max_seq_length,
        )

    def _load_from_unsloth(self):
        """Load model with Unsloth optimization"""
        try:
            # Import unsloth here to avoid early import
            from unsloth import FastLanguageModel

            # Check attention implementation
            attn_implementation = self._check_attention_support()

            # Determine max memory
            max_memory = None
            if torch.cuda.is_available():
                # Use 85% of available GPU memory
                max_memory = {
                    0: f"{int(torch.cuda.get_device_properties(0).total_memory * 0.85 / 1024 / 1024)}MiB"
                }
                logger.info(f"Setting max memory: {max_memory}")

            # Setup model args
            model_args = {
                "max_seq_length": self.max_seq_length,
                "device_map": self.device_map,
            }

            # Add quantization config
            if isinstance(self.quantization, str):
                if self.quantization == "4bit":
                    model_args["load_in_4bit"] = True
                elif self.quantization == "8bit":
                    model_args["load_in_8bit"] = True
            elif isinstance(self.quantization, BitsAndBytesConfig):
                if self.quantization.load_in_4bit:
                    model_args["load_in_4bit"] = True
                elif self.quantization.load_in_8bit:
                    model_args["load_in_8bit"] = True

            # Add attention implementation if not default
            if attn_implementation != "default":
                model_args["attn_implementation"] = attn_implementation
                logger.info(f"Using attention implementation: {attn_implementation}")

            # Load model and tokenizer
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.model_name,
                token=self.source_hub_config.token if self.source_hub_config else None,
                max_memory=max_memory,
                **model_args,
            )

        except ImportError:
            logger.error("Unsloth import failed. Please install unsloth with: pip install unsloth")
            raise

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
    ) -> str:
        """
        Generate a response from the model.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling or greedy generation

        Returns:
            str: The generated text response
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode the output
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt from the response
        response = response[len(prompt) :].strip()
        return response

    def generate_with_streaming(
        self,
        prompt: str,
        max_new_tokens: int = 768,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.0,
        do_sample: bool = True,
        min_p: float = 0.1,
        stream: bool = True,
    ):
        """
        Generate a response from the model with streaming support.

        Args:
            prompt: The input prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling probability
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeated tokens
            do_sample: Whether to use sampling or greedy generation
            min_p: Minimum probability for sampling (recommended 0.1)
            stream: Whether to stream the output or return the full response

        Returns:
            If stream=True: TextIteratorStreamer object that yields tokens as they're generated
            If stream=False: Complete response as string
        """
        import threading

        from transformers import TextIteratorStreamer

        try:
            from unsloth import FastLanguageModel
        except ImportError:
            pass

        # Enable faster inference if using Unsloth
        if self.model_source == ModelSource.UNSLOTH:
            try:
                FastLanguageModel.for_inference(self.model)
            except ImportError:
                pass

        # Format the prompt using chat template
        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)

        # Create attention mask
        attention_mask = torch.ones_like(inputs)

        if stream:
            # Use TextIteratorStreamer for streaming output
            streamer = TextIteratorStreamer(
                self.tokenizer, skip_prompt=True, skip_special_tokens=True
            )

            # Generation args
            generation_args = {
                "input_ids": inputs,
                "attention_mask": attention_mask,
                "streamer": streamer,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "min_p": min_p,
                "use_cache": True,
            }

            # Start generation in a separate thread
            thread = threading.Thread(target=self.model.generate, kwargs=generation_args)
            thread.start()

            # Return the streamer object
            return streamer
        else:
            # Generate without streaming
            outputs = self.model.generate(
                input_ids=inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                min_p=min_p,
                use_cache=True,
            )

            # Decode the output
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove the prompt from the response
            prompt_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            response = response[len(prompt_text) :].strip()

            return response

    def calculate_perplexity(self, prompt: str, answer: str, temperature: float = 0.0) -> float:
        """
        Calculate perplexity of the given answer for a prompt.

        Args:
            prompt: The input prompt
            answer: The answer to evaluate
            temperature: Sampling temperature

        Returns:
            float: Perplexity score (lower is better)
        """
        import math

        # Combine prompt and answer
        full_text = prompt + answer

        # Tokenize
        encodings = self.tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.model.device)
        target_ids = input_ids.clone()

        # Determine where the answer starts
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        prompt_length = prompt_ids.shape[1]

        # Set prompt part to -100 so it's ignored in loss calculation
        target_ids[:, :prompt_length] = -100

        # Calculate loss
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss.item()

        # Count tokens in answer
        answer_length = target_ids.shape[1] - prompt_length

        # Calculate perplexity: exp(average negative log-likelihood)
        perplexity = math.exp(neg_log_likelihood)

        return perplexity

    def calculate_answer_loss(self, prompt: str, answer: str) -> float:
        """
        Calculate the loss specifically on the answer portion of the text.

        Args:
            prompt: The input prompt
            answer: The answer to evaluate

        Returns:
            float: Loss value for the answer
        """
        # Combine prompt and answer
        full_text = prompt + answer

        # Tokenize
        encodings = self.tokenizer(full_text, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.model.device)
        target_ids = input_ids.clone()

        # Determine where the answer starts
        prompt_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
        prompt_length = prompt_ids.shape[1]

        # Set prompt part to -100 so it's ignored in loss calculation
        target_ids[:, :prompt_length] = -100

        # Calculate loss on answer only
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=target_ids)
            loss = outputs.loss.item()

        return loss

    def save_to_hub(self, hub_config: HubConfig, merge_adapter: bool = False):
        """
        Save model to Hugging Face Hub.

        Args:
            hub_config: Configuration for Hub saving
            merge_adapter: Whether to merge the adapter weights before saving

        Returns:
            str: URL of the saved model on the Hub
        """
        try:
            logger.info(f"Saving model to {hub_config.model_id}...")

            # Create repository if needed
            if hub_config.token:
                from huggingface_hub import create_repo

                try:
                    create_repo(
                        hub_config.model_id, private=hub_config.private, token=hub_config.token
                    )
                    logger.info(f"Created repository: {hub_config.model_id}")
                except Exception as e:
                    # Repository likely already exists
                    logger.info(f"Repository exists or couldn't be created: {str(e)}")

            # Save based on method
            if hub_config.save_method == "lora":
                # Save LoRA adapter only
                if hasattr(self.model, "peft_config"):
                    logger.info("Saving LoRA adapter...")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

                    # Save tokenizer
                    self.tokenizer.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )
                else:
                    logger.warning("Model doesn't have LoRA adapter, saving full model...")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

            elif hub_config.save_method == "merged_16bit":
                # Merge adapter and save in 16-bit
                if hasattr(self.model, "merge_and_unload"):
                    logger.info("Merging adapter and saving in 16-bit...")
                    merged_model = self.model.merge_and_unload()
                    merged_model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

                    # Save tokenizer
                    self.tokenizer.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )
                else:
                    logger.warning("Model doesn't support merge_and_unload, saving as is...")
                    self.model.save_pretrained(
                        hub_config.model_id, token=hub_config.token, push_to_hub=True
                    )

            elif hub_config.save_method == "merged_4bit":
                # Create optimized 4-bit model
                logger.info("Saving 4-bit quantized model is not fully supported yet")
                logger.info("Falling back to standard saving...")
                self.model.save_pretrained(
                    hub_config.model_id, token=hub_config.token, push_to_hub=True
                )

            elif hub_config.save_method == "gguf":
                logger.warning("GGUF export not yet supported, saving in standard format")
                self.model.save_pretrained(
                    hub_config.model_id, token=hub_config.token, push_to_hub=True
                )

            else:
                raise ValueError(f"Unsupported save method: {hub_config.save_method}")

            # Generate model URL
            hf_hub_url = f"https://huggingface.co/{hub_config.model_id}"
            logger.info(f"Model saved successfully to {hf_hub_url}")

            return hf_hub_url

        except Exception as e:
            logger.error(f"Error saving model to Hub: {str(e)}")
            raise

    def save_model(self, output_dir: str, save_method: str = "lora") -> str:
        """
        Save model to disk

        Args:
            output_dir: Directory to save the model
            save_method: Method to use for saving ("lora", "merged_16bit", "merged_4bit", "gguf")

        Returns:
            Path to saved model
        """
        os.makedirs(output_dir, exist_ok=True)

        if self.model_source == ModelSource.UNSLOTH:
            # Use Unsloth's saving methods
            if save_method == "lora":
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
            elif save_method == "merged_16bit":
                self.model.save_pretrained_merged(
                    output_dir, self.tokenizer, save_method="merged_16bit"
                )
            elif save_method == "merged_4bit":
                self.model.save_pretrained_merged(
                    output_dir, self.tokenizer, save_method="merged_4bit"
                )
            elif save_method == "gguf":
                self.model.save_pretrained_gguf(
                    output_dir, self.tokenizer, quantization_method="q4_k_m"
                )
            else:
                raise ValueError(f"Unknown save method: {save_method}")
        else:
            # Use Hugging Face's saving methods
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)

        print(f"Model saved to {output_dir} using method {save_method}")
        return output_dir

    def push_to_hub(self, hub_config: HubConfig) -> str:
        """
        Push model to Hugging Face Hub

        Args:
            hub_config: Configuration for pushing to HuggingFace Hub

        Returns:
            URL of the pushed model
        """
        if self.model_source == ModelSource.UNSLOTH:
            # Use Unsloth's hub methods
            if hub_config.save_method == "lora":
                self.model.push_to_hub_merged(
                    hub_config.model_id, self.tokenizer, save_method="lora", token=hub_config.token
                )
            elif hub_config.save_method == "merged_16bit":
                self.model.push_to_hub_merged(
                    hub_config.model_id,
                    self.tokenizer,
                    save_method="merged_16bit",
                    token=hub_config.token,
                )
            elif hub_config.save_method == "merged_4bit":
                self.model.push_to_hub_merged(
                    hub_config.model_id,
                    self.tokenizer,
                    save_method="merged_4bit",
                    token=hub_config.token,
                )
            elif hub_config.save_method == "gguf":
                self.model.push_to_hub_gguf(
                    hub_config.model_id,
                    self.tokenizer,
                    quantization_method=["q4_k_m", "q5_k_m"],
                    token=hub_config.token,
                )
            else:
                raise ValueError(f"Unknown save method: {hub_config.save_method}")
        else:
            # Use Hugging Face's hub methods
            self.model.push_to_hub(
                hub_config.model_id, token=hub_config.token, private=hub_config.private
            )
            self.tokenizer.push_to_hub(
                hub_config.model_id, token=hub_config.token, private=hub_config.private
            )

        hub_url = f"https://huggingface.co/{hub_config.model_id}"
        print(f"Model successfully pushed to: {hub_url}")
        return hub_url


class MCQGradioApp:
    """Gradio interface for the multiple choice question answering model"""

    def __init__(self, model_path="tuandunghcmut/Qwen25_Coder_MultipleChoice_v4"):
        """Initialize the application with model"""
        self.model_path = model_path
        self.model_handler = None
        self.prompt_creator = PromptCreator(prompt_type=PromptCreator.YAML_REASONING)
        self.response_parser = ResponseParser.from_prompt_type(self.prompt_creator.prompt_type)
        self.response_cache = {}  # Cache for model responses

        # Initialize the model (will be loaded on first use to save memory)
        self.load_model()

    def load_model(self):
        """Load the model from Hugging Face Hub or local checkpoint"""
        if self.model_handler is None:
            print(f"Loading model from {self.model_path}...")

            try:
                self.model_handler = QwenModelHandler(
                    model_name=self.model_path,
                    max_seq_length=2048,
                    # quantization=None,  # Disable quantization
                    device_map="auto",  # Automatically choose best device
                    # attn_implementation="flash_attention_2",  # Use flash attention for better performance
                    # force_attn_implementation=True,  # Force flash attention even if not optimal
                    model_source="huggingface",  # Use Unsloth's optimized model
                )
                # Set model to float16 after loading
                if self.model_handler.model is not None:
                    self.model_handler.model = self.model_handler.model.to(torch.float16)
                print("Model loaded successfully!")
            except Exception as e:
                print(f"Error loading model: {str(e)}")
                raise

    @spaces.gpu
    def inference(
        self,
        question,
        choices,
        temperature,
        max_new_tokens,
        top_p,
        top_k,
        repetition_penalty,
        do_sample,
    ):
        """Run inference with the model"""
        try:
            print("\n=== Debug: Inference Process ===")
            print(f"Input Question: {question}")
            print(f"Input Choices: {choices}")

            # Create cache key
            cache_key = f"{question}|{choices}|{temperature}|{max_new_tokens}|{top_p}|{top_k}|{repetition_penalty}|{do_sample}"
            print(f"Cache Key: {cache_key}")

            # Check cache first
            if cache_key in self.response_cache:
                print("Cache hit! Returning cached response")
                return self.response_cache[cache_key]

            # Create the prompt using the standard format from prompt_creator
            print("\nCreating prompt with PromptCreator...")
            prompt = self.prompt_creator.create_inference_prompt(question, choices)
            print(f"Generated Prompt:\n{prompt}")

            # Get model response using streaming generation
            print("\nStarting streaming generation...")
            response_chunks = []

            # Get streamer object
            streamer = self.model_handler.generate_with_streaming(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                do_sample=do_sample,
                min_p=0.1,  # Recommended value for better generation
                stream=True,
            )

            # Iterate through streaming chunks
            for chunk in streamer:
                if chunk:  # Only append non-empty chunks
                    response_chunks.append(chunk)
                    # Yield partial response for real-time display
                    partial_response = "".join(response_chunks)
                    # Format partial response for display
                    formatted_response = f"""Question: {question}

Choices:
{choices}

{partial_response}"""

                    # Yield to Gradio for display
                    yield prompt, formatted_response, "", ""

            # Combine all chunks for final response
            response = "".join(response_chunks)
            print(f"Complete Model Response:\n{response}")

            # Format the final response
            final_response = f"""Question: {question}

Choices:
{choices}

{response}"""

            # Parse YAML for structured display
            yaml_raw_display = f"```yaml\n{response}\n```"

            try:
                # Try to parse the YAML
                yaml_data = yaml.safe_load(response)
                yaml_json_display = f"```json\n{json.dumps(yaml_data, indent=2)}\n```"
            except Exception as e:
                print(f"Error parsing YAML: {e}")
                yaml_json_display = (
                    f"**Error parsing YAML:** {str(e)}\n\n**Raw Response:**\n```\n{response}\n```"
                )

            print("\nFinal Formatted Response:")
            print(final_response)

            result = (prompt, final_response, yaml_raw_display, yaml_json_display)

            # Cache the result
            self.response_cache[cache_key] = result
            print("\nCached result for future use")

            # Yield final response with structured YAML
            yield result

        except Exception as e:
            print(f"\nError during inference: {e}")
            # Format error response in YAML format
            error_response = f"""Question: {question}

Choices:
{choices}

understanding: |
  An error occurred during processing
analysis: |
  The system encountered an error while processing the request
reasoning: |
  {str(e)}
conclusion: |
  Please try again or contact support if the error persists
answer: X

Raw model output:
{response if 'response' in locals() else 'No response available'}"""
            yield prompt, error_response, "", ""

    def process_example(self, example_idx):
        """Process an example from the preset list"""
        if example_idx is None:
            return "", ""

        # Convert string index to integer if needed
        if isinstance(example_idx, str):
            try:
                # Extract the number from the string (e.g., "Example 13: ..." -> 13)
                example_idx = int(example_idx.split(":")[0].split()[-1]) - 1
            except (ValueError, IndexError) as e:
                print(f"Error converting example index: {e}")
                return "", ""

        try:
            if not isinstance(example_idx, int):
                print(f"Invalid example index type: {type(example_idx)}")
                return "", ""

            if example_idx < 0 or example_idx >= len(CODING_EXAMPLES):
                print(f"Example index out of range: {example_idx}")
                return "", ""

            example = CODING_EXAMPLES[example_idx]
            question = example["question"]
            choices = "\n".join(example["choices"])

            return question, choices

        except (ValueError, IndexError) as e:
            print(f"Error processing example: {e}")
            return "", ""

    def get_category_examples(self, category_name):
        """Get examples for a specific category"""
        if category_name == "All Categories":
            choices = [f"Example {i+1}: {ex['question']}" for i, ex in enumerate(CODING_EXAMPLES)]
        elif category_name in CODING_EXAMPLES_BY_CATEGORY:
            # Find the starting index for this category in the flattened list
            start_idx = 0
            for cat, examples in CODING_EXAMPLES_BY_CATEGORY.items():
                if cat == category_name:
                    break
                start_idx += len(examples)

            choices = [
                f"Example {start_idx+i+1}: {ex['question']}"
                for i, ex in enumerate(CODING_EXAMPLES_BY_CATEGORY[category_name])
            ]
        else:
            choices = []

        return gr.Dropdown(choices=choices, value=None, interactive=True)

    def create_interface(self):
        """Create the Gradio interface"""
        with gr.Blocks(title="Coding Multiple Choice Q&A with YAML Reasoning") as interface:
            gr.Markdown("# Coding Multiple Choice Q&A with YAML Reasoning")
            gr.Markdown(
                """
            This app uses a fine-tuned Qwen2.5-Coder-1.5B model to answer multiple-choice coding questions with structured YAML reasoning.

            The model breaks down its thought process in a structured way, providing:
            - Understanding of the question
            - Analysis of all options
            - Detailed reasoning process
            - Clear conclusion
            """
            )

            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown(
                        "### Examples (from the bank of 200 high-quality MCQs by Claude 3.7 Sonnet)"
                    )

                    # Category selector
                    category_dropdown = gr.Dropdown(
                        choices=["All Categories"] + list(CODING_EXAMPLES_BY_CATEGORY.keys()),
                        value="All Categories",
                        label="Select a category",
                    )

                    # Example selector
                    example_dropdown = gr.Dropdown(
                        choices=[
                            f"Example {i+1}: {q['question']}" for i, q in enumerate(CODING_EXAMPLES)
                        ],
                        label="Select an example question",
                        value=None,
                    )

                    gr.Markdown("### Your Question (or you can manually enter your input)")

                    # Question and choices inputs
                    question_input = gr.Textbox(
                        label="Question", lines=3, placeholder="Enter your coding question here..."
                    )
                    choices_input = gr.Textbox(
                        label="Choices (one per line)",
                        lines=4,
                        placeholder="Enter each choice on a new line, e.g.:\nOption A\nOption B\nOption C\nOption D",
                    )

                    # Parameters
                    temperature_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.001,
                        step=0.005,
                        label="Temperature (higher = more creative, lower = more deterministic)",
                    )

                    # Additional generation parameters
                    max_new_tokens_slider = gr.Slider(
                        minimum=128,
                        maximum=2048,
                        value=768,
                        step=128,
                        label="Max New Tokens (maximum length of generated response)",
                    )

                    top_p_slider = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                        label="Top-p (nucleus sampling probability)",
                    )

                    top_k_slider = gr.Slider(
                        minimum=1,
                        maximum=100,
                        value=80,
                        step=1,
                        label="Top-k (number of highest probability tokens to consider)",
                    )

                    repetition_penalty_slider = gr.Slider(
                        minimum=1.0,
                        maximum=2.0,
                        value=1.2,
                        step=0.1,
                        label="Repetition Penalty (higher = less repetition)",
                    )

                    do_sample_checkbox = gr.Checkbox(
                        value=True,
                        label="Enable Sampling (unchecked for greedy generation)",
                    )

                    # Submit button
                    submit_btn = gr.Button("Submit", variant="primary")

                with gr.Column(scale=3):
                    gr.Markdown("### Model Input")
                    prompt_display = gr.Textbox(
                        label="Prompt sent to model",
                        lines=8,
                        interactive=False,
                        show_copy_button=True,
                    )

                    gr.Markdown("### Model Streaming Response")
                    output = gr.Markdown(label="Response")

                    with gr.Accordion("Structured YAML Response", open=True):
                        gr.Markdown(
                            "Once the model completes its response, the YAML will be displayed here in a structured format."
                        )
                        yaml_raw = gr.Markdown(label="Raw YAML")
                        yaml_json = gr.Markdown(label="YAML as JSON")

            # Set up category selection
            category_dropdown.change(
                fn=self.get_category_examples,
                inputs=[category_dropdown],
                outputs=[example_dropdown],
            )

            # Set up example selection
            example_dropdown.change(
                fn=self.process_example,
                inputs=[example_dropdown],
                outputs=[question_input, choices_input],
            )

            # Update prompt display when question or choices change
            def update_prompt(question, choices):
                print("\n=== Debug: Prompt Update ===")
                print(f"Question Input: {question}")
                print(f"Choices Input: {choices}")

                if not question or not choices:
                    print("Empty question or choices, returning empty prompt")
                    return ""

                try:
                    print("\nCreating prompt with PromptCreator...")
                    prompt = self.prompt_creator.create_inference_prompt(question, choices)
                    print(f"Generated Prompt:\n{prompt}")
                    return prompt
                except Exception as e:
                    print(f"Error creating prompt: {e}")
                    return ""

            # Add prompt update on question/choices change
            question_input.change(
                fn=update_prompt, inputs=[question_input, choices_input], outputs=[prompt_display]
            )

            choices_input.change(
                fn=update_prompt, inputs=[question_input, choices_input], outputs=[prompt_display]
            )

            # Set up submission with loading indicator
            submit_btn.click(
                fn=self.inference,
                inputs=[
                    question_input,
                    choices_input,
                    temperature_slider,
                    max_new_tokens_slider,
                    top_p_slider,
                    top_k_slider,
                    repetition_penalty_slider,
                    do_sample_checkbox,
                ],
                outputs=[prompt_display, output, yaml_raw, yaml_json],
                show_progress=True,  # Show progress bar
                queue=True,  # Enable queueing for better handling of multiple requests
            )

        return interface


def main():
    """Main function to run the app"""
    app = MCQGradioApp()
    interface = app.create_interface()
    # Enable queueing at the app level
    interface.queue()
    interface.launch(share=True)


if __name__ == "__main__":
    main()

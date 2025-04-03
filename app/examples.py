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

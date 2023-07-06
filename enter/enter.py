#  In Python, you can use line continuation to break up long lines of code, including variable assignments and commands. 
#  There are a few ways you can do this: 

# 1. Implicit Line Continuation: Python allows line breaks between elements inside parentheses (), brackets [], and braces {}.

x = (1 + 2
     + 3 + 4)


# 2. Explicit Line Continuation: You can also use a backslash \ at the end of a line to indicate that the line should continue:

x = 1 + 2 \
    + 3 + 4

# 3. String concatenation: In Python, two string literals 
# (i.e., the ones enclosed between quotes) next to each other are automatically concatenated by the Python interpreter.

long_string = ("This is a very long string that "
               "is split over two lines.")


# Environment:
This project is write by python3 with numpy and opencv

# Simply run the project by:
    ~~~ python dartDetector.py dart0.jpg ~~~

# You can also add option values --si and/or --th
* --si is a switch of using opencv Hough function or self implemented Hough algorithm. follow by 0 or 1, default value is 0
    ... 0: means use opencv Hough function ...
    ... 1: means self implemented Hough algorithm ...
* --th is a threshold follow by an integer, default value is 5

# example:
    ~~~ python dartDetector.py dart0.jpg --si 1 --th 2 ~~~

** Note: The default value gives best result **
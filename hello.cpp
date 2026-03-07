// Comments can be used to explain C++ code, and to make it more readable. It can also be used to prevent execution when testing alternative code. 
// Comments can be singled-lined or multi-lined.
// Single-line comments start with two forward slashes (//).

/* Multi-line comments 
    start withand 
    ends with 
*/

// Note: Nested block comments are not allowed in C++.


// How to run on MAC:
// c++ -std=c++17 hello.cpp -o hello
// ./hello

// The following is a header file library that lets us work with input and output objects, such as cout (used in line 5).
// Header files add functionality to C++ programs.
#include <iostream>

//using namespace std means that we can use names for 
// objects and variables from the standard library.

// he using namespace std; statement can be omitted, and replaced with the std keyword followed by the :: operator, 
// for some objects (like std::cout in the example below):
using namespace std;


// It is important that you end the statement with a semicolon ;


// Another thing that always appear in a C++ program is int main(). 
// This is called a function. Any code inside its curly brackets {} will be executed.
int main() {

    //  cout (pronounced "see-out") is an object used together with the insertion operator (<<) to output/print text. 
    // In our example, it will output "Hello World!".
    
    cout << "Hello C++!"; 

    //You can also use cout() to print numbers.
    // However, unlike text, we don't put numbers inside double quotes:

    cout << 2024*25 + 1000;

    // To insert a new line in your output, you can use the \n character:
    cout << "\n Hello \n"; 
    cout << "Next line \n"; 

    //You can also use another << operator and place the \n character after the text, like this:

    cout << "Hello again!" << "\n"; 
    cout << "My name is " << "Pranav \n";

    // Another way to insert a new line, is with the endl manipulator:

    cout << "Hello again!" << endl;
    cout << "I am learning C++";

    // Other escape sequences:
    // \t : Creates a horizontal tab
    // \\ : Inserts a backslash character (\)
    // \" : Inserts a double quote character

    // In C++, there are different types of variables (defined with different keywords), for example:
    // int - stores integers (whole numbers), without decimals, such as 123 or -123
    // double - stores floating point numbers, with decimals, such as 19.99 or -19.99
    // char - stores single characters, such as 'a' or 'B'. Char values are surrounded by single quotes
    // string - stores text, such as "Hello World". String values are surrounded by double quotes
    // bool - stores values with two states: true or false

    // To create a variable, specify the type and assign it a value:

    int myNum = 8; 
    cout << myNum << "\n";

    // Note that if you assign a new value to an existing variable, it will overwrite the previous value:
    myNum = 15; 
    cout << myNum << "\n";

    // You can also declare a variable without assigning the value, and assign the value later:

    int myNum2; 
    muNum2 = 10; 
    cout << myNum2 << "\n";

    // Other types 

    int myNum = 5;               // Integer (whole number without decimals)
    double myFloatNum = 5.99;    // Floating point number (with decimals)
    char myLetter = 'D';         // Character
    string myText = "Hello";     // String (text)
    bool myBoolean = true;       // Boolean (true or false)









    return 0;
}

python3 -m pip install pybind11 setuptools wheel
python3 -m pip install -e .

from cppnn import NeuralNetwork
nn = NeuralNetwork(3, [4], 2)
print(nn.forward([1.0, 2.0, 3.0]))
print(len(nn.get_param()))


c++ -O3 -std=c++17 -shared -fPIC \
  $(python3 -m pybind11 --includes) \
  cppnn_module.cpp \
  -undefined dynamic_lookup \
  -o cppnn$(python3-config --extension-suffix)
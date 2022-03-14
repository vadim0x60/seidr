import psb2
from programlib import Program

from benchmark import build_prompt, task_descriptions, DATA_PATH

if __name__ == '__main__':
    examples, _ = psb2.fetch_examples(DATA_PATH, 'fizz-buzz', 2, 0, format='competitive')
    print(build_prompt('fizz-buzz', task_descriptions['fizz-buzz'], examples))

    fizzbuzz = Program("""
    #include <iostream>
    using namespace std;
    int main ()
    {
    int i;
    cin >> i;
    if(i % 3 == 0 && i % 5 == 0) {
    cout << "FizzBuzz" << endl;
    } else if(i % 3 == 0) {
    cout << "Fizz" << endl;
    } else if(i % 5 == 0) {
    cout << "Buzz" << endl;
    } else {
    cout << i << endl;
    }
    return 0;
    }
    """, language='C++')

    train_data, test_data = psb2.fetch_examples(DATA_PATH, 'fizz-buzz', 0, 2000, format='competitive')
    print(fizzbuzz.score(test_data))
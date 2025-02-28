import os
import pathlib
import logging
import jsonlines
import re

from black import format_str, FileMode
from typing import List, Any

TEST = {
    "python": {
        44: {
            5: \
                """def check(change_base):
                    assert change_base(7, 2) == "111"
                
                
                check(change_base)
                """,
            6: \
                """def check(change_base):
                    for x in range(2, 8):
                        assert change_base(x, x + 1) == str(x)
                
                
                check(change_base)
                """
        },
        53: {
            4:
                """def check(add):
                    import random
                
                    assert add(7, 5) == 12
                
                
                check(add)
                """,
            5:
                """def check(add):
                    import random
                
                    for i in range(100):
                        x, y = random.randint(0, 1000), random.randint(0, 1000)
                        assert add(x, y) == x + y
                
                check(add)"""
        }
    },
    "cpp": {
        44: {
            5:
                """#undef NDEBUG
#include<assert.h>
int main(){
    assert (change_base(7, 2) == "111");
}""",
            6:
                """#undef NDEBUG
#include<assert.h>
int main(){
    for (int x=2;x<8;x++)
    {
        assert (change_base(x, x + 1) == to_string(x));
    }
}"""
        },
        53: {
            4:
"""#undef NDEBUG
#include<assert.h>
int main(){
    assert (add(7, 5) == 12);
}""",
            5:
"""#undef NDEBUG
#include<assert.h>
int main(){
    for (int i=0;i<100;i+=1)
    {
        int x=rand()%1000;
        int y=rand()%1000;
        assert (add(x, y) == x + y);
    }
}"""
        }
    }

}


def load_jsonl(input_path: pathlib.Path or str) -> List[dict]:
    """
    Read list of objects from a json-lines file.
    """
    data = []
    with jsonlines.open(input_path, mode='r') as f:
        for line in f.iter():
            data.append(line)
    logging.info(f'Read {len(data)} records from {input_path}')
    return data


class ParseHumanEvalTests:
    """Parser for HumanEval tests"""
    def __init__(self, language: str, data: dict[str, Any]):
        self.language = language

        self.task_id = data["task_id"]
        self.test_code = data["test"]

    def get_test_start(self) -> str:
        """Find the starting point of the next test"""
        raise NotImplementedError

    def get_number_assert_words(self) -> int:
        """Count number of assertions"""
        return self.test_code.count(" assert ")

    def get_last_line(self) -> str:
        """Get the line where the test function is executed"""
        code = self.test_code.strip()
        lines = code.split("\n")
        return lines[-1]

    def split_tests(self) -> List[str]:
        """Parse one test with several assertions to a list of tests"""
        raise NotImplementedError


class ParseHumanEvalPythonTests(ParseHumanEvalTests):
    """Parser for HumanEval-Python tests"""
    def __init__(self, language: str, data: dict[str, Any]):
        super().__init__(language, data)

    def get_number_assert_words(self) -> int:
        return self.test_code.count(" assert ") - self.test_code.count(" this assert fails ")

    def get_test_start(self) -> str:
        return self.test_code[self.test_code.find("def "):self.test_code.find("assert")]

    def split_tests(self) -> List[str]:
        code = self.test_code
        code = code[code.find("def "):]
        header = self.get_test_start()
        assert_line_start_points = [m.start() for m in re.finditer("\n    assert", code)]
        assert_line_start_points += [m.start() for m in re.finditer("\n        assert", code)]
        assert_line_start_points = sorted(assert_line_start_points)
        num_asserts = self.get_number_assert_words()
        last_line = self.get_last_line()

        tests = []
        for i in range(len(assert_line_start_points)):
            start = assert_line_start_points[i]
            if i == len(assert_line_start_points) - 1:
                end = len(code)
                test = header + code[start:end]
            else:
                end = assert_line_start_points[i + 1]
                test = header + code[start:end] + "\n" + last_line

            try:
                test = format_str(test, mode=FileMode())
            except Exception as e:
                logging.error(f"\n\ntask_id: {self.task_id}\n\n")
                logging.error(test)
                logging.error(e)
            tests += [test]

        num_tests = len(tests)
        if num_tests != num_asserts:
            logging.error(
                f"{self.task_id}:\n"
                f"number of tests extracted: {num_tests}\n"
                f"number of assert word occurrences: {num_asserts}")
            logging.error(f"\ncode:\n{self.test_code}\n\ntests:")
            for num, t in enumerate(tests):
                logging.error(f"test {num} of {self.task_id}:\n{t}")

        return tests


class ParseHumanEvalCppTests(ParseHumanEvalTests):
    def __init__(self, language, data):
        super().__init__(language, data)

    def get_test_start(self) -> str:
        self.test_code = self.add_tabulation_to_assertion()
        return self.test_code[:self.test_code.find("\n    assert")]

    def add_tabulation_to_assertion(self) -> str:
        """Custom formatting for C++: add tabulation"""
        if self.task_id.split("/")[1] not in [str(i) for i in [32, 38, 44, 50, 53]]:
            code = re.sub(pattern="\\n\s+assert", repl="\n    assert", string=self.test_code)
            return code
        else:
            return self.test_code

    def split_tests(self) -> List[str]:
        code = self.add_tabulation_to_assertion()
        header = self.get_test_start()

        code = code[code.find("int main(){"):]

        assert_line_start_points = [m.start() for m in re.finditer("\n    assert", code)]
        num_asserts = self.get_number_assert_words()

        last_line = "}\n"

        tests = []
        for i in range(len(assert_line_start_points)):
            start = assert_line_start_points[i]

            if i == len(assert_line_start_points) - 1:
                end = len(code)
                test = header + code[start:end]
            else:
                end = assert_line_start_points[i + 1]
                test = header + code[start:end] + "\n" + last_line

            tests += [test]

        if len(tests) == 0:
            tests += [self.test_code]

        num_tests = len(tests)
        if num_tests != num_asserts:
            logging.error(
                f"{self.task_id}:\n"
                f"number of tests extracted: {num_tests}\n"
                f"number of assert word occurrences: {num_asserts}")
            logging.error(f"\ncode:\n{self.test_code}\n\ntests:")
            for num, t in enumerate(tests):
                logging.error(f"test {num} of {self.task_id}:\n{t}")

        return tests


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level="INFO")

    language = "cpp"
    language = language.lower()
    data = load_jsonl(pathlib.Path(os.getenv("DATA_PATH")) / "humaneval" / f"humaneval_{language}.jsonl")

    for i in range(len(data)):
        data_class = {
            "python": ParseHumanEvalPythonTests,
            "cpp": ParseHumanEvalCppTests
        }[language](language=language, data=data[i])
        tests = data_class.split_tests()
        data[i]["tests_split"] = tests

    data[44]["tests_split"][5] = TEST[language][44][5]
    try:
        data[44]["tests_split"][6] = TEST[language][44][6]
    except IndexError as e:
        data[44]["tests_split"] += [TEST[language][44][6]]

    data[53]["tests_split"][4] = TEST[language][53][4]

    try:
        data[53]["tests_split"][5] = TEST[language][53][5]
    except IndexError as e:
        data[53]["tests_split"] += [TEST[language][53][5]]

    fileout = pathlib.Path(os.getenv("DATA_PATH")) / "humaneval" / f"humaneval_{language}_split_tests.jsonl"
    with jsonlines.open(fileout, mode='w') as writer:
        writer.write_all(data)

    fileout = pathlib.Path(os.getenv("DATA_PATH")) / "humaneval" / f"humaneval_{language}_split_tests_only.jsonl"
    with jsonlines.open(fileout, mode='w') as writer:
        tests_only = [{"task_id": data[i]["task_id"], "tests_split": data[i]["tests_split"]} for i in range(len(data))]
        writer.write_all(tests_only)

    logging.info("End of script")

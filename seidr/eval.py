from abc import ABC
from programlib import Program

dont_change = 'Do not change anything'

class Evaluation(ABC):
    """
    A method for evaluating a system
    Produces a binary pass/fail result, a float score, and a text report
    """

    def __init__(self, SUT, passing_score=1):
        """
        SUT: System Under Test
        passing_score: float score required to pass the evaluation
        """

        self.SUT = SUT
        self.passing_score = passing_score

    def check(self) -> bool:
        """Produce a binary pass/fail evaluation"""
        return self.score() >= self.passing_score

    def score(self) -> float:
        """Produce a float score"""
        raise NotImplementedError

    def pen_report(self) -> str:
        """Produce a text report"""
        raise NotImplementedError

class IOMatch(Evaluation):
    def __init__(self, code, language, input, output, 
                 task_description=None):
        program = Program(code, language=language)
        super().__init__(program)
        self.input = input
        self.output = output
        self.test_run = None
        self.task_description = task_description

    def run_test(self, rerun=True):
        if rerun or not self.test_run:
            self.test_run = self.SUT.test([[self.input, self.output]])[0]

    def score(self) -> float:
        self.run_test(rerun=False)
        return self.SUT.avg_score

    def pen_report(self) -> str:
        self.run_test(rerun=False)

        if self.check():
            return dont_change
        elif self.test_run.exit_status:
            return '\n'.join(self.test_run.output_lines)
        else:
            input = '\n'.join(self.test_run.input_lines)
            expected_output = '\n'.join(self.test_run.expected_output_lines)
            output = '\n'.join(self.test_run.output_lines)
            return  f"it must return {expected_output} for input {input}, but it returns {output}. "
                    

class UnitTest(Evaluation):
    def __init__(self, code, language, test):
        program = Program(code + '\n' + test, language=language)
        super().__init__(program)
        self.output = None

    def run_test(self, rerun=True):
        if rerun or not self.output:
            if self.SUT.compile_error:
                self.output = self.SUT.term.emulate(self.SUT.stdout)
            else:
                self.output = self.SUT.run(force=True)

    def score(self) -> float:
        self.run_test(rerun=False)
        return 0.0 if self.SUT.exitstatus else 1.0
    
    def check(self) -> bool:
        self.run_test(rerun=False)
        return not self.SUT.exitstatus

    def pen_report(self) -> str:
        self.run_test(rerun=False)
        if self.check():
            return dont_change
        else:
             self.output = "\n".join(self.output) if type(self.output) == list else self.output
             return self.output

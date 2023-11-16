from abc import ABC
from programlib import Program
from seidr.prompt import write_debug_prompt, dont_change

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

    def check(self):
        """Produce a binary pass/fail evaluation"""
        return self.score() >= self.passing_score

    def score(self):
        """Produce a float score"""
        pass

    def pen_report(self):
        """Produce a text report"""
        pass

class IOMatch(Evaluation):
    def __init__(self, code, language, input, output, 
                 debug_template='Make sure {i} -> {o}', 
                 task_description=None):
        program = Program(code, language=language)
        super().__init__(program)
        self.input = input
        self.output = output
        self.test_run = None
        self.debug_template = debug_template
        self.task_description = task_description

    def run_test(self, rerun=True):
        if rerun or not self.test_run:
            self.test_run = self.SUT.test([[self.input, self.output]])[0]

    def score(self):
        self.run_test(rerun=False)
        return self.SUT.avg_score

    def pen_report(self):
        self.run_test(rerun=False)
        return write_debug_prompt(self.test_run, 
                                  self.debug_template, 
                                  self.task_description)

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

    def score(self):
        self.run_test(rerun=False)
        return not self.SUT.exitstatus

    def pen_report(self):
        self.run_test(rerun=False)
        if self.score():
            return dont_change
        else:
             self.output = "\n".join(self.output) if type(self.output) == list else self.output
             return self.output

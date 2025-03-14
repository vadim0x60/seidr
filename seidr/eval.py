from abc import ABC
from programlib import Program

dont_change = "Do not change anything"


class Evaluation(ABC):
    """
    A class for evaluating SEIDR performance by running tests on generated programs (problem solutions)
    Produces a binary pass/fail result, a float score, and a text report
    """

    def __init__(self, SUT, passing_score: float = 1.0):

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

    def run_test(self, rerun=True):
        """Test the program on a given I/O pair and update the score, errors or other output.
        Force test by default if `rerun=True`"""
        raise NotImplementedError


class IOMatch(Evaluation):
    def __init__(self, code, language, input, output, task_description=None):
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
            return "\n".join(self.test_run.output_lines)
        else:
            input = "\n".join(self.test_run.input_lines)
            expected_output = "\n".join(self.test_run.expected_output_lines)
            output = "\n".join(self.test_run.output_lines)
            return f"it must return {expected_output} for input {input}, but it returns {output}. "


class UnitTest(Evaluation):
    def __init__(self, code, language, test):
        program = Program(code + "\n" + test, language=language)
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
            self.output = (
                "\n".join(self.output) if type(self.output) is list else self.output
            )
            return self.output

class Gymnasium(Evaluation):
    def __init__(self, env, code, language, passing_score, error_reward=-1000):
        self.action_mode = type(env.action_space).__name__.lower()
        program = Program(code, language=language)
        super().__init__(program, passing_score)

        self.env = env
        self.tot_reward = 0
        self.tot_txt = ''
        self.done = False
        self.error_reward = error_reward

    def play(self):
        if self.done:
            return

        self.tot_reward = 0
        self.tot_txt = ''
        agent = self.SUT.spawn(action_mode=self.action_mode)

        try:
            observation, info = self.env.reset()
            self.tot_txt += info.get('memos', '')
            terminated = False
            truncated = False
            
            while not (terminated or truncated):
                if 'ascii' in self.env.metadata.get('render.modes', []):
                    ascii_render = self.env.render(mode='ascii')
                    self.tot_txt += ascii_render

                action, _ = agent.predict(observation, deterministic=True)

                observation, reward, terminated, truncated, info = self.env.step(action)
                self.tot_reward += reward
                self.tot_txt += info.get('memos', '')
        except RuntimeError as e:
            self.tot_reward = self.error_reward
            self.tot_txt += f'FATAL {e}'
        finally:
            agent.close()

        self.done = True

    def score(self):
        self.play()
        return self.tot_reward

    def pen_report(self):
        self.play()
        self.tot_txt += f'\nFinal reward: {self.tot_reward}'
        return self.tot_txt